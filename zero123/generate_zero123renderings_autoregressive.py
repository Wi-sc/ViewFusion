import os
import diffusers  # 0.12.1
import math
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.my_ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
from ldm.models.brdf_renderer import Renderer
import cv2
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import glob

_GPU_INDEX = 0

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, ddim_steps, scale, ddim_eta, T, interpolation_weight=None):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            batch_size, _, h, w = input_im.shape
            c = model.get_learned_conditioning(input_im)
            T = T.unsqueeze(1)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(batch_size, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None,
                                             interpolation_weight=interpolation_weight)
            print(samples_ddim.shape)
            samples_ddim = samples_ddim.mean(dim=0, keepdim=True)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp(x_samples_ddim, min=-1., max=1.0)


class RenderingDataset(Dataset):
    def __init__(self, generated_dir, save_dir='/home/ubuntu/data/GSO/gso_renderings_elevation60', ):
        self.meta = []

        with open('/home/ubuntu/data/GSO/rendering_test_list_video_condition_id.json', "r") as f:
            model_dict = json.load(f)
        for model_name in model_dict:
            inference_id = model_dict[model_name]
            with open(f'{save_dir}/{model_name}/angles.json', "r") as f:
                angle_json = json.load(f)
            for target_id in range(12):
                if target_id==inference_id:
                    continue
                meta_dict = {
                    "model_name": model_name,
                    "inference_index": inference_id,
                    "target_index": target_id,
                    "inference_image_path": f"{save_dir}/{model_name}/{int(inference_id):03d}",
                    "target_image_path": f"{save_dir}/{model_name}/{int(target_id):03d}",
                    "inference_angle": angle_json[str(inference_id)]["angle"],
                    "target_angle": angle_json[str(target_id)]["angle"],
                    "azimuth_offset": (angle_json[str(target_id)]["angle"][0]-angle_json[str(inference_id)]["angle"][0])%(2*np.pi),
                    "elevation_offset": (angle_json[str(target_id)]["angle"][1]-angle_json[str(inference_id)]["angle"][1]),
                    "distance_offset": angle_json[str(target_id)]["angle"][2]-angle_json[str(inference_id)]["angle"][2]
                }
                self.meta.append(meta_dict)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.geometry_transform = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path):
        img = plt.imread(image_path + ".png")
        img[img[:, :, -1] == 0.] = [1., 1., 1., 1.]
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img = img.convert("RGB")
        img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        meta_info = self.meta[idx]
        inference_image_path = meta_info['inference_image_path']
        target_image_path = meta_info['target_image_path']
        model_name = meta_info['model_name']
        inference_index = meta_info['inference_index']
        target_index = meta_info['target_index']

        angle_info = [meta_info['elevation_offset'], meta_info['azimuth_offset'], meta_info['distance_offset']]
        angle_info[1] = angle_info[1]-2*np.pi if angle_info[1]>np.pi else angle_info[1]
        angle_info = torch.tensor(angle_info)

        target_im = self.load_filtered_img(target_image_path)
        cond_im = self.load_filtered_img(inference_image_path)

        cond_im = cond_im * 2 - 1
        target_im = target_im*2 - 1

        return cond_im, target_im, angle_info, model_name, target_index, inference_index




def main_run(models, device, dataset, inference_temp, auto_temp, output_dir, scale=3.0, ddim_steps=75, ddim_eta=1.0, precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    sampler = DDIMSampler(models['turncam'])
    
    for data in dataloader:
        cond_im, target_im, angle_info, model_name, target_index, inference_index = data
        batch_size = angle_info.shape[0]
        degree_steps = torch.max(torch.ceil(torch.abs(angle_info[0][1])/(np.pi/18)), torch.ceil(torch.abs(angle_info[0][0])/(np.pi/18)))
        degree_steps = int(degree_steps.item())
        print("angle_info:", angle_info/np.pi*180)
        print("degree_steps:", degree_steps)
        autoregressive_list = [(cond_im.to(device), np.array([0, 0, 0]))]

        # cond_im = cond_im.to(device)
        
        # if os.path.exists(f"{output_dir}/{model_name[0]}/") and len(glob.glob(f"{output_dir}/{model_name[0]}/*_generation.png"))==11:
        #     continue
        
        for view_id in range(degree_steps):
            rotation_cond_tmp = torch.zeros(len(autoregressive_list), 4)
            
            view_angle = np.array([angle_info[0][0]/degree_steps*(view_id+1), angle_info[0][1]/degree_steps*(view_id+1), angle_info[0][2]/degree_steps*(view_id+1)])

            anchor_view_angle_list = np.array([auto_reg[1] for auto_reg in autoregressive_list]).astype(np.float32)
            
            rotation_cond_angle = view_angle - anchor_view_angle_list
            # print(view_id, view_angle/np.pi*180, rotation_cond_angle/np.pi*180, anchor_view_angle_list)
            print("angle offset:", rotation_cond_angle)
            # interpolation_weight = np.where(np.abs(rotation_cond_angle)<np.pi, np.abs(rotation_cond_angle), 2*np.pi-np.abs(rotation_cond_angle))
            interpolation_weight = np.abs(rotation_cond_angle[:, 0]) + np.abs(rotation_cond_angle[:, 1])
            print("interpolation_weight:", interpolation_weight/np.pi*180)
            valid_index = interpolation_weight<125/180*np.pi
            valid_index[0] = 1
            print("valid_index:", valid_index)

            interpolation_weight = torch.from_numpy(interpolation_weight)[valid_index]
            if view_id>=1:
                inference_weight = np.exp(-interpolation_weight[0]/np.pi/inference_temp)
                # inference_weight = 0.5
                interpolation_weight[1:] = torch.nn.functional.softmax(-interpolation_weight[1:]/interpolation_weight[1:].max()/auto_temp)*(1-inference_weight)
                interpolation_weight[0] = inference_weight
            else:
                interpolation_weight[0] = 1
            print("interpolation_weight:", interpolation_weight)
            interpolation_weight = interpolation_weight.reshape(sum(valid_index), 1, 1, 1).to(device)
            cond_img_autoregressive = torch.cat([auto_reg[0] for auto_reg in autoregressive_list], dim=0).to(device)[valid_index]

            rotation_cond_tmp[:, 0] = torch.from_numpy(rotation_cond_angle[:, 0])
            rotation_cond_tmp[:, 1] = torch.sin(torch.from_numpy(rotation_cond_angle[:, 1]))
            rotation_cond_tmp[:, 2] = torch.cos(torch.from_numpy(rotation_cond_angle[:, 1]))
            rotation_cond_tmp[:, 3] = torch.from_numpy(rotation_cond_angle[:, 2])
            rotation_cond_tmp = rotation_cond_tmp[valid_index].to(device)

            x_samples_ddim_tmp = sample_model(cond_img_autoregressive, models['turncam'], sampler, precision, ddim_steps, scale, ddim_eta, rotation_cond_tmp, interpolation_weight=interpolation_weight)

            autoregressive_list.append((x_samples_ddim_tmp.clamp(-1, 1), view_angle))

        gen_target_im = 255.0 * rearrange(x_samples_ddim_tmp.cpu().numpy()/2+0.5, 'b c h w ->b h w c')
        inference_im_batch = 255.0 * rearrange(cond_im.cpu().numpy()/2 + 0.5, 'b c h w ->b h w c')
        target_im_batch = 255.0 * rearrange(target_im.numpy()/2 + 0.5, 'b c h w ->b h w c')
        if not os.path.exists(f"{output_dir}/{model_name[0]}/"):
            os.makedirs(f"{output_dir}/{model_name[0]}/")

        x_sample = gen_target_im[0]
        img = Image.fromarray(x_sample.astype(np.uint8))
        output_path = f"{output_dir}/{model_name[0]}/{inference_index[0]}_{target_index[0]}_generation.png"
        img.save(output_path)

        target_im = Image.fromarray(target_im_batch[0].astype(np.uint8))
        target_im.save(f"{output_dir}/{model_name[0]}/{inference_index[0]}_{target_index[0]}_target.png")
        
        inference_im = Image.fromarray(inference_im_batch[0].astype(np.uint8))
        inference_im.save(f"{output_dir}/{model_name[0]}/{inference_index[0]}_{target_index[0]}_condition.png")
        
        torch.cuda.empty_cache()
    return
if __name__ == '__main__':
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    mp.set_start_method('spawn')
    torch.set_num_threads(1)

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inference_temp', default=0.5, help='inference temperature', type=float)
    parser.add_argument('--auto_temp', default=1.0, help='autoregressive temperature', type=float)
    args = parser.parse_args()

    device_idx=_GPU_INDEX
    ckpt='/home/ubuntu/workspace/zero123/zero123/zero123-xl.ckpt'
    config='/home/ubuntu/workspace/zero123/zero123/configs/sd-swap_att-c_concat-256.yaml'

    # device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # models = dict()
    # print('Instantiating LatentDiffusion...')
    # models['turncam'] = load_model_from_config(config, ckpt, device=device)

    # main_run(models, device)

    inference_temp = args.inference_temp
    auto_temp = args.auto_temp
    output_dir = f'/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_autoregressive/gen_gso_blender_elevation60_inference_t{inference_temp:.2f}_auto_t{auto_temp:.2f}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/parameter.txt", "w") as f:
        f.write(f"inference_temp: {inference_temp}\n")
        f.write(f"auto_temp: {auto_temp}\n")

    processes = []
    dataset = RenderingDataset(output_dir)
    dataset_len = len(dataset)
    rank_num = 8
    subset_split = [dataset_len//rank_num for i in range(rank_num)]
    if sum(subset_split)!=dataset_len:
        subset_split[-1] += dataset_len-sum(subset_split)
    print(subset_split)
    dataset_list = torch.utils.data.random_split(dataset, subset_split)
    for rank in range(rank_num):
        device = f'cuda:{rank}'
        models = dict()
        models['turncam'] = load_model_from_config(config, ckpt, device=device)
        p = mp.Process(target=main_run, args=(models, device, dataset_list[rank], inference_temp, auto_temp, output_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()