import torch
from torch import autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lpips

# SAVE_DIR = "/home/ubuntu/workspace/zero123/zero123/experiments/pretrain/gen"
SAVE_DIR = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_autoregressive_syncdreamer/gen_gso_syncdreamer_inference_t0.50_auto_t1.00"

class GenDataset(Dataset):
    def __init__(self, save_dir='/home/ubuntu/data/GSO'):
        self.meta = []

        with open(f'{save_dir}/rendering_test_list_video_condition_id.json', "r") as f:
            model_dict = json.load(f)
        for model_name in model_dict:
            for inference_id in range(16):
                for target_id in range(16):
                    meta_dict = {
                        "model_name": model_name,
                        "inference_index": int(inference_id),
                        "target_index": int(target_id),
                        "inference_image_path": f"{SAVE_DIR}/{model_name}/{int(inference_id)}/{int(target_id)}",
                        "target_image_path": f"{SAVE_DIR}/{model_name}/{int(inference_id)}/{int(target_id)}",
                        "mkl_path": f"{save_dir}/gso_syncdreamer_random/{model_name}/meta.pkl",
                    }
                    self.meta.append(meta_dict)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.geometry_transform = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, mask_path=None, gray_im=False):
        if gray_im:
            image = Image.open(image_path).convert("1")
        else:
            image = Image.open(image_path).convert("RGB")
        if mask_path!=None:
            # mask = Image.open(mask_path).convert("1")
            im = Image.open(mask_path)
            mask = np.array(im)[:,:,3]
            mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_NEAREST)
            mask[mask>0]=1
            mask = Image.fromarray((mask*255).astype(np.uint8))
            image_filtered = Image.new("RGB", image.size, color=(255,255,255))
            image_filtered.paste(image, mask=mask)
        else:
            image_filtered = image
        image_filtered = self.transform(image_filtered)
        return image_filtered
    
    def __getitem__(self, idx):
        meta_info = self.meta[idx]
        inference_image_path = meta_info['inference_image_path']
        target_image_path = meta_info['target_image_path']
        model_name = meta_info['model_name']
        inference_index = meta_info['inference_index']
        target_index = meta_info['target_index']
        # target_mask_path = meta_info['target_mask_path']

        # angle_info = [meta_info['azimuth_offset'], meta_info['elevation_offset'], meta_info['distance_offset']]
        # angle_info[0] = angle_info[0]-2*np.pi if angle_info[0]>np.pi else angle_info[0]
        # angle_info = torch.tensor(angle_info)

        generate_im = self.load_filtered_img(target_image_path+"_generation.png")
        # generate_diffuse = self.load_filtered_img(target_image_path+"_gen_diffuse.png") * 2 - 1
        # generate_normal = self.load_filtered_img(target_image_path+"_gen_normal.png") * 2 - 1
        # generate_roughness = self.load_filtered_img(target_image_path+"_gen_roughness.png", gray_im=True) * 2 - 1
        # generate_specular = self.load_filtered_img(target_image_path+"_gen_specular.png", gray_im=True) * 2 - 1

        target_im = self.load_filtered_img(target_image_path+"_target.png")
        # target_diffuse = self.load_filtered_img(target_image_path+"_target_diffuse.png") * 2 - 1
        # target_normal = self.load_filtered_img(target_image_path+"_target_normal.png") * 2 - 1
        # target_roughness = self.load_filtered_img(target_image_path+"_target_roughness.png", gray_im=True) * 2 - 1
        # target_specular = self.load_filtered_img(target_image_path+"_target_specular.png", gray_im=True) * 2 - 1

        return {
            # "gen_diffuse":generate_diffuse, 
            # "gen_normal":generate_normal, 
            # "gen_roughness":generate_roughness, 
            # "gen_specular":generate_specular, 
            "gen_im":generate_im, 
            # "target_diffuse":target_diffuse, 
            # "target_normal":target_normal, 
            # "target_specular":target_specular, 
            # "target_roughness":target_roughness, 
            "target_im":target_im, 
            # "angle_info":angle_info, 
            "model_name":model_name, 
            "target_index":target_index, 
            "inference_index":inference_index
        }


if __name__ == '__main__':

    
    # dataset = GeometryDataset(save_dir='/home/ubuntu/data/abo/zero123_renderings_elevation60')
    dataset = GenDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    ssim_list, psnr_list, lpips_list = [], [], []
    model_list = []
    # lpips = 0
    psnr_metrics = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1,2,3))
    ssim_metrics = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
    # lpips_metrics = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').cuda() # LPIPS needs the images to be in the [-1, 1] range.
    lpips_metrics = lpips.LPIPS(net='vgg').cuda()
    for data in dataloader:
        # real_imgs, gen_imgs, model_name, env_index, target_index = data
        real_imgs = data['target_im']
        gen_imgs = data['gen_im']
        real_imgs = real_imgs.cuda()
        gen_imgs = gen_imgs.cuda()
        # print(gen_imgs.shape, real_imgs.shape)
        psnr_batch = psnr_metrics(gen_imgs, real_imgs).cpu().tolist()
        ssim_batch = ssim_metrics(gen_imgs, real_imgs).cpu().tolist()
        psnr_list.extend(psnr_batch)
        ssim_list.extend(ssim_batch)

        # lpips += lpips_metrics(gen_imgs*2-1, real_imgs*2-1).cpu().item() * gen_imgs.shape[0]
        lpips_batch = lpips_metrics(gen_imgs*2-1, real_imgs*2-1).cpu().squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().tolist()
        lpips_list.extend(lpips_batch)
        # print(psnr_list[:20])
        # print(ssim_list[:20])
        # print(lpips)

        for i in range(gen_imgs.shape[0]):
            model_list.append((data['model_name'][i], data['inference_index'][i], lpips_list[i]))

        # real_imgs = real_imgs.cpu().permute(0, 2, 3, 1)
        # for i in range(real_imgs.shape[0]):
        #     Image.fromarray((255*real_imgs[i].numpy()).astype(np.uint8)).save(f"{SAVE_DIR}/gen/{data['model_name'][i]}/{data['inference_index'][i]}_target_white.png")
        
    ssim = np.mean(ssim_list)
    psnr = np.mean(psnr_list)
    # lpips = lpips/len(psnr_list)
    lpips_mean = np.mean(lpips_list)
    print(f"SSIM: {ssim}")
    print(f"PSNR: {psnr}")
    print(f"LSIPS: {lpips_mean}")

    model_list = sorted(model_list, key=lambda x: -x[2])

    # save_text_path = "/".join(SAVE_DIR.split("/")[:-1])
    with open(f"{SAVE_DIR}.txt", "w") as f:
        f.write(f"SSIM: {ssim}\n")
        f.write(f"PSNR: {psnr}\n")
        f.write(f"LSIPS: {lpips_mean}\n")
        for i in range(1000):
            f.write(f"model-{model_list[i][0]}-inference_view-{model_list[i][1]}: {model_list[i][2]}\n")
