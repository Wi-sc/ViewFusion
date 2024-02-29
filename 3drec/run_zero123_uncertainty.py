import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import kornia
import cv2
import math
import os
from torchvision import transforms
from torch import autocast
from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_blender, my_load_blender, my_load_abo_zero123_blender, my_load_real
from my3d import get_T, depth_smooth_loss
from contextlib import nullcontext

import sys
sys.path.append('../zero123/')
from ldm.models.diffusion.ddim import DDIMSampler
from diffusers import DDIMScheduler
from PIL import Image
import lpips
# # diet nerf
# import sys
# sys.path.append('./DietNeRF/dietnerf')
# from DietNeRF.dietnerf.run_nerf import get_embed_fn
# import torch.nn.functional as F

device_glb = torch.device("cuda")

def load_im(im_path):
    from PIL import Image
    import requests
    from torchvision import transforms
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp*2-1

def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }

@torch.no_grad()
def sample_model(input_im, model, T, sampler, precision='fp32', ddim_steps=75, scale=3.0, ddim_eta=1.0):
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
                                             x_T=None)
            return samples_ddim
        
class SJC(BaseConf):
    family:     str = "sd"
    sd:         SD = SD(
        variant="objaverse",
        scale=100.0
    )
    lr:         float = 0.01
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)

    emptiness_scale:    int = 10
    emptiness_weight:   float = 0.
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    depth_smooth_weight: float = 500
    near_view_weight: float = 500

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'chair'
    index:              int = 2

    view_weight:        int = 500
    prefix:             str = 'exp'
    nerf_path:          str = "data/nerf_wild"

    uncertainty_threshold: float = 0.02
    uncertainty_loss_step: int = 8000

    density_loss_weight: int = 100
    density_loss_step: int = 5000
    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)

def sjc_3d(poser, vox, model: ScoreAdapter,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, train_view, scene, index, view_weight, prefix, nerf_path, \
    depth_smooth_weight, near_view_weight, grad_accum, uncertainty_threshold, uncertainty_loss_step, density_loss_weight, density_loss_step, **kwargs):

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    vox.d_scale = torch.nn.Parameter(torch.tensor(3.0))
    vox.d_scale.requires_grad = False

    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    folder_name = prefix + '/scene-%s-index-%d_scale-%s_train-view-%s_view-weight-%s_depth-smooth-wt-%s_near-view-wt-%s' % \
                            (scene, index, model.scale, train_view, view_weight, depth_smooth_weight, near_view_weight)

    # load nerf view
    # images_, _, poses_, mask_, fov_x = load_blender('train', scene=scene, path=nerf_path)
    images_, K_, poses_, mask_, fov_x = my_load_abo_zero123_blender(model_name=scene, path=nerf_path)
    # images_, K_, poses_, mask_, fov_x = my_load_real(model_name=scene, path=nerf_path)
    # K_ = poser.get_K(H, W, fov_x * 180. / math.pi)
    K_ = poser.K
    print(poses_)
    print(poser.K)
    input_image, input_K, input_pose, input_mask = images_[index], K_, poses_[index], mask_[index]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cv2.imwrite(f'{folder_name}/input_view.jpg', cv2.cvtColor(input_image*255, cv2.COLOR_RGB2BGR))
    print(input_pose[:3, -1], np.linalg.norm(input_pose[:3, -1]))
    input_pose[:3, -1] = input_pose[:3, -1] / np.linalg.norm(input_pose[:3, -1]) * poser.R
    print(input_pose[:3, -1], np.linalg.norm(input_pose[:3, -1]))
    background_mask, image_mask = input_mask == 0., input_mask != 0.
    input_image = cv2.resize(input_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    image_mask = cv2.resize(image_mask.astype(np.float32), dsize=(256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    background_mask = cv2.resize(background_mask.astype(np.float32), dsize=(H, W), interpolation=cv2.INTER_NEAREST).astype(bool)

    # to torch tensor
    input_image = torch.as_tensor(input_image, dtype=float, device=device_glb)
    input_image = input_image.permute(2, 0, 1)[None, :, :]
    input_image = input_image * 2. - 1.
    image_mask = torch.as_tensor(image_mask, dtype=bool, device=device_glb)
    image_mask = image_mask[None, None, :, :].repeat(1, 3, 1, 1)
    background_mask = torch.as_tensor(background_mask, dtype=bool, device=device_glb)

    sampler = DDIMSampler(model.model)
    uncertainty_set_num = 3
    lpips_metrics = lpips.LPIPS(net='vgg').to(device_glb)



    num_train_timesteps = 1000
    scheduler = DDIMScheduler(
        num_train_timesteps,
        0.00085,
        0.0120,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    num_train_timesteps = scheduler.config.num_train_timesteps
    
    min_step_percent=0.02
    max_step_percent=0.98
    min_step = int(num_train_timesteps * min_step_percent)
    max_step = int(num_train_timesteps * max_step_percent)
    
    alphas = scheduler.alphas_cumprod.to(device_glb)
    
    grad_clip_val = None

    # prepare_embeddings(self.cfg.cond_image_path)

    t = torch.randint(
        min_step,
        max_step + 1,
        [bs],
        dtype=torch.long,
        device=device_glb,
    )
    print('==== loaded input view for training ====')

    opt.zero_grad()

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(folder_name) as metric:
        
        with torch.no_grad():

            tforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256))
            ])

            input_im = tforms(input_image)

            # get input embedding
            model.clip_emb = model.model.get_learned_conditioning(input_im.float()).tile(1,1,1).detach()
            model.vae_emb = model.model.encode_first_stage(input_im.float()).mode().detach()

        cond = {}
        cond['c_concat'] = [model.vae_emb.repeat(uncertainty_set_num, 1, 1, 1)]
        uc = {}
        uc['c_concat'] = [torch.zeros_like(cond['c_concat'][0]).to(device_glb)]
        uc['c_crossattn'] = [torch.zeros(uncertainty_set_num, 1, 768).to(device_glb)]
        
        rgb_loss_steps = 0
        for i in range(n_steps):
            if fuse.on_break():
                break
            
            return_density_loss = i>density_loss_step
            # if i==uncertainty_loss_step:
            #     vox.d_scale.requires_grad = False

            if train_view:

                # supervise with input view
                # if i < 100 or i % 10 == 0:
                with torch.enable_grad():
                    y_, depth_, ws_, density_loss_input = render_one_view(vox, aabb, H, W, input_K, input_pose, return_w=True, return_density_loss=return_density_loss)
                    y_ = model.decode(y_)
                
                if return_density_loss and (not torch.isnan(density_loss_input).any()):
                    density_loss_input = density_loss_input.mean() * (1. + density_loss_weight * i / n_steps)
                    print("===density_loss_input===",  density_loss_input)
                    density_loss_input.backward(retain_graph=True)

                rgb_loss = ((y_ - input_image) ** 2).mean()

                # depth smoothness loss
                input_smooth_loss = depth_smooth_loss(depth_) * depth_smooth_weight * 0.1
                input_smooth_loss.backward(retain_graph=True)

                input_loss = rgb_loss * float(view_weight)
                input_loss.backward(retain_graph=True)
                if train_view and i % 100 == 0:
                    metric.put_artifact("input_view", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(y_)[0]))

            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]

            y, depth, ws, density_loss = render_one_view(vox, aabb, H, W, Ks[i], poses[i], return_w=True, return_density_loss=return_density_loss)

            # near-by view
            eye = poses[i][:3, -1]
            near_eye = sample_near_eye(eye)
            near_pose = camera_pose(near_eye, -near_eye, poser.up)
            if return_density_loss:
                density_loss = density_loss.mean() * (1. + density_loss_weight * i / n_steps)
                print("===density_loss===",  density_loss)
                if not torch.isnan(density_loss):
                    density_loss.backward(retain_graph=True)
                else:
                    print("Found NaN and Skip")

            y_near, depth_near, ws_near, density_loss_near = render_one_view(vox, aabb, H, W, Ks[i], near_pose, return_w=True, return_density_loss=return_density_loss)
            near_loss = ((y_near - y).abs().mean() + (depth_near - depth).abs().mean()) * near_view_weight
            near_loss.backward(retain_graph=True)
            if return_density_loss:
                density_loss_near = density_loss_near.mean() * (1. + density_loss_weight * i / n_steps)
                if not torch.isnan(density_loss_near):
                    density_loss_near.backward(retain_graph=True)
                else:
                    print("Found NaN and Skip")

            # get T from input view
            pose = poses[i]
            T_target = pose[:3, -1]
            T_cond = input_pose[:3, -1]
            T = get_T(T_target, T_cond).to(model.device)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            cond['c_crossattn'] = [model.model.cc_projection(torch.cat([model.clip_emb, T[None, None, :]], dim=-1)).repeat(uncertainty_set_num, 1, 1)]
            

            if i%100==0 or i>uncertainty_loss_step:
                is_uncertainty_flag, generate_image_set = check_view_uncertainty(model.model, sampler, cond, uc, lpips_metrics, uncertainty_threshold, uncertainty_set_num=uncertainty_set_num)
            else:
                is_uncertainty_flag = True
                generate_image_set = None

            if is_uncertainty_flag:
                # with torch.no_grad():
                #     chosen_σs = np.random.choice(ts, bs, replace=False)
                #     chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                #     chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
                #     # chosen_σs = us[i]

                #     noise = torch.randn(bs, *y.shape[1:], device=model.device)

                #     zs = y + chosen_σs * noise

                #     score_conds = model.img_emb(input_im, conditioning_key='hybrid', T=T)

                #     Ds = model.denoise_objaverse(zs, chosen_σs, score_conds)

                #     if var_red:
                #         grad = -(Ds - y) / chosen_σs
                #     else:
                #         grad = -(Ds - zs) / chosen_σs
                    

                #     grad = grad.mean(0, keepdim=True)

                # y.backward(grad, retain_graph=True)

                # predict the noise residual with unet, NO grad!
                with autocast("cuda"):
                    with torch.no_grad():
                        # add noise
                        latents = y
                        noise = torch.randn_like(latents)  # TODO: use torch generator
                        latents_noisy = scheduler.add_noise(latents, noise, t)
                        # pred noise
                        x_in = torch.cat([latents_noisy] * 2)
                        t_in = torch.cat([t] * 2)
                        score_conds = model.img_emb(input_im, conditioning_key='hybrid', T=T)
                        noise_pred = model.model.apply_model(x_in, t_in, score_conds)

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + model.scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                w = (1 - alphas[t]).reshape(-1, 1, 1, 1)
                grad = w * (noise_pred - noise)
                grad = torch.nan_to_num(grad)
                # clip grad for stable training?
                if grad_clip_val is not None:
                    grad = grad.clamp(-grad_clip_val, grad_clip_val)

                # loss = SpecifyGradient.apply(latents, grad)
                # SpecifyGradient is not straghtforward, use a reparameterization trick instead
                target = (latents - grad).detach()
                # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
                loss_sds = 0.5 * torch.nn.functional.mse_loss(latents, target, reduction="sum") / bs
                loss_sds.backward(retain_graph=True)

            if not is_uncertainty_flag and generate_image_set is not None:
                generate_image_set = generate_image_set.mean(dim=0, keepdim=True).detach()
                nerf_img = model.decode(y).clamp(-1, 1)
                print("nerf_img", nerf_img.shape, nerf_img.min(), nerf_img.max())
                print("gen_img", generate_image_set.shape, generate_image_set.min(), generate_image_set.max())
                rgb_loss = ((nerf_img- generate_image_set) ** 2).mean()
                rgb_loss = rgb_loss * float(view_weight)
                rgb_loss.backward(retain_graph=True)

                rgb_loss_steps += 1
                print("rgb_loss_steps", rgb_loss_steps, 'uncertainty_percent', rgb_loss_steps/(i+1))

            if i%100==0:
                if not os.path.exists(f"{folder_name}/uncertainty_images"):
                    os.makedirs(f"{folder_name}/uncertainty_images", exist_ok=True)
                generate_image_set = 255.0 * rearrange(generate_image_set.mean(dim=0, keepdim=True).cpu().numpy()/2 + 0.5, 'b c h w ->b h w c')
                Image.fromarray(generate_image_set[0].astype(np.uint8)).save(f"{folder_name}/uncertainty_images/{i}_diffusion.png")

                nerf_img = 255.0 * rearrange(model.decode(y).detach().cpu().numpy()/2 + 0.5, 'b c h w ->b h w c')
                nerf_img = np.clip(nerf_img, 0, 255)
                Image.fromarray(nerf_img[0].astype(np.uint8)).save(f"{folder_name}/uncertainty_images/{i}_nerf.png")


            # emptiness_loss = (torch.log(1 + emptiness_scale * ws) * (-1 / 2 * ws)).mean() # negative emptiness loss
            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            # if emptiness_step * n_steps <= i:
            #     emptiness_loss *= emptiness_multiplier
            emptiness_loss = emptiness_loss * (1. + emptiness_multiplier * i / n_steps)
            emptiness_loss.backward(retain_graph=True)

            # depth smoothness loss
            smooth_loss = depth_smooth_loss(depth) * depth_smooth_weight

            if i >= emptiness_step * n_steps:
                smooth_loss.backward(retain_graph=True)

            depth_value = depth.clone()

            if i % grad_accum == (grad_accum-1):
                opt.step()
                opt.zero_grad()

            metric.put_scalars(**tsr_stats(y))

            if i % 1000 == 0 and i != 0:
                with EventStorage(model.im_path.replace('/', '-') + '_scale-' + str(model.scale) + "_test"):
                    evaluate(model, vox, poser)

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth_value)

            metric.step()
            pbar.update()
            pbar.set_description(model.im_path)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser, folder_name=folder_name)

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, vox, poser, folder_name=None):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    if folder_name is not None:
        for threshold_mult in range(2, 22, 2):
            try:
                vox.export_mesh(f"{folder_name}/model_m{threshold_mult}.obj", threshold_mult=threshold_mult)
            except:
                print(threshold_mult, "Error for export mesh")
                    
    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False, return_density_loss=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    if return_density_loss:
        rgbs, depth, weights, density_loss = render_ray_bundle(vox, ro, rd, t_min, t_max, return_density_loss=return_density_loss)
    else:
        rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max, return_density_loss=return_density_loss)
        density_loss = 0
    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    weights = rearrange(weights, "N (h w) 1 -> N h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights, density_loss
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    # y = torch.nn.functional.interpolate(y, 512, mode='bilinear', antialias=True)
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    # cfg = optional_load_config(fname="full_config_objaverse.yml")
    cfg = optional_load_config()
    cfg = SJC(**cfg).dict()
    from my.config import argparse_cfg_template
    cfg = argparse_cfg_template(cfg)  # cmdline takes priority
    
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)
    print(mod)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()
    print(model.prompt)
    pbar = tqdm(range(1))

    scene = cfg.pop('scene')
    index = cfg.pop('index')
    train_view = cfg.pop('train_view')
    view_weight = cfg.pop('view_weight')
    depth_smooth_weight = cfg.pop('depth_smooth_weight')
    near_view_weight = cfg.pop('near_view_weight')
    prefix = cfg.pop('prefix')
    folder_name = prefix + '/scene-%s-index-%d_scale-%s_train-view-%s_view-weight-%s_depth-smooth-wt-%s_near-view-wt-%s' % \
                            (scene, index, model.scale, train_view, view_weight, depth_smooth_weight, near_view_weight)
    print(folder_name)
    print(os.listdir(folder_name))
    with EventStorage(folder_name), HeartBeat(pbar):
        ckpt_fname = latest_ckpt(folder_name)
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage(folder_name):
            evaluate(model, vox, poser, folder_name)


def latest_ckpt(folder_name):
    ts, ys = read_stats(folder_name, "ckpt")
    assert len(ys) > 0
    return ys[-1]

def check_view_uncertainty(model, sampler, cond, uc, lpips_metrics, threshold, precision='fp32', ddim_steps=75, scale=3.0, ddim_eta=1.0, uncertainty_set_num=6):
    shape = [4, 32, 32]
    with nullcontext('cuda'):
        with model.ema_scope():
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=cond,
                                                    batch_size=uncertainty_set_num,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta,
                                                    x_T=None)
            # generate_features = samples_ddim.detach()
            generate_multi_images = model.decode_first_stage(samples_ddim).detach()
    generate_multi_images = torch.clamp((generate_multi_images), min=-1.0, max=1.0)
    generate_multi_images_a = generate_multi_images.repeat(uncertainty_set_num-1, 1, 1, 1)

    index_shift_list = []
    for index_shift in range(1, uncertainty_set_num):
        index_shift_list.append([i%uncertainty_set_num for i in range(index_shift, index_shift+uncertainty_set_num)])
        
    generate_multi_images_b = torch.cat([generate_multi_images[index_shift] for index_shift in index_shift_list], dim=0)

    lpips_metrics = lpips_metrics(generate_multi_images_a, generate_multi_images_b).squeeze().detach().cpu()
    uncertainty_score = torch.median(lpips_metrics)

    print("lpips_metrics max, min, median", torch.max(lpips_metrics), torch.min(lpips_metrics), uncertainty_score)
    return uncertainty_score>threshold, generate_multi_images

if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
