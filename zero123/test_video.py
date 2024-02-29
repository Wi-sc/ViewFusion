import torch
from torch import autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from PIL import Image
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lpips
import os
import glob
import clip
from PIL import Image


# SAVE_DIR = "/home/ubuntu/workspace/zero123/zero123/experiments/pretrain/gen"
SAVE_DIR = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_360_autoregressive/gen_abo_blender_elevation60_inference_t0.50_auto_t0.10"

class GeometryDataset(Dataset):
    def __init__(self, clip_process):
        self.meta = []
        self.clip_process = clip_process
        with open('/home/ubuntu/data/abo/rendering_test_list_video_condition_id.json', "r") as f:
            model_dict = json.load(f)
        for model_name in model_dict:
        # for model_name in os.listdir(SAVE_DIR):
            # env_id = os.listdir(os.path.join(SAVE_DIR, model_name))[0]
            env_id = str(model_dict[model_name])
            frame_num = len(glob.glob(os.path.join(SAVE_DIR, model_name, env_id, "[0-9]*.png")))
            assert frame_num==36 or frame_num==16
            for inference_id in range(frame_num):
                target_id = (inference_id+1)%frame_num
                # inference_id = str(target_id)
                meta_dict = {
                    "model_name": model_name,
                    "inference_index": int(inference_id),
                    "target_index": int(target_id),
                    "inference_image_path": f"{SAVE_DIR}/{model_name}/{env_id}/{int(inference_id)}.png",
                    "target_image_path": f"{SAVE_DIR}/{model_name}/{env_id}/{int(target_id)}.png",
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
        image = Image.open(image_path).convert("RGB")
        image_filtered = self.transform(image)
        image_clip = self.clip_process(image)
        return image_filtered, image_clip
    
    def __getitem__(self, idx):
        meta_info = self.meta[idx]
        inference_image_path = meta_info['inference_image_path']
        target_image_path = meta_info['target_image_path']
        model_name = meta_info['model_name']
        inference_index = meta_info['inference_index']
        target_index = meta_info['target_index']
        # target_mask_path = meta_info['target_mask_path']

        generate_im, generate_clip = self.load_filtered_img(inference_image_path)

        target_im, target_clip = self.load_filtered_img(target_image_path)

        return {
            "gen_im":generate_im, 
            "target_im":target_im, 
            "gen_im_clip":generate_clip, 
            "target_im_clip":target_clip, 
            "model_name":model_name, 
            "target_index":target_index, 
            "inference_index":inference_index
        }

def calculate_sift_matching_points(image1, image2):
    # Initialize SIFT detector

    sift = cv2.SIFT_create()

    # Convert images to NumPy arrays
    image1 = image1.cpu().numpy().squeeze().astype(np.uint8)
    image2 = image2.cpu().numpy().squeeze().astype(np.uint8)

    # Find keypoints and descriptors in both images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    if descriptors1 is None or descriptors2 is None:
        print("Found descriptors None ")
        return 0
    # Initialize a Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    if len(matches)>0 and len(matches[0])>1:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # print(len(good_matches))
        # Return the number of matching points
        return len(good_matches)
    else:
        return 0

if __name__ == '__main__':

    
    # dataset = GeometryDataset(save_dir='/home/ubuntu/data/abo/zero123_renderings_elevation60')

    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    dataset = GeometryDataset(preprocess)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    ssim_list, psnr_list, lpips_list, clip_list = [], [], [], []
    model_list = {}
    # lpips = 0
    # psnr_metrics = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1,2,3))
    # ssim_metrics = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
    # lpips_metrics = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').cuda() # LPIPS needs the images to be in the [-1, 1] range.
    lpips_metrics = lpips.LPIPS(net='vgg').cuda()
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    for data in dataloader:
        # real_imgs, gen_imgs, model_name, env_index, target_index = data
        real_imgs = data['target_im']
        gen_imgs = data['gen_im']
        real_imgs = real_imgs.cuda()
        gen_imgs = gen_imgs.cuda()
        # print(gen_imgs.shape, real_imgs.shape)
        # psnr_batch = psnr_metrics(gen_imgs, real_imgs).cpu().tolist()
        # ssim_batch = ssim_metrics(gen_imgs, real_imgs).cpu().tolist()
        # psnr_list.extend(psnr_batch)
        # ssim_list.extend(ssim_batch)

        lpips_batch = lpips_metrics(gen_imgs*2-1, real_imgs*2-1).cpu().squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().tolist()
        lpips_list.extend(lpips_batch)
        
        gen_clip_features = clip_model.encode_image(data['gen_im_clip'].cuda())
        real_clip_features = clip_model.encode_image(data['target_im_clip'].cuda())
        # print(gen_clip_features.shape, real_clip_features.shape)
        cos_batch = cos_sim(gen_clip_features, real_clip_features).detach().cpu().tolist()
        clip_list.extend(cos_batch)

        image1 = torch.mean(real_imgs, dim=1, keepdim=True)*255
        image2 = torch.mean(gen_imgs, dim=1, keepdim=True)*255
        
            
        for i in range(gen_imgs.shape[0]):
            # print(data['model_name'][i], data['inference_index'][i])
            if data['model_name'][i] not in model_list:
                model_list[data['model_name'][i]] = {}
                # model_list[data['model_name'][i]]['ssim'] = [ssim_batch[i]]
                # model_list[data['model_name'][i]]['psnr'] = [psnr_batch[i]]
                model_list[data['model_name'][i]]['lpips'] = [lpips_list[i]]
                model_list[data['model_name'][i]]['clip'] = [clip_list[i]]
                model_list[data['model_name'][i]]['matching_points'] = [calculate_sift_matching_points(image1[i], image2[i])]
                model_list[data['model_name'][i]]['view_id'] = [data['inference_index'][i]]
            else:
                # model_list[data['model_name'][i]]['ssim'].append(ssim_batch[i])
                # model_list[data['model_name'][i]]['psnr'].append(psnr_batch[i])
                model_list[data['model_name'][i]]['lpips'].append(lpips_list[i])
                model_list[data['model_name'][i]]['clip'].append(clip_list[i])
                model_list[data['model_name'][i]]['matching_points'].append(calculate_sift_matching_points(image1[i], image2[i]))
                model_list[data['model_name'][i]]['view_id'].append(data['inference_index'][i])

        # real_imgs = real_imgs.cpu().permute(0, 2, 3, 1)
        # for i in range(real_imgs.shape[0]):
        #     Image.fromarray((255*real_imgs[i].numpy()).astype(np.uint8)).save(f"{SAVE_DIR}/gen/{data['model_name'][i]}/{data['inference_index'][i]}_target_white.png")
        
    # ssim = np.mean(ssim_list)
    # psnr = np.mean(psnr_list)
    # lpips = lpips/len(psnr_list)
    lpips_mean = np.mean(lpips_list)
    clip_mean = np.mean(clip_list)

    macthing_point_num_list = []
    for model_name in model_list:
        macthing_point_num_list.append(np.mean(model_list[model_name]['matching_points']))
    
    print(f"SIFT: {np.mean(macthing_point_num_list)}")
    # print(f"SSIM: {ssim}")
    # print(f"PSNR: {psnr}")
    print(f"LSIPS: {lpips_mean}")
    print(f"CLIP: {clip_mean}")

    # model_list = sorted(model_list, key=lambda x: -x[2])

    # save_text_path = "/".join(SAVE_DIR.split("/")[:-1])
    # with open(f"{save_text_path}/test_video_abo_blender.txt", "w") as f:
    with open(f"{SAVE_DIR}.txt", "w") as f:
        f.write(f"SIFT: {np.mean(macthing_point_num_list)}\n")
        # f.write(f"SSIM: {ssim}\n")
        # f.write(f"PSNR: {psnr}\n")
        f.write(f"LSIPS: {lpips_mean}\n")
        f.write(f"CLIP: {clip_mean}\n")
        for model_name in model_list:
            f.write(f"model-{model_name}-sift-{np.mean(model_list[model_name]['matching_points']):04f}-clip-{np.mean(model_list[model_name]['clip']):04f}-lpips-{np.mean(model_list[model_name]['lpips']):04f}\n")