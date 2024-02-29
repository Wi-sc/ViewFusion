from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import glob

# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class ABODataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = GeometryData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = GeometryData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(GeometryData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = 12
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class ObjaverseDataSwap(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        n_view=8,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        self.n_view = n_view
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = 12

        target_im_list = []
        cond_im_list = []
        T_list = []
        for _ in range(self.n_view):
            index_target, index_cond = random.sample(range(total_view), 2) # without replacement
            filename = os.path.join(self.root_dir, self.paths[index])

            # print(self.paths[index])

            if self.return_paths:
                data["path"] = str(filename)
            
            color = [1., 1., 1., 1.]

            try:
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            except:
                # very hacky solution, sorry about this
                filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
                target_im = torch.zeros_like(target_im)
                cond_im = torch.zeros_like(cond_im)
                
            target_im_list.append(target_im)
            cond_im_list.append(cond_im)
            T_list.append(self.get_T(target_RT, cond_RT))

        data["image_target"] = torch.cat([target_im_list], dim=0)
        data["image_cond"] = torch.cat([cond_im_list], dim=0)
        data["T"] = torch.cat([T_list], dim=0)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    
class ABOData(Dataset):
    def __init__(self,
        root_dir="/home/ec2-user/data/abo/render",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=90,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        # self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        # if not isinstance(ext, (tuple, list, ListConfig)):
        #     ext = [ext]

        # with open(os.path.join(root_dir, 'valid_paths.json')) as f:
        #     self.paths = json.load(f)
            
        self.meta = []
        split = pd.read_csv(f'{root_dir}/train_test_split.csv')
        if validation:
            split = split[split['SPLIT'] == 'TEST']['MODEL']
        else:
            split = split[split['SPLIT'] == 'TRAIN']['MODEL']
        for model_name in split:
            for env_index in os.listdir(f"{self.root_dir}/{model_name}/render"):
                print(model_name, env_index)
                meta_dict = {
                    "model_name": model_name,
                    "env_index": env_index,
                }
                self.meta.append(meta_dict)
        self.transform = image_transforms

    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, mask_path):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")
        image_filtered = Image.new("RGB", image.size, color=(255,255,255))
        image_filtered.paste(image, mask=mask)
        image_filtered = self.transform(image_filtered)
        return image_filtered

    def __getitem__(self, index):
        model_name = self.meta[index]["model_name"]
        env_index = self.meta[index]["env_index"]
        with open(f'/home/ec2-user/data/abo/angle/{model_name}.json', "r") as f:
            angle_json = json.load(f)
        data = {}
        total_view = 12
        inference_id, target_id = random.sample(range(total_view), 2) # without replacement

        inference_image_path = f"{self.root_dir}/{model_name}/render/{env_index}/render_{inference_id}.jpg"
        inference_mask_path = f"{self.root_dir}/{model_name}/segmentation/segmentation_{inference_id}.jpg"
        target_image_path = f"{self.root_dir}/{model_name}/render/{env_index}/render_{target_id}.jpg"
        target_mask_path = f"{self.root_dir}/{model_name}/segmentation/segmentation_{target_id}.jpg"

        inference_id = str(inference_id)
        target_id = str(target_id)
        azimuth_offset = (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi)
        elevation_offset = angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]
        distance_offset = angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
        
        if self.return_paths:
            data["path"] = str(inference_image_path)

        
        target_im = self.load_filtered_img(target_image_path, target_mask_path)
        cond_im = self.load_filtered_img(inference_image_path, inference_mask_path)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = torch.tensor([elevation_offset, math.sin(azimuth_offset), math.cos(azimuth_offset), distance_offset])

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

class Zero123Data(Dataset):
    def __init__(self,
        root_dir="/home/ec2-user/data/abo/",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        # self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        # if not isinstance(ext, (tuple, list, ListConfig)):
        #     ext = [ext]

        # with open(os.path.join(root_dir, 'valid_paths.json')) as f:
        #     self.paths = json.load(f)
            
        self.meta = []
        split = pd.read_csv(f'{root_dir}/render/train_test_split.csv')
        if validation:
            split = split[split['SPLIT'] == 'TEST']['MODEL']
        else:
            split = split[split['SPLIT'] == 'TRAIN']['MODEL']
        for model_name in split:
            for view_id in range(90):
                if os.path.exists(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, f"{view_id:03d}_brdf_rgb_white.png")):
                    meta_dict = {
                        "model_name": model_name,
                        "view_id": view_id,
                    }
                    self.meta.append(meta_dict)
        self.transform = image_transforms
        if validation:
            print("validation data number", len(self.meta))
        else:
            print("train data number", len(self.meta))

    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, color=[1., 1., 1., 1.]):
        # img = plt.imread(image_path)
        # img[img[:, :, -1] == 0.] = color
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img = img.convert("RGB")
        # img = self.transform(img)
        # return img

        img = plt.imread(image_path)
        img_white_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img_white_background = img_white_background.convert("RGB")
        img_white_background = self.transform(img_white_background)
        return img_white_background

    def __getitem__(self, index):
        model_name = self.meta[index]["model_name"]
        inference_id = self.meta[index]["view_id"]
        with open(f'{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/angles.json', "r") as f:
            angle_json = json.load(f)
        data = {}

        total_view = 90
        # inference_id, target_id = random.sample(range(total_view), 2) # without replacement
        target_id = random.sample(range(total_view), 1)[0]
        while inference_id==target_id:
            target_id = random.sample(range(total_view), 1)[0]
        
        inference_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{inference_id:03d}_brdf_rgb_white.png"
        target_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}_brdf_rgb_white.png"

        inference_id = str(inference_id)
        target_id = str(target_id)
        azimuth_offset = (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi)
        elevation_offset = angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]
        distance_offset = angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
        
        if self.return_paths:
            data["path"] = str(inference_image_path)

        
        target_im = self.load_filtered_img(target_image_path)
        cond_im = self.load_filtered_img(inference_image_path)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = torch.tensor([elevation_offset, math.sin(azimuth_offset), math.cos(azimuth_offset), distance_offset])

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

class GeometryData(Dataset):
    def __init__(self,
        root_dir="/home/ubuntu/data/abo/",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=90,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        # self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
            
        self.meta = []
        split = pd.read_csv(f'{root_dir}/render/train_test_split.csv')
        if validation:
            split = split[split['SPLIT'] == 'TEST']['MODEL']
        else:
            split = split[split['SPLIT'] == 'TRAIN']['MODEL']
        for model_name in split:
            # for view_id in range(len(glob.glob(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, "*_brdf_rgb.png")))):
            for view_id in range(90):
                if os.path.exists(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, f"{view_id:03d}_brdf_rgb.png")):
                    meta_dict = {
                        "model_name": model_name,
                        "view_id": view_id,
                    }
                    self.meta.append(meta_dict)
        self.transform = image_transforms
        self.geometry_transform = torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        if validation:
            print("validation data number", len(self.meta))
        else:
            print("train data number", len(self.meta))

    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, color=[1., 1., 1.]):
        # img = plt.imread(image_path + "_brdf_rgb.png")
        # img_black_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img_black_background = img_black_background.convert("RGB")
        # img_black_background = self.transform(img_black_background)

        # mask = plt.imread(image_path + "_rgb.png")
        # img[mask[:, :, -1] == 0.] = color
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img = img.convert("RGB")
        # img = self.transform(img)
        # return img.permute(2,0,1), img_black_background.permute(2,0,1)

        img = plt.imread(image_path + "_brdf_rgb_white.png")
        img_white_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img_white_background = img_white_background.convert("RGB")
        img_white_background = self.transform(img_white_background)
        return img_white_background.permute(2,0,1)

    def load_geometry_maps(self, target_image_path):
        h, w = 512, 512
        target_mask_path = target_image_path + "_rgb.png"
        target_albedo_path = target_image_path + "_albedo0001.png"
        target_normal_path = target_image_path + "_normal.png"
        target_roughness_path = target_image_path + "_roughness.png"

        mask = cv2.imread(target_mask_path, cv2.IMREAD_UNCHANGED)/255
        mask = mask[:,:,3:]
        mask[mask>0.1] = 1
        mask = np.concatenate([mask, mask, mask], axis=2)

        diffuse = cv2.imread(target_albedo_path, cv2.IMREAD_UNCHANGED)
        diffuse = diffuse[:,:,:3]
        diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB)
        diffuse = diffuse*mask + (1-mask)*255
        diffuse = diffuse.transpose(2,0,1)/255
        diffuse = torch.from_numpy(diffuse)
        diffuse = self.geometry_transform(diffuse)
        diffuse = diffuse.clamp(0,1)

        specular = torch.ones(h,w,1)*0.5
        specular = specular*torch.from_numpy(mask[:,:,0:1])
        specular = specular.permute(2,0,1)
        specular = self.geometry_transform(specular)
        # specular = specular.expand(3, -1, -1).float()

        roughness = cv2.imread(target_roughness_path, cv2.IMREAD_UNCHANGED)
        roughness = cv2.cvtColor(roughness, cv2.COLOR_BGRA2GRAY)[:,:,np.newaxis]
        roughness = roughness[...,:1]*mask[:,:,0:1]
        roughness = roughness.transpose(2,0,1)/255
        roughness = torch.from_numpy(roughness)
        roughness = self.geometry_transform(roughness)
        # roughness = roughness.expand(3, -1, -1).float()

        normal = cv2.imread(target_normal_path, cv2.IMREAD_UNCHANGED)
        normal = normal[...,:3]
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal_map_x = normal[:, :, 0]/127.5-1
        normal_map_y = (normal[:, :, 1]/127.5-1)
        normal_map_z = (normal[:, :, 2]/127.5-1)
        normal = np.stack([normal_map_x, normal_map_y, normal_map_z], axis=-1)
        background_normal = np.zeros((h,w,3))
        background_normal[:,:,2] = 1
        normal = normal*mask + (1-mask)*background_normal
        normal = normal / np.sqrt(np.sum(normal**2, axis=-1, keepdims=True))
        normal = normal.transpose(2,0,1)
        normal = torch.from_numpy(normal)[:3,:,:]
        normal = self.geometry_transform(normal)

        return diffuse*2-1, normal, roughness*2-1, specular*2-1
    
    def __getitem__(self, index):
        model_name = self.meta[index]["model_name"]
        inference_id = self.meta[index]["view_id"]
        target_id = inference_id
        # with open(f'{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/angles.json', "r") as f:
        #     angle_json = json.load(f)
        data = {}

        # total_view = self.total_view 
        # target_id = random.sample(range(total_view), 1)[0]
        # while inference_id==target_id:
        #     target_id = random.sample(range(total_view), 1)[0]
        
        # inference_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{inference_id:03d}"
        target_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}"
        # target_albedo_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}_albedo0001.png"
        # target_normal_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}_normal.png"
        # target_roughness_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}_roughness.png"

        inference_id = str(inference_id)
        target_id = str(target_id)
        # azimuth_offset = (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi)
        # elevation_offset = angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]
        # distance_offset = angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
        
        if self.return_paths:
            # data["path"] = str(inference_image_path)
            data["path"] = str(target_image_path)

        # target_im, target_im_black = self.load_filtered_img(target_image_path)
        # cond_im, _ = self.load_filtered_img(inference_image_path)
        target_im = self.load_filtered_img(target_image_path)
        target_diffuse, target_normal, target_roughness, target_specular = self.load_geometry_maps(target_image_path)

        data["brdf_maps"] = torch.cat([target_diffuse, target_normal, target_roughness, target_specular], dim=0)
        data["rgb"] = target_im
        # data["T"] = torch.tensor([elevation_offset, math.sin(azimuth_offset), math.cos(azimuth_offset), distance_offset])

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

class MaskingGeometryData(Dataset):
    def __init__(self,
        root_dir="/home/ubuntu/data/abo/",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=90,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        # self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.brdf_clip_signals = {}
        for map in ["Albedo", "Normal", "Roughness", "Specular"]:
            self.brdf_clip_signals[map] = torch.from_numpy(np.load(f"/home/ubuntu/workspace/zero123/{map}.npy"))

        self.meta = []
        split = pd.read_csv(f'{root_dir}/render/train_test_split.csv')
        if validation:
            split = split[split['SPLIT'] == 'TEST']['MODEL']
        else:
            split = split[split['SPLIT'] == 'TRAIN']['MODEL']
        for model_name in split:
            # for view_id in range(len(glob.glob(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, "*_brdf_rgb.png")))):
            for view_id in range(90):
                if os.path.exists(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, f"{view_id:03d}_brdf_rgb.png")):
                    meta_dict = {
                        "model_name": model_name,
                        "view_id": view_id,
                    }
                    self.meta.append(meta_dict)
        self.transform = image_transforms
        self.geometry_transform = torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        if validation:
            print("validation data number", len(self.meta))
        else:
            print("train data number", len(self.meta))

    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, color=[1., 1., 1.]):
        # img = plt.imread(image_path + "_brdf_rgb.png")
        # img_black_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img_black_background = img_black_background.convert("RGB")
        # img_black_background = self.transform(img_black_background)

        # mask = plt.imread(image_path + "_rgb.png")
        # img[mask[:, :, -1] == 0.] = color
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img = img.convert("RGB")
        # img = self.transform(img)
        # return img.permute(2,0,1), img_black_background.permute(2,0,1)

        img = plt.imread(image_path + "_brdf_rgb_white.png")
        img_white_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img_white_background = img_white_background.convert("RGB")
        img_white_background = self.transform(img_white_background)
        return img_white_background

    def load_geometry_maps(self, target_image_path, geometry_flag):
        h, w = 512, 512
        target_mask_path = target_image_path + "_rgb.png"
        target_albedo_path = target_image_path + "_albedo0001.png"
        target_normal_path = target_image_path + "_normal.png"
        target_roughness_path = target_image_path + "_roughness.png"

        mask = cv2.imread(target_mask_path, cv2.IMREAD_UNCHANGED)/255
        mask = mask[:,:,3:]
        mask[mask>0.1] = 1
        mask = np.concatenate([mask, mask, mask], axis=2)

        if geometry_flag == 0:
            diffuse = cv2.imread(target_albedo_path, cv2.IMREAD_UNCHANGED)
            diffuse = diffuse[:,:,:3]
            diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB)
            diffuse = diffuse*mask + (1-mask)*255
            diffuse = diffuse.transpose(2,0,1)/255
            diffuse = torch.from_numpy(diffuse)
            diffuse = self.geometry_transform(diffuse)
            diffuse = diffuse.clamp(0,1)
            return diffuse*2-1
        elif geometry_flag == 1:
            normal = cv2.imread(target_normal_path, cv2.IMREAD_UNCHANGED)
            normal = normal[...,:3]
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal_map_x = normal[:, :, 0]/127.5-1
            normal_map_y = (normal[:, :, 1]/127.5-1)
            normal_map_z = (normal[:, :, 2]/127.5-1)
            normal = np.stack([normal_map_x, normal_map_y, normal_map_z], axis=-1)
            background_normal = np.zeros((h,w,3))
            background_normal[:,:,2] = 1
            normal = normal*mask + (1-mask)*background_normal
            normal = normal / np.sqrt(np.sum(normal**2, axis=-1, keepdims=True))
            normal = normal.transpose(2,0,1)
            normal = torch.from_numpy(normal)[:3,:,:]
            normal = self.geometry_transform(normal)
            return normal
        elif geometry_flag == 2:
            roughness = cv2.imread(target_roughness_path, cv2.IMREAD_UNCHANGED)
            roughness = cv2.cvtColor(roughness, cv2.COLOR_BGRA2GRAY)[:,:,np.newaxis]
            roughness = roughness[...,:1]*mask[:,:,0:1]
            roughness = roughness.transpose(2,0,1)/255
            roughness = torch.from_numpy(roughness)
            roughness = self.geometry_transform(roughness)
            roughness = roughness.expand(3, -1, -1)
            return roughness*2-1
        else:
            specular = torch.ones(h,w,1)*0.5
            specular = specular*torch.from_numpy(mask[:,:,0:1])
            specular = specular.permute(2,0,1)
            specular = self.geometry_transform(specular)
            specular = specular.expand(3, -1, -1)
            return specular*2-1
    
    def __getitem__(self, index):
        model_name = self.meta[index]["model_name"]
        inference_id = self.meta[index]["view_id"]
        total_view = 90
        target_id = random.sample(range(total_view), 1)[0]
        while inference_id==target_id:
            target_id = random.sample(range(total_view), 1)[0]
        with open(f'{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/angles.json', "r") as f:
            angle_json = json.load(f)
        data = {}

        inference_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{inference_id:03d}"
        target_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/{target_id:03d}"

        inference_id = str(inference_id)
        target_id = str(target_id)
        azimuth_offset = (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi)
        elevation_offset = angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]
        distance_offset = angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
        
        if self.return_paths:
            # data["path"] = str(inference_image_path)
            data["path"] = str(target_image_path)

        # target_im, target_im_black = self.load_filtered_img(target_image_path)
        # cond_im, _ = self.load_filtered_img(inference_image_path)
        geometry_flag = random.randint(0, 3)
        assert geometry_flag>=0 and geometry_flag<4
        cond_im = self.load_filtered_img(inference_image_path)
        target_map = self.load_geometry_maps(target_image_path, geometry_flag)
        target_map = target_map.permute(1, 2, 0)

        data["image_target"] = target_map
        assert data["image_target"].shape[-1]==3
        data["image_cond"] = cond_im
        if geometry_flag==0:
            geometry_signal = self.brdf_clip_signals["Albedo"]
        elif geometry_flag==1:
            geometry_signal = self.brdf_clip_signals["Normal"]
        elif geometry_flag==2:
            geometry_signal = self.brdf_clip_signals["Roughness"]
        else:
            geometry_signal = self.brdf_clip_signals["Specular"]
        pose_offset = torch.tensor([elevation_offset, math.sin(azimuth_offset), math.cos(azimuth_offset), distance_offset])
        data["T"] = torch.cat([pose_offset, geometry_signal])
        
        if self.postprocess is not None:
            data = self.postprocess(data)

        return data
    
class GeometryFeatureData(Dataset):
    def __init__(self,
        root_dir="/home/ubuntu/data/abo/",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=90,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        # self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
            
        self.meta = []
        split = pd.read_csv(f'{root_dir}/render/train_test_split.csv')
        if validation:
            split = split[split['SPLIT'] == 'TEST']['MODEL']
        else:
            split = split[split['SPLIT'] == 'TRAIN']['MODEL']
        for model_name in split:
            # for view_id in range(len(glob.glob(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512', model_name, "*_brdf_rgb.png")))):
            for view_id in range(90):
                if os.path.exists(os.path.join(self.root_dir, 'zero123_renderings_pbr_90_res512_noise_latent', model_name, f"{view_id:03d}_rgb_latent.npy")):
                    meta_dict = {
                        "model_name": model_name,
                        "view_id": view_id,
                        "target_denoised_feature_path": glob.glob(f"{self.root_dir}/zero123_renderings_pbr_90_res512_noise_latent/{model_name}/{view_id:03d}_cond_0[0-9][0-9]_rgb_denoise_latent.npy")[0]
                    }
                    self.meta.append(meta_dict)
        self.transform = image_transforms
        self.geometry_transform = torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        if validation:
            print("validation data number", len(self.meta))
        else:
            print("train data number", len(self.meta))

    def __len__(self):
        return len(self.meta)

    def load_filtered_img(self, image_path, color=[1., 1., 1.]):
        # img = plt.imread(image_path + "_brdf_rgb.png")
        # img_black_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img_black_background = img_black_background.convert("RGB")
        # img_black_background = self.transform(img_black_background)

        # mask = plt.imread(image_path + "_rgb.png")
        # img[mask[:, :, -1] == 0.] = color
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        # img = img.convert("RGB")
        # img = self.transform(img)
        # return img.permute(2,0,1), img_black_background.permute(2,0,1)

        img = plt.imread(image_path + "_brdf_rgb_white.png")
        img_white_background = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        img_white_background = img_white_background.convert("RGB")
        img_white_background = self.transform(img_white_background)
        return img_white_background.permute(2,0,1)

    def load_geometry_maps(self, target_image_path):
        h, w = 512, 512
        target_mask_path = target_image_path + "_rgb.png"
        target_albedo_path = target_image_path + "_albedo0001.png"
        target_normal_path = target_image_path + "_normal.png"
        target_roughness_path = target_image_path + "_roughness.png"

        mask = cv2.imread(target_mask_path, cv2.IMREAD_UNCHANGED)/255
        mask = mask[:,:,3:]
        mask[mask>0.1] = 1
        mask = np.concatenate([mask, mask, mask], axis=2)

        diffuse = cv2.imread(target_albedo_path, cv2.IMREAD_UNCHANGED)
        diffuse = diffuse[:,:,:3]
        diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB)
        diffuse = diffuse*mask + (1-mask)*255
        diffuse = diffuse.transpose(2,0,1)/255
        diffuse = torch.from_numpy(diffuse)
        diffuse = self.geometry_transform(diffuse)
        diffuse = diffuse.clamp(0,1)

        specular = torch.ones(h,w,1)*0.5
        specular = specular*torch.from_numpy(mask[:,:,0:1])
        specular = specular.permute(2,0,1)
        specular = self.geometry_transform(specular)
        specular = specular.expand(3, -1, -1).float()

        roughness = cv2.imread(target_roughness_path, cv2.IMREAD_UNCHANGED)
        roughness = cv2.cvtColor(roughness, cv2.COLOR_BGRA2GRAY)[:,:,np.newaxis]
        roughness = roughness[...,:1]*mask[:,:,0:1]
        roughness = roughness.transpose(2,0,1)/255
        roughness = torch.from_numpy(roughness)
        roughness = self.geometry_transform(roughness)
        roughness = roughness.expand(3, -1, -1).float()

        normal = cv2.imread(target_normal_path, cv2.IMREAD_UNCHANGED)
        normal = normal[...,:3]
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal_map_x = normal[:, :, 0]/127.5-1
        normal_map_y = (normal[:, :, 1]/127.5-1)
        normal_map_z = (normal[:, :, 2]/127.5-1)
        normal = np.stack([normal_map_x, normal_map_y, normal_map_z], axis=-1)
        background_normal = np.zeros((h,w,3))
        background_normal[:,:,2] = 1
        normal = normal*mask + (1-mask)*background_normal
        normal = normal / np.sqrt(np.sum(normal**2, axis=-1, keepdims=True))
        normal = normal.transpose(2,0,1)
        normal = torch.from_numpy(normal)[:3,:,:]
        normal = self.geometry_transform(normal)

        return diffuse*2-1, normal, specular*2-1, roughness*2-1
    
    def load_geometry_features(self, path):
        return torch.from_numpy(np.load(path)).float()
    
    def __getitem__(self, index):
        model_name = self.meta[index]["model_name"]
        inference_id = self.meta[index]["view_id"]
        target_id = inference_id
        # with open(f'{self.root_dir}/zero123_renderings_pbr_90_res512/{model_name}/angles.json', "r") as f:
        #     angle_json = json.load(f)
        data = {}

        target_image_path = f"{self.root_dir}/zero123_renderings_pbr_90_res512_noise_latent/{model_name}/{target_id:03d}"

        # inference_id = str(inference_id)
        target_id = str(target_id)
        # azimuth_offset = (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi)
        # elevation_offset = angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]
        # distance_offset = angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
        
        if self.return_paths:
            data["path"] = str(target_image_path)

        # target_im = self.load_filtered_img(target_image_path)
        # cond_im = target_im
        # target_diffuse, target_normal, target_specular, target_roughness = self.load_geometry_maps(target_image_path)

        # data["image_target"] = torch.cat([target_diffuse, target_normal, target_specular, target_roughness, target_im], dim=0)
        # assert data["image_target"].shape[0]==11
        # data["image_cond"] = cond_im

        rgb_feature = self.load_geometry_features(self.meta[index]["target_denoised_feature_path"])
        data["rgb"] = rgb_feature
        albedo_feature = self.load_geometry_features(target_image_path+"_albedo_latent.npy")
        data["albedo"] = albedo_feature
        normal_feature = self.load_geometry_features(target_image_path+"_normal_latent.npy")
        data["normal"] = normal_feature
        roughness_feature = self.load_geometry_features(target_image_path+"_roughness_latent.npy")
        data["roughness"] = roughness_feature
        specular_feature = self.load_geometry_features(target_image_path+"_specular_latent.npy")
        data["specular"] = specular_feature

        # if self.postprocess is not None:
        #     data = self.postprocess(data)

        return data

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
