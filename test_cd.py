import torch
from torch import autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from PIL import Image
import numpy as np
from chamfer_distance import ChamferDistance
import open3d as o3d
import sys
import os
sys.path.append("/home/ubuntu/data/abo")
from common.io_assets.eames.load_glb import eames_load_asset

SAVE_DIR = "/home/ubuntu/workspace/zero123/3drec/experiments/finetune_1000steps_sofa_zero123_renderings"

class ABODataset(Dataset):
    def __init__(self, threshold=16):
        self.threshold = threshold
        metadata_csv = pd.read_csv('/home/ubuntu/data/abo/3dmodels/metadata/3dmodels.csv')
        self.model_index_to_path = dict()
        for i in range(len(metadata_csv)):
            self.model_index_to_path[metadata_csv.iloc[i]['3dmodel_id']] = metadata_csv.iloc[i]['path']

        self.meta = []
        split = pd.read_csv('/home/ubuntu/data/abo/render/train_test_split.csv')
        split = split[split['SPLIT'] == 'TEST']['MODEL']

        for model_name in split:
            # if not os.path.exists(f"{SAVE_DIR}/scene-{model_name}-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/model.obj"):
            #     continue
            with open(f'/home/ubuntu/data/abo/angle/{model_name}.json', "r") as f:
                angle_json = json.load(f)
            with open(f'/home/ubuntu/data/abo/inference_index/{model_name}.json', "r") as f:
                inference_json = json.load(f)
            for env_index in inference_json:
                # for inference_id, target_id in zip(inference_json[env_index]["view_inference"], inference_json[env_index]["view_target"]):
                #     inference_id = str(inference_id)
                #     target_id = str(target_id)
                #     meta_dict = {
                #         "model_name": model_name,
                #         "inference_index": inference_id,
                #         "target_index": target_id,
                #         "env_index": env_index,
                #         "inference_image_path": f"/home/ubuntu/data/abo/render/{model_name}/render/{env_index}/render_{inference_id}.jpg",
                #         "inference_mask_path": f"/home/ubuntu/data/abo/render/{model_name}/segmentation/segmentation_{inference_id}.jpg",
                #         "target_image_path": f"/home/ubuntu/data/abo/render/{model_name}/render/{env_index}/render_{target_id}.jpg",
                #         "target_mask_path": f"/home/ubuntu/data/abo/render/{model_name}/segmentation/segmentation_{target_id}.jpg",
                #         "gen_image_path": f"{SAVE_DIR}/{model_name}/{env_index}/{target_id}.jpg",
                #         "inference_angle": angle_json[inference_id]["angle"],
                #         "target_angle": angle_json[target_id]["angle"],
                #         "azimuth_offset": (angle_json[target_id]["angle"][0]-angle_json[inference_id]["angle"][0])%(2*np.pi),
                #         "elevation_offset": (angle_json[target_id]["angle"][1]-angle_json[inference_id]["angle"][1]),
                #         "distance_offset": angle_json[target_id]["angle"][2]-angle_json[inference_id]["angle"][2]
                #     }
                #     self.meta.append(meta_dict)

                inference_id = str(22)
                if os.path.exists(f'/home/ubuntu/workspace/zero123/3drec/data/sofa/{model_name}/{env_index}_{inference_id}.jpg'):
                    meta_dict = {
                        "model_name": model_name,
                        "inference_index": inference_id,
                        "env_index": env_index,
                        "inference_image_path": f"/home/ubuntu/data/abo/render/{model_name}/render/{env_index}/render_{inference_id}.jpg",
                        "inference_mask_path": f"/home/ubuntu/data/abo/render/{model_name}/segmentation/segmentation_{inference_id}.jpg",
                        "inference_angle": angle_json[inference_id]["angle"],
                    }
                    self.meta.append(meta_dict)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta_info = self.meta[idx]
        model_name, env_index = meta_info['model_name'], meta_info['env_index']

        model_path = self.model_index_to_path[model_name]
        target_path = f"/home/ubuntu/data/abo/3dmodels/original/{model_path}"
        json_path = f"/home/ubuntu/data/abo/render/{model_name}/metadata.json"
        target_points = self.load_normalized_glb(target_path, json_path=json_path)

        gen_path = f"{SAVE_DIR}/scene-{model_name}-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/model_m{self.threshold}.obj"
        
        generation_points = self.load_normalized_obj(gen_path)

        target_points = np.asarray(target_points.points)
        generation_points = np.asarray(generation_points.points)
        
        return torch.tensor(target_points), torch.tensor(generation_points)#, model_name, env_index, target_index
    
    def load_normalized_glb(self, glb_path, json_path):
        # with open(json_path, "r") as f:
        #     render_json = json.load(f)
        open3d_obj = eames_load_asset(glb_path)
        # exhange_axis = np.array([[-1,0,0],[0,0,-1],[0,1,0]])
        verts = np.asarray(open3d_obj.vertices)
        # verts = verts@exhange_axis
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        vcen = (vmin+vmax)/2
        obj_size = np.abs(verts - vcen).max()
        verts = verts - vcen.reshape(1,3)
        verts = verts/obj_size
        open3d_obj.vertices = o3d.utility.Vector3dVector(verts)
        points = open3d_obj.sample_points_uniformly(10000)
        return points

    def load_normalized_obj(self, obj_path):
        open3d_obj = o3d.io.read_triangle_mesh(obj_path)
        verts = np.asarray(open3d_obj.vertices)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        vcen = (vmin+vmax)/2
        obj_size = np.abs(verts - vcen).max()
        verts = verts - vcen.reshape(1,3)
        verts = verts/obj_size
        open3d_obj.vertices = o3d.utility.Vector3dVector(verts)
        points = open3d_obj.sample_points_uniformly(10000)
        return points




if __name__ == '__main__':
    chamfer_dist = ChamferDistance()
    cd_list = {'20':[],
                '18':[],
                '16':[],
                '14':[],
                '12':[],
                '10':[],
                '8':[],
                '6':[],
                '4':[],
                '2':[]
    }
    for threshold in cd_list:
        print("Calculate threshold", threshold)
        dataset = ABODataset(threshold=int(threshold))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        for data in dataloader:
            real_points, gen_points = data#, model_name, env_index, target_index = data
            real_points = real_points.cuda()
            gen_points = gen_points.cuda()
            dist1, dist2, idx1, idx2 = chamfer_dist(real_points, gen_points)
            loss = (torch.mean(dist1, dim=1)) + (torch.mean(dist2, dim=1))
            cd_list[threshold].extend(loss.cpu().tolist())

        data_num = len(cd_list[threshold])
        cd = np.mean(cd_list[threshold])
        print(f"Size: {data_num}")
        print(f"CD: {cd}")
    
        with open(f"{SAVE_DIR}/test_all_threshold.txt", "a") as f:
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Size: {data_num}\n")
            f.write(f"CD: {cd}\n")
            f.write(f"************\n")
