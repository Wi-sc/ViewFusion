import os
import math
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import glob
import trimesh
import mesh2sdf
from chamfer_distance import ChamferDistance as chamfer_dist
import pickle
import open3d as o3d

class GSODataset(Dataset):
    def __init__(self, generated_dir, save_dir='/home/ubuntu/data/GSO/shapes'):
        self.meta = []
        self.mesh_scale = 0.8
        self.size = 128
        self.level = 2 / self.size
        # with open('/home/ubuntu/data/abo/rendering_test_list_video_condition_id.json', "r") as f:
        #     model_dict = json.load(f)
        # for model_name in model_dict:
        #     inference_id = model_dict[model_name]
        #     with open(f'{save_dir}/{model_name}/angles.json', "r") as f:
        #         angle_json = json.load(f)
        #     for target_id in range(12):
        #         if target_id==inference_id or \
        #             os.path.exists(f"{generated_dir}/{model_name}/{inference_id}_{target_id}_generation.png"):
        #             continue
        #         print(model_name, target_id)
        #         meta_dict = {
        #             "model_name": model_name,
        #             "inference_index": inference_id,
        #             "target_index": target_id,
        #             "inference_image_path": f"{save_dir}/{model_name}/{int(inference_id):03d}",
        #             "target_image_path": f"{save_dir}/{model_name}/{int(target_id):03d}",
        #             "inference_angle": angle_json[str(inference_id)]["angle"],
        #             "target_angle": angle_json[str(target_id)]["angle"],
        #             "azimuth_offset": (angle_json[str(target_id)]["angle"][0]-angle_json[str(inference_id)]["angle"][0])%(2*np.pi),
        #             "elevation_offset": (angle_json[str(target_id)]["angle"][1]-angle_json[str(inference_id)]["angle"][1]),
        #             "distance_offset": angle_json[str(target_id)]["angle"][2]-angle_json[str(inference_id)]["angle"][2]
        #         }
        #         self.meta.append(meta_dict)

        with open('/home/ubuntu/data/GSO/rendering_test_list_video_condition_id.json', "r") as f:
            model_dict = json.load(f)
        for model_name in model_dict:
            inference_id = model_dict[model_name]

            # with open(f"/home/ubuntu/data/GSO/gso_syncdreamer/gso_syncdreamer_random/{model_name}/meta.pkl", 'rb') as f:
            #     K, azimuths, elevations, distances, cam_poses = pickle.load(f)
            # azimuth = azimuths[inference_id]

            with open(f'/home/ubuntu/data/GSO/gso_renderings_elevation60/{model_name}/angles.json', "r") as f:
                angle_json = json.load(f)
            azimuth = angle_json[str(inference_id)]['angle'][0] # + np.pi/18

            meta_dict = {
                "model_name": model_name,
                "inference_index": inference_id,
                "gen_shape_path": f"{generated_dir}/{model_name}/mesh.ply",
                "gt_shape_path": f"{save_dir}/{model_name}/meshes/model.obj",
                "gen_sdf_path": f"{generated_dir}/{model_name}/mesh2sdf.npy",
                "gt_sdf_path": f"{save_dir}/{model_name}/mesh2sdf.npy",
                "azimuth": azimuth
            }
            self.meta.append(meta_dict)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.geometry_transform = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
    def __len__(self):
        return len(self.meta)

    def load_ply(self, shap_path):
        mesh_scale = 0.8
        mesh = trimesh.load(shap_path, force='mesh')

        # normalize mesh
        vertices = mesh.vertices
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale

        mesh.vertices = vertices
        return mesh
    
    def load_obj(self, shap_path):
        mesh_scale = 0.8
        mesh = trimesh.load(shap_path, force='mesh')
        # normalize mesh
        vertices = mesh.vertices
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = vertices
        return mesh

    def __getitem__(self, idx):
        meta_info = self.meta[idx]
        gen_shape_path = meta_info['gen_shape_path']
        gt_shape_path = meta_info['gt_shape_path']
        model_name = meta_info['model_name']
        gen_sdf_path = meta_info['gen_sdf_path']
        gt_sdf_path = meta_info['gt_sdf_path']
        azimuth = 2*np.pi-meta_info['azimuth']
        
        gen_mesh = self.load_ply(gen_shape_path)
        gt_mesh = self.load_obj(gt_shape_path)

        # if os.path.exists(gen_sdf_path):
        #     gen_sdf = np.load(gen_sdf_path)
        # else:
        #     gen_sdf, _ = mesh2sdf.compute(gen_mesh.vertices, gen_mesh.faces, self.size, fix=True, level=self.level, return_mesh=True)
        #     np.save(gen_sdf_path, gen_sdf)
        # if os.path.exists(gt_sdf_path):
        #     gt_sdf = np.load(gt_sdf_path)
        # else:
        #     gt_sdf, _ = mesh2sdf.compute(gt_mesh.vertices, gen_mesh.faces, self.size, fix=True, level=self.level, return_mesh=True)
        #     np.save(gt_sdf_path, gt_sdf)
        
        gen_sdf = np.random.randn(3,3,3) - 0.5
        gt_sdf = np.random.randn(3,3,3) - 0.5
        gen_sdf = np.where(gen_sdf < 0, 1, 0)
        gt_sdf = np.where(gt_sdf < 0, 1, 0)

        # gen_points, _ = trimesh.sample.sample_surface_even(gen_mesh, 10000)
        # gt_points, _ = trimesh.sample.sample_surface_even(gt_mesh, 10000)
        gen_points = gen_mesh.sample(10000)
        gt_points = gt_mesh.sample(10000)

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(gen_points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(gt_points)
        threshold = 0.02
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p.transformation)
        gen_points = np.concatenate([gen_points, np.ones((10000, 1))], axis=1)@reg_p2p.transformation.T
        gen_points = gen_points[:, :3]
        print(gen_points.shape, gt_points.shape)
        trimesh.Trimesh(vertices=gen_points, faces=[]).export(gen_shape_path[:-4]+"_canonical_pred.obj")
        trimesh.Trimesh(vertices=gt_points, faces=[]).export(gen_shape_path[:-4]+"_canonical_gt.obj")

        gen_points = np.array(gen_points)
        gt_points = np.array(gt_points)
        # print(type(gen_points), type(gt_sdf))

        return model_name, gen_points, gt_points, gen_sdf, gt_sdf

def get_fscore(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def main_run(dataset, output_dir, device):
    # Calculate intersection and union
    cd_list = []
    iou_list = []
    fscore_list = []
    model_list = []
    chd = chamfer_dist()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for data in dataloader:
        model_name, gen_points, gt_points, gen_sdf, gt_sdf = data
        gen_sdf, gt_sdf = gen_sdf[0].numpy(), gt_sdf[0].numpy()
        intersection = np.logical_and(gen_sdf, gt_sdf).sum()
        union = np.logical_or(gen_sdf, gt_sdf).sum()
        # Calculate IoU
        iou = intersection / union
        dist1, dist2, idx1, idx2 = chd(gen_points.to(device), gt_points.to(device))
        cd = (torch.mean(dist1)) + (torch.mean(dist2))
        cd = cd.detach().cpu().item()
        fscore, _, _ = get_fscore(dist1, dist2)
        fscore = fscore.squeeze().item()
        print(cd, iou, fscore)
        iou_list.append(iou)
        cd_list.append(cd)
        fscore_list.append(fscore)
        model_list.append((model_name, cd, fscore, iou))

    mean_cd = np.mean(cd_list)
    mean_fscore = np.mean(fscore_list)
    mean_iou = np.mean(iou_list)
    print(f"CD: {mean_cd}")
    print(f"Fscore: {mean_fscore}")
    print(f"IoU: {mean_iou}")

    model_list = sorted(model_list, key=lambda x: -x[2])

    # save_text_path = "/".join(SAVE_DIR.split("/")[:-1])
    # with open(f"{save_text_path}/test_gso.txt", "w") as f:
    with open(f"{output_dir}.txt", "w") as f:
        f.write(f"CD: {mean_cd}\n")
        f.write(f"Fscore: {mean_fscore}\n")
        f.write(f"IoU: {mean_iou}\n")
        for i in range(len(model_list)):
            f.write(f"model-{model_list[i][0]}-cd-{model_list[i][1]}-fscore-{model_list[i][2]}-iou-{model_list[i][3]}\n")


if __name__ == '__main__':
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    mp.set_start_method('spawn')
    torch.set_num_threads(1)

    # import argparse
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # args = parser.parse_args()

    device_idx=0

    processes = []
    output_dir = '/home/ubuntu/workspace/zero123/syncdreamer_3drec/multi_view/pointe_gso'
    dataset = GSODataset(output_dir)
    dataset_len = len(dataset)

    main_run(dataset, output_dir, device_idx)
    # rank_num = 8
    # subset_split = [dataset_len//rank_num for i in range(rank_num)]
    # if sum(subset_split)!=dataset_len:
    #     subset_split[-1] += dataset_len-sum(subset_split)
    # print(subset_split)
    # dataset_list = torch.utils.data.random_split(dataset, subset_split)
    # for rank in range(rank_num):
    #     device = f'cuda:{rank}'
    #     p = mp.Process(target=main_run, args=(dataset_list[rank], output_dir, device))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()