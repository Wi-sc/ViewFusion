import os
import json
import pandas as pd
import numpy as np
import pickle

with open('/home/ubuntu/data/GSO/rendering_test_list_video_condition_id.json', "r") as f:
    model_dict = json.load(f)
save_dir = "/home/ubuntu/data/GSO/gso_syncdreamer"
# syncdreamer
for model_name in model_dict:
    # inference_id = model_dict[model_name]
    inference_id = 13
    cmd = f"CUDA_VISIBLE_DEVICES=5 python train_renderer.py \
        -i /home/ubuntu/workspace/zero123/zero123/experiments_cvpr/syncdreamer/gso_syncdreamer/{model_name}/{inference_id} \
        -n {model_name} \
        -l syncdreamer_renderings/gso_syncdreamer_rec"
    print(cmd)

    if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/syncdreamer_renderings/gso_syncdreamer_rec/{model_name}/mesh.ply"):
        os.system(cmd)
    else:
        print("skip")

# img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/syncdreamer_renderings_for_3d/pretrain_zero123_xl_360v36_autoregressive/gen_gso_blender_elevation60_inference_t0.50_auto_t1.00"
# # img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/syncdreamer_renderings_for_3d/pretrain_zero123_xl_360/gen_gso_spin36"
# for model_name in model_dict:
#     inference_id = 13
#     with open(f"{save_dir}/gso_syncdreamer_random/{model_name}/meta.pkl", 'rb') as f:
#         K, azimuths, elevations, distances, cam_poses = pickle.load(f)
#     azimuths, elevations = azimuths.astype(np.float32), elevations.astype(np.float32)
#     elevation = elevations[inference_id]
#     distance = distances[inference_id]
#     cmd = f"CUDA_VISIBLE_DEVICES=6 python my_train_renderer.py \
#         -i {img_dir}/{model_name}/{inference_id} \
#         -n {model_name} \
#         -e {elevation} \
#         -d {distance} \
#         -l syncdreamer_renderings/gso_autoregressive_rec_spin36"
#     print(cmd)

#     if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/syncdreamer_renderings/gso_autoregressive_rec_spin36/{model_name}/mesh.ply"):
#         os.system(cmd)
#     else:
#         print("skip")