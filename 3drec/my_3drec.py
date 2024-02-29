import os
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = '/home/ubuntu/workspace/zero123/3drec/data/sofa_zero123_rendering'
# OUTPUT_DIR = '/home/ubuntu/workspace/zero123/3drec/data/real_images_zero123'
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# split = pd.read_csv('/home/ubuntu/data/abo/render/train_test_split.csv')
# split = split[split['SPLIT'] == 'TEST']['MODEL']
# meta = []

# with open(f'/home/ubuntu/data/abo/model_category.json', "r") as f:
#     model_catgeroy_dict = json.load(f)

# for model_name in split:
#     with open(f'/home/ubuntu/data/abo/angle/{model_name}.json', "r") as f:
#         angle_json = json.load(f)
#     with open(f'/home/ubuntu/data/abo/inference_index/{model_name}.json', "r") as f:
#         inference_json = json.load(f)
#     model_cat = model_catgeroy_dict[model_name]
#     if model_cat!="SOFA":
#         continue
#     for env_index in inference_json:
#         inference_id = str(22)
#         meta_dict = {
#             "model_name": model_name,
#             "inference_index": inference_id,
#             "env_index": env_index,
#             "inference_image_path": f"/home/ubuntu/data/abo/render/{model_name}/render/{env_index}/render_{inference_id}.jpg",
#             "inference_mask_path": f"/home/ubuntu/data/abo/render/{model_name}/segmentation/segmentation_{inference_id}.jpg",
#             "inference_angle": angle_json[inference_id]["angle"],
#             }
#         meta.append(meta_dict)

# show_idx = np.random.choice(len(meta), 10, replace=False)
# for i in show_idx:
#     meta_info = meta[i]
#     inference_image_path = meta_info["inference_image_path"]
#     model_name = meta_info["model_name"]
#     env_index = meta_info["env_index"]
#     inference_index = meta_info["inference_index"]
#     rec_path = f"{OUTPUT_DIR}/{model_name}/{env_index}_{inference_index}.jpg"
#     if not os.path.exists(f"{OUTPUT_DIR}/{model_name}/"):
#         os.makedirs(f"{OUTPUT_DIR}/{model_name}/")
#     print(f"cp {inference_image_path} {rec_path}")
#     os.system(f"cp {inference_image_path} {rec_path}")


for model_name in os.listdir(OUTPUT_DIR):
    # if model_name.endswith('.png'):
    #     model_name = model_name[:-4]
    # cmd = f"python run_zero123_uncertainty.py \
    #     --scene {model_name} \
    #     --index 0 \
    #     --n_steps 10000 \
    #     --lr 0.05 \
    #     --sd.scale 100.0 \
    #     --emptiness_weight 0 \
    #     --depth_smooth_weight 10000. \
    #     --near_view_weight 10000. \
    #     --train_view True \
    #     --prefix experiments/pretrained_diffusion_on_blender_uncertainty \
    #     --vox.blend_bg_texture False \
    #     --nerf_path data/sofa_zero123_rendering\
    #     --pose.FoV 49.1 \
    #     --uncertainty_threshold 0.05 \
    #     --pose.R 2.0"
    cmd = f"python run_zero123_with_consistent_renderings_autoregressive.py \
        --scene {model_name} \
        --index 0 \
        --n_steps 10000 \
        --lr 0.05 \
        --sd.scale 100.0 \
        --emptiness_weight 1000. \
        --depth_smooth_weight 500. \
        --view_weight 10000 \
        --near_view_weight 500. \
        --train_view True \
        --prefix experiments/pretrain_autoregreesive_120degree_inference_5e-1_consistent_emptiness_1000  \
        --vox.blend_bg_texture False \
        --nerf_path data/sofa_zero123_rendering \
        --pose.FoV 49.1 \
        --uncertainty_threshold 0.03 \
        --pose.R 2.0"
    print(cmd)
    if not os.path.exists(f"/home/ubuntu/workspace/zero123/3drec/experiments/pretrain_autoregreesive_120degree_inference_5e-1_consistent_emptiness_1000/scene-{model_name}-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-500.0_near-view-wt-500.0"):
        os.system(cmd)
    else:
        print("skip")
    # os.system(cmd)