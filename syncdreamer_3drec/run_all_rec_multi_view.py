import os
import json
import pandas as pd
import numpy as np

with open('/home/ubuntu/data/abo/rendering_test_3view_condition_id.json', "r") as f:
    model_dict = json.load(f)

# syncdreamer
# for model_name in model_dict:
#     inference_id = model_dict[model_name]
#     cmd = f"CUDA_VISIBLE_DEVICES=2 python train_renderer.py \
#         -i /home/ubuntu/workspace/zero123/zero123/experiments_cvpr/syncdreamer/gso_cvpr_video_2ndrun/{model_name}/{inference_id} \
#         -n {model_name} \
#         -l 2ndrun/gso_syncdreamer_rec_2nd"
#     print(cmd)

#     if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/2ndrun/gso_syncdreamer_rec_2nd/{model_name}/mesh.ply"):
#         os.system(cmd)
#     else:
#         print("skip")

img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_360v36_multiview_autoregressive/gen_abo_blender_elevation60_inference_t0.50_auto_t1.00"
# img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_360v36_multiview/gen_gso"
for model_name in model_dict:
    inference_id = model_dict[model_name][0]
    with open(f'/home/ubuntu/data/abo/zero123_renderings_elevation60/{model_name}/angles.json', "r") as f:
        angle_json = json.load(f)
    elevation = np.pi/2 - angle_json[str(inference_id)]["angle"][1]
    distance = angle_json[str(inference_id)]["angle"][2]
    cmd = f"CUDA_VISIBLE_DEVICES=6 python my_train_renderer_multi_view.py \
        -i {img_dir}/{model_name}/{inference_id} \
        -n {model_name} \
        -e {elevation} \
        -d {distance} \
        -l multi_view/abo_autoregressive_rec_spin36"
    print(cmd)

    if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/multi_view/abo_autoregressive_rec_spin36/{model_name}/mesh.ply"):
        os.system(cmd)
    else:
        print("skip")