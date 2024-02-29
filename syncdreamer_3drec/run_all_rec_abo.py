import os
import json
import pandas as pd
import numpy as np

with open('/home/ubuntu/data/abo/rendering_test_list_video_condition_id.json', "r") as f:
    model_dict = json.load(f)

# syncdreamer
for model_name in model_dict:
    inference_id = model_dict[model_name]
    cmd = f"CUDA_VISIBLE_DEVICES=7 python train_renderer.py \
        -i /home/ubuntu/workspace/zero123/zero123/experiments_cvpr/syncdreamer/abo_cvpr_video/{model_name}/{inference_id} \
        -n {model_name} \
        -l abo/abo_syncdreamer"
    print(cmd)

    if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/abo/abo_syncdreamer/{model_name}/mesh.ply"):
        os.system(cmd)
    else:
        print("skip")

# img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_360v16_autoregressive/gen_abo_blender_elevation60_inference_t0.50_auto_t1.00"
# # img_dir = "/home/ubuntu/workspace/zero123/zero123/experiments_cvpr/pretrain_zero123_xl_360/gen_abo_blender_spin36_fix_0index"
# for model_name in model_dict:
#     inference_id = model_dict[model_name]
#     with open(f'/home/ubuntu/data/abo/zero123_renderings_elevation60/{model_name}/angles.json', "r") as f:
#         angle_json = json.load(f)
#     elevation = np.pi/2 - angle_json[str(inference_id)]["angle"][1]
#     distance = angle_json[str(inference_id)]["angle"][2]
#     cmd = f"CUDA_VISIBLE_DEVICES=7 python my_train_renderer_spin16.py \
#         -i {img_dir}/{model_name}/{inference_id} \
#         -n {model_name} \
#         -e {elevation} \
#         -d {distance} \
#         -l abo/abo_autoregressive_rec_spin16"
#     print(cmd)

#     if not os.path.exists(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/abo/abo_autoregressive_rec_spin16/{model_name}/mesh.ply"):
#         os.remove(f"/home/ubuntu/workspace/zero123/syncdreamer_3drec/abo/abo_autoregressive_rec_spin16/{model_name}/ckpt/last.ckpt")
#         os.system(cmd)
#     else:
#         print("skip")