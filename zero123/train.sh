python main.py \
    -t \
    --name sd_masking_clip_signal_abo_diffusion_pbr_view90 \
    --base configs/sd-abo-masking_geometry-c_concat-256.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 1 \
    --finetune_from /home/ubuntu/workspace/zero123/zero123/105000.ckpt
    # --resume /home/ubuntu/workspace/xhuiyang_zero123/zero123/logs/2023-06-10T11-42-05_sd-abo-finetune-c_concat-256/checkpoints/last.ckpt