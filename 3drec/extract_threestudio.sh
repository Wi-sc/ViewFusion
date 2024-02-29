
GPU_ID=$1         # e.g. 0
IMAGE_PREFIX=$2   # e.g. "anya_front"
ZERO123_PREFIX=$3 # e.g. "zero123-xl"
ELEVATION=$4      # e.g. 0
REST=${@:5:99}  

python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-noise_atten" \
    system.loggers.wandb.name="real_images_zero123_{$IMAGE_PREFIX}_zero123_${ZERO123_PREFIX}...fov20_${REST}" \
    data.image_path=/home/ubuntu/workspace/zero123/3drec/data/real_images_zero123/${IMAGE_PREFIX}.png system.freq.guidance_eval=37 \
    system.guidance.pretrained_model_name_or_path="/home/ubuntu/workspace/zero123/zero123/${ZERO123_PREFIX}.ckpt" \
    system.guidance.cond_elevation_deg=$ELEVATION \
    ${REST}