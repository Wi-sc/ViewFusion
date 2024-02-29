# ViewFusion: Towards Multi-View Consistency via Interpolated Denoising
### [Project Page](https://wi-sc.github.io/ViewFusion.github.io/)  | [Paper]() | [Arxiv]() | [Video]()

[ViewFusion: Towards Multi-View Consistency via Interpolated Denoising](https://wi-sc.github.io/ViewFusion.github.io/)  
 [Xianghui Yang](https://wi-sc.github.io/xianghui-yang)<sup>1,2</sup>, [Yan Zuo](https://www.amazon.science/author/yan-zuo)<sup>1</sup>, [Sameera Ramasinghe](https://www.amazon.science/author/sameera-ramasinghe)<sup>1</sup>, [Loris Bazzani](https://lorisbaz.github.io/)<sup>1</sup>, [Gil Avraham](https://www.amazon.science/author/gil-avraham/)<sup>1</sup>, [Anton van den Hengel](https://researchers.adelaide.edu.au/profile/anton.vandenhengel)<sup>1,3</sup> <br>
 <sup>1</sup>Amazon, <sup>2</sup>The University of Sydney, <sup>3</sup>The University of Adelaide

### [Novel View Synthesis](https://github.com/Wi-sc/ViewFusion#novel-view-synthesis-1):
<video  width="320" height="240" auto-play="true" loop="loop" muted="muted" plays-inline="true">
  <source src="video/real_images.mp4" type="video/mp4">
</video>

### [3D Reconstruction](https://github.com/Wi-sc/ViewFusion#3d-reconstruction-neus):
<video auto-play="true" loop="loop" muted="muted" plays-inline="true">
  <source src="./video/gso_shape.mp4" type="video/mp4">
</video>


## Updates
- [ ] The clean code has been uploaded!
- [x] The project page is online now 🤗: https://wi-sc.github.io/ViewFusion.github.io/.  
- [x] We've limited the autoregressive window size, so don't worry about the memory requirement. It needs around 23GB VRAM so it's totally runnable on a RTX 3090/4090(Ti)!  

##  Usage
###  Novel View Synthesis
We use the totally same environment with [Zero-1-to-3](https://github.com/cvlab-columbia/zero123).
```
conda create -n zero123 python=3.9
conda activate zero123
cd zero123
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `zero123` through one of the following sources:

```
https://huggingface.co/cvlab/zero123-weights/tree/main
wget https://cv.cs.columbia.edu/zero123/assets/$iteration.ckpt    # iteration = [105000, 165000, 230000, 300000]
```
[Zero-1-to-3](https://github.com/cvlab-columbia/zero123) has released 5 model weights: `105000.ckpt`, `165000.ckpt`, `230000.ckpt`, `300000.ckpt`, and `zero123-xl.ckpt`. By default, we use `zero123-xl.ckpt`, but we also find that 105000.ckpt which is the checkpoint after finetuning 105000 iterations on objaverse has better generalization ablitty. So if you are trying to generate novel-view images and find one model fails, you can try another one.

### Training

Training? We don't need any training or finetuning. :wink:

### Dataset

Download our objaverse renderings with:
```
wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
```
Disclaimer: note that the renderings are generated with Objaverse. The renderings as a whole are released under the ODC-By 1.0 license. The licenses for the renderings of individual objects are released under the same license creative commons that they are in Objaverse.

### 3D Reconstruction (NeuS)
Note that we haven't extensively tuned the hyperparameters for 3D recosntruction. Feel free to explore and play around!
```
cd 3drec
pip install -r requirements.txt
python run_zero123.py \
    --scene pikachu \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 0 \
    --depth_smooth_weight 10000. \
    --near_view_weight 10000. \
    --train_view True \
    --prefix "experiments/exp_wild" \
    --vox.blend_bg_texture False \
    --nerf_path "data/nerf_wild"
```
- You can see results under: `3drec/experiments/exp_wild/$EXP_NAME`.  


- To export a mesh from the trained Voxel NeRF with marching cube, use the [`export_mesh`](https://github.com/cvlab-columbia/zero123/blob/3736c13fc832c3fc8bf015de833e9da68a397ed9/3drec/voxnerf/vox.py#L71) function. For example, add a line:

    ``` vox.export_mesh($PATH_TO_EXPORT)```

    under the [`evaluate`](https://github.com/cvlab-columbia/zero123/blob/3736c13fc832c3fc8bf015de833e9da68a397ed9/3drec/run_zero123.py#L304) function.  


- The dataset is formatted in the same way as NeRF for the convenience of dataloading. In reality, the recommended input in addition to the input image is an estimate of the elevation angle of the image (e.g. if the image is taken from top, the angle is 0, front is 90, bottom is 180). This is hard-coded now to the extrinsics matrix in `transforms_train.json`

- We tested the installation processes on a system with Ubuntu 20.04 with an NVIDIA GPU with Ampere architecture.


##  Acknowledgement
This repository is based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Zero-1-to-3](https://zero123.cs.columbia.edu/), [Objaverse](https://objaverse.allenai.org/), [NeuS](https://github.com/Totoro97/NeuS) and [SyncDreamer](https://github.com/pals-ttic/sjc/). We would like to thank the authors of these work for publicly releasing their code.


##  Citation
```
@misc{,
      title={ViewFusion: Towards Multi-View Consistency via Interpolated Denoising}, 
      author={},
      year={},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
