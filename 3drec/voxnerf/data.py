from pathlib import Path
import json
import numpy as np
import imageio
import os
import cv2
from .utils import blend_rgba


def my_load_blender(model_name, path="data/abo"):
    with open(f'/home/ubuntu/data/abo/render/{model_name}/metadata.json', "r") as f:
        abo_render_json = json.load(f)
    root = Path(path) / model_name
    file_name = str(list(root.glob('*.jpg'))[0])
    env_index, view_index = file_name.split("/")[-1].split(".")[0].split("_")
    poses = np.array(abo_render_json['views'][int(view_index)]['pose']).reshape(4,4).astype(np.float32)
    # poses[:3,:3] = poses[:3,:3].T
    # poses = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0], [0,0,0,1]])@poses
    assert abo_render_json['views'][int(view_index)]['index'] == int(view_index)
    K = np.array(abo_render_json['views'][int(view_index)]['K']).reshape(3,3).astype(np.float32)
    fov = 60
    
    inference_image_path = f"/home/ubuntu/data/abo/render/{model_name}/render/{env_index}/render_{view_index}.jpg"
    inference_mask_path = f"/home/ubuntu/data/abo/render/{model_name}/segmentation/segmentation_{view_index}.jpg"
    im = imageio.imread(inference_image_path)
    im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_CUBIC)
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
    # im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_CUBIC)
    im = (np.array(im) / 255.).astype(np.float32)

    mask = imageio.imread(inference_mask_path)
    mask = cv2.resize(mask, (512, 512), interpolation = cv2.INTER_CUBIC)
    # mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
    # mask = cv2.resize(mask, (512, 512), interpolation = cv2.INTER_CUBIC)
    mask = (np.array(mask) / 255.).astype(np.float32)
    im = im*mask[:,:,None] + (1-mask[:,:,None])

    im = np.expand_dims(im, axis=0)
    poses = np.expand_dims(poses, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return im, K, poses, mask, fov

def my_load_abo_zero123_blender(model_name, path="data/abo"):
    root = Path(path) / model_name
    file_name = str(list(root.glob('*.png'))[0])
    view_index = 22

    poses = np.eye(4)
    poses_numpy_load = np.load(str(list(root.glob('*.npy'))[0])).reshape(3,4).astype(np.float32)
    c2w = poses_numpy_load[:3,:3].T
    camera_loc = c2w@(-poses_numpy_load[:3,3].reshape(3,1))
    poses[:3, :3] = c2w
    poses[:3, 3] = camera_loc.reshape(3,)
    # poses = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0], [0,0,0,1]])@poses
    
    fov = 49.1
    K=np.array([[256/np.tan(fov/2/180*np.pi), 0, 0],
                     [0, 256/np.tan(fov/2/180*np.pi), 0],
                     [0, 0, 1]])
    
    im = imageio.imread(file_name)
    im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_CUBIC)
    im = (np.array(im) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = im[:, :, -1]
    im = blend_rgba(im)

    im = np.expand_dims(im, axis=0)
    poses = np.expand_dims(poses, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return im, K, poses, mask, fov

def my_load_real(model_name, path="data/abo"):
    root = Path(path) / model_name
    file_name = str(root) + '.png'

    poses = np.eye(4)
    poses_numpy_load = np.load('/home/ubuntu/workspace/zero123/3drec/data/sofa_zero123_rendering/B07B4MMV3N/000.npy').reshape(3,4).astype(np.float32)
    c2w = poses_numpy_load[:3,:3].T
    camera_loc = c2w@(-poses_numpy_load[:3,3].reshape(3,1))
    poses[:3, :3] = c2w
    poses[:3, 3] = camera_loc.reshape(3,)
    # poses = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0], [0,0,0,1]])@poses
    
    fov = 49.1
    K=np.array([[256/np.tan(fov/2/180*np.pi), 0, 0],
                     [0, 256/np.tan(fov/2/180*np.pi), 0],
                     [0, 0, 1]])
    
    im = imageio.imread(file_name)
    im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_CUBIC)
    im = (np.array(im) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = im[:, :, -1]
    im = blend_rgba(im)

    im = np.expand_dims(im, axis=0)
    poses = np.expand_dims(poses, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return im, K, poses, mask, fov

def load_blender(split, scene="lego", half_res=False, path="data/nerf_synthetic"):
    assert split in ("train", "val", "test")

    root = Path(path) / scene

    with open(root / f'transforms_{split}.json', "r") as f:
        meta = json.load(f)

    imgs, poses = [], []

    for frame in meta['frames']:
        file_name = root / f"{frame['file_path']}.png"
        im = imageio.imread(file_name)
        im = cv2.resize(im, (800, 800), interpolation = cv2.INTER_CUBIC)

        c2w = frame['transform_matrix']

        imgs.append(im)
        poses.append(c2w)

    imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = imgs[:, :, :, -1]
    imgs = blend_rgba(imgs)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    f = 1 / np.tan(camera_angle_x / 2) * (W / 2)

    if half_res:
        raise NotImplementedError()

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])  # note OpenGL -ve z convention;

    fov = meta['camera_angle_x']

    return imgs, K, poses, mask, fov

def load_wild(dataset_root, scene, index):

    root = Path(dataset_root) / scene

    with open(root / f'transforms_train.json', "r") as f:
        meta = json.load(f)
        
    frame = meta['frames'][index]
    file_name = root / f"{frame['file_path']}.png"
    img = imageio.imread(file_name)

    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    pose = meta['frames'][index]['transform_matrix']
    pose = np.array(pose)

    return img, pose, mask, None


def load_googlescan_data(dataset_root, scene, index, split='render_mvs'):
    render_folder = os.path.join(dataset_root, scene, split, "model")
    if not os.path.exists(render_folder):
        print(f"Render folder {render_folder} does not exist")

    image_path = os.path.join(render_folder, f"{index:03d}.png")
    cam_path = os.path.join(render_folder, f"{index:03d}.npy")
    mesh_path = os.path.join(dataset_root, scene, split, "model_norm.obj")
    if not os.path.exists(mesh_path):
        mesh_path = None

    img = imageio.imread(image_path)
    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    pose = np.load(cam_path) # [3, 4]
    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
    return img, pose, mask, mesh_path


def load_rtmv_data(dataset_root, scene, index):
    scene_dir = os.path.join(dataset_root, scene)
    if not os.path.exists(scene_dir):
        print(f"Render folder {scene_dir} does not exist")

    image_path = os.path.join(scene_dir, f"{index:05d}.png")
    mesh_path = os.path.join(dataset_root, scene, "scene.ply")
    if not os.path.exists(mesh_path):
        mesh_path = None

    img = imageio.imread(image_path)
    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    with open(os.path.join(scene_dir, "transforms.json"), "r") as f:
        meta = json.load(f)

    pose = meta['frames'][index]['transform_matrix']
    pose = np.array(pose)
    return img, pose, mask, mesh_path
