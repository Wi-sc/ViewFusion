import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import io
import sys
sys.path.append("/home/ubuntu/data/abo")
from common.io_assets.eames.load_glb import eames_load_asset
import cv2
import imageio
import glob

# data_dir = '/home/ubuntu/workspace/zero123/3drec/data/sofa'
# gen_dir_0 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut8"
# gen_dir_1 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut4"
# gen_dir_2 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut2"
# gt_dir = "/home/ubuntu/data/abo/3dmodels/original"

data_dir = '/home/ubuntu/workspace/zero123/3drec/data/sofa_zero123_rendering'
# gen_dir = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut2"
gen_dir = "/home/ubuntu/workspace/zero123/3drec/experiments/finetune_1000steps_sofa_zero123_renderings"
gt_dir = "/home/ubuntu/data/abo/3dmodels/original"

# data_dir = '/home/ubuntu/workspace/zero123/3drec/data/real_images_zero123'
# gen_dir = "/home/ubuntu/workspace/zero123/3drec/experiments/real_images_zero123"

def get_renderings(obj_path, exchange_axis_flag=False, camera_loc=None):
    if obj_path.endswith("glb"):
        open3d_obj = eames_load_asset(obj_path)
    else:
        open3d_obj = o3d.io.read_triangle_mesh(obj_path)
        
    
    verts = np.asarray(open3d_obj.vertices)
    if exchange_axis_flag:
        exchange_axis = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        verts = verts@exchange_axis

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vcen = (vmin+vmax)/2
    obj_size = np.abs(verts - vcen).max()
    verts = verts - vcen.reshape(1,3)
    verts = verts/obj_size
    open3d_obj.vertices = o3d.utility.Vector3dVector(verts)
    
    image = render_meshes(open3d_obj, camera_loc=camera_loc)
    return image

def render_meshes(meshes, camera_loc=None):
    meshes.compute_vertex_normals()
    meshes.compute_triangle_normals()
    meshes.paint_uniform_color(np.array([0.9, 0.9, 0.9]).reshape(3,1))

    render = o3d.visualization.rendering.OffscreenRenderer(width=224, height=224)
    render.scene.set_background(np.array([1, 1, 1, 1]))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.aspect_ratio = 1.0
    mat.shader = 'defaultLit'
    mat.base_color = [0.9,0.9,0.9,1]
    mat.base_roughness = 0.8
    mat.absorption_color = [0, 0, 0]
    mat.base_reflectance = 0
    mat.point_size = 4

    render.scene.add_geometry("scene1", meshes, mat)
    if camera_loc==None:
        render.setup_camera(60, [0, 0, 0], [0, 2, 2], [0, 1, 0]) # lookat eye up
    else:
        render.setup_camera(60, [0, 0, 0], camera_loc, [0, 1, 0]) # lookat eye up
        
    render.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    # render.scene.scene.add_directional_light("directionallight", [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 1000, True) 
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.enable_indirect_light(True)
    render.scene.view.set_post_processing(False)

    image = render.render_to_image()
    o3d.io.write_image("tmp.png", image, 9)
    image = np.asarray(image).astype('uint8')
    # print(image.max(), image.min())
    
    # print(type(image))
    # image = Image.fromarray(image)

    # # Convert the PIL Image to bytes
    # image_bytes = io.BytesIO()
    # image.save(image_bytes, format='PNG')
    # image_bytes.seek(0)

    return image

def add_image_to_canvas(frame_id, img, canvas, row_id, col_id):
    # img = imageio.imread(os.path.join(image_path))
    img = cv2.resize(img, (224, 224))
    # print(img.shape)
    if img.shape[-1]==4:
        mask = (np.array(img[:,:,3:]) / 255.).astype(np.float32)
        img = img*mask + (1-mask)*255
    canvas[frame_id, row_id*256+16:row_id*256+240, col_id*256+16:col_id*256+240] = np.asarray(img)[:,:,:3]
    return canvas


def export_movie(seqs, fname, fps=5):
    writer = imageio.get_writer(fname, fps=fps)
    for img in seqs:
        writer.append_data(img)
    writer.close()


# model_name_list = sorted(os.listdir(data_dir))[:2]
model_name_list = ['B07DB8XGY2', 'B07P5LNMDK']
folder_id_list = [str(i) for i in range(19)] + [chr(ord("A")+i) for i in range(26)]

n_rows = len(model_name_list)
n_coloumns = 4
n_frames = 100
box = 256
canvas = np.ones((100, n_rows*box, n_coloumns*box, 3))*255

for i, model_name in enumerate(model_name_list):
    # for zero123rendering
    file_name = glob.glob(os.path.join(data_dir, model_name, "*.png"))[0]
    
    # for real
    # file_name = model_name
    # model_name = model_name[:-4]
    
    # for original abo
    # env_id, view_id = file_name.split(".")[0].split("_")

    for folder_id in folder_id_list:
        gt_mesh_path = os.path.join(gt_dir, folder_id, model_name+".glb")
        if os.path.exists(gt_mesh_path):
            break

    for frame_id in range(n_frames):
        x = 2*np.sin(-frame_id/n_frames*2*np.pi)
        z = 2*np.cos(frame_id/n_frames*2*np.pi)
        y = 2
        gen_mesh = get_renderings(gt_mesh_path, exchange_axis_flag=False, camera_loc=[x, y, z])
        canvas[frame_id, i*256+16:i*256+240, 16:240] = gen_mesh

    gen_model_dir = f"scene-{model_name}-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/"

    # gen_mesh_0 =  os.path.join(gen_dir, gen_model_path, "gen_render_m16.png")
    # for frame_id in range(n_frames):
    #     canvas = add_image_to_canvas(frame_id, gen_mesh_0, canvas, i, 1)

    gen_mesh_path =  os.path.join(gen_dir, gen_model_dir, "model_m8.obj")
    for frame_id in range(n_frames):
        x = 2*np.sin(-frame_id/n_frames*2*np.pi)
        z = 2*np.cos(frame_id/n_frames*2*np.pi)
        y = 2
        gen_mesh = get_renderings(gen_mesh_path, exchange_axis_flag=False, camera_loc=[x, y, z])
        canvas = add_image_to_canvas(frame_id, gen_mesh, canvas, i, 1)

    # gen_mesh_2  = os.path.join(gen_dir, gen_model_path, "gen_render_m4.png")
    # for frame_id in range(n_frames):
    #     canvas = add_image_to_canvas(frame_id, gen_mesh_2, canvas, i, 3)

    # gen_mesh_3  = os.path.join(gen_dir, gen_model_path, "gen_render_m2.png")
    for frame_id in range(n_frames):
        # canvas = add_image_to_canvas(frame_id, gen_mesh_3, canvas, i, 4)
        gen_nerf = imageio.imread(os.path.join(gen_dir, gen_model_dir, "test_10000/img", "step_%d.png"%frame_id))
        canvas = add_image_to_canvas(frame_id, gen_nerf, canvas, i, 2)

    image_inference = imageio.imread(os.path.join(data_dir, model_name, file_name))
    # image_inference = imageio.imread(os.path.join(data_dir, file_name))
    for frame_id in range(n_frames):
        canvas = add_image_to_canvas(frame_id, image_inference, canvas, i, 3)


video_path = os.path.join("finetuned_zero123_on_abo_sofa_2.mp4")
export_movie(canvas, video_path)