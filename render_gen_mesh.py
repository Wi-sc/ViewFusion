import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import io
import sys
sys.path.append("/home/ubuntu/data/abo")
from common.io_assets.eames.load_glb import eames_load_asset

# data_dir = '/home/ubuntu/workspace/zero123/3drec/data/sofa'
# gen_dir_0 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut8"
# gen_dir_1 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut4"
# gen_dir_2 = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut2"
# gt_dir = "/home/ubuntu/data/abo/3dmodels/original"

data_dir = '/home/ubuntu/workspace/zero123/3drec/data/sofa_zero123_rendering'
gen_dir = "/home/ubuntu/workspace/zero123/3drec/experiments/sofa_zero123_rendering_thresholdmut2"
gt_dir = "/home/ubuntu/data/abo/3dmodels/original"

# data_dir = '/home/ubuntu/workspace/zero123/3drec/data/real_images_zero123'
# gen_dir = "/home/ubuntu/workspace/zero123/3drec/experiments/real_images_zero123"

def get_renderings(obj_path):
    if obj_path.endswith("glb"):
        open3d_obj = eames_load_asset(obj_path)
    else:
        open3d_obj = o3d.io.read_triangle_mesh(obj_path)
        
    # exhange_axis = np.array([[-1,0,0],[0,0,-1],[0,1,0]])
    verts = np.asarray(open3d_obj.vertices)
    # verts = verts@exhange_axis
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vcen = (vmin+vmax)/2
    obj_size = np.abs(verts - vcen).max()
    verts = verts - vcen.reshape(1,3)
    verts = verts/obj_size
    open3d_obj.vertices = o3d.utility.Vector3dVector(verts)
    
    image = render_meshes(open3d_obj)
    return image

def render_meshes(meshes, width=224, height=224):
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

    render.setup_camera(60, [0, 0, 0], [0, 2, 2], [0, 1, 0]) # lookat eye up
    render.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    # render.scene.scene.add_directional_light("directionallight", [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 1000, True) 
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.enable_indirect_light(True)
    render.scene.view.set_post_processing(False)

    image = render.render_to_image()
    o3d.io.write_image("tmp.png", image, 9)
    image = np.asarray(image).astype('uint8')
    print(image.max(), image.min())
    
    # print(type(image))
    image = Image.fromarray(image)

    # # Convert the PIL Image to bytes
    # image_bytes = io.BytesIO()
    # image.save(image_bytes, format='PNG')
    # image_bytes.seek(0)

    return image


model_name_list = os.listdir(data_dir)
folder_id_list = [str(i) for i in range(19)] + [chr(ord("A")+i) for i in range(26)]

fig, axes = plt.subplots(nrows=len(model_name_list), ncols=6, figsize=[10, 20])

for i, model_name in enumerate(model_name_list):
    file_name = os.listdir(os.path.join(data_dir, model_name))[0]

    # for real
    # file_name = model_name
    # model_name = model_name[:-4]
    
    # for original abo
    # env_id, view_id = file_name.split(".")[0].split("_")

    for folder_id in folder_id_list:
        gt_mesh_path = os.path.join(gt_dir, folder_id, model_name+".glb")
        if os.path.exists(gt_mesh_path):
            break
    gen_gt_mesh = get_renderings(gt_mesh_path)
    axes[i, 0].imshow(gen_gt_mesh)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"GT")

    gen_model_path = f"scene-{model_name}-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/"

    gen_mesh_0 = get_renderings(os.path.join(gen_dir, gen_model_path, "model_m16.obj"))
    gen_mesh_0 = gen_mesh_0.resize((224,224))
    gen_mesh_0.save(os.path.join(gen_dir, gen_model_path, "gen_render_m16.png"))
    axes[i, 1].imshow(gen_mesh_0)
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f"x16")

    gen_mesh_1 = get_renderings(os.path.join(gen_dir, gen_model_path, "model_m8.obj"))
    gen_mesh_1 = gen_mesh_1.resize((224,224))
    gen_mesh_1.save(os.path.join(gen_dir, gen_model_path, "gen_render_m8.png"))
    axes[i, 2].imshow(gen_mesh_1)
    axes[i, 2].axis('off')
    axes[i, 2].set_title(f"x8")

    gen_mesh_2 = get_renderings(os.path.join(gen_dir, gen_model_path, "model_m4.obj"))
    gen_mesh_2 = gen_mesh_2.resize((224,224))
    gen_mesh_2.save(os.path.join(gen_dir, gen_model_path, "gen_render_m4.png"))
    axes[i, 3].imshow(gen_mesh_2)
    axes[i, 3].axis('off')
    axes[i, 3].set_title(f"x4")

    gen_mesh_3 = get_renderings(os.path.join(gen_dir, gen_model_path, "model_m2.obj"))
    gen_mesh_3 = gen_mesh_3.resize((224,224))
    gen_mesh_3.save(os.path.join(gen_dir, gen_model_path, "gen_render_m2.png"))
    axes[i, 4].imshow(gen_mesh_3)
    axes[i, 4].axis('off')
    axes[i, 4].set_title(f"x2")

    image_inference = Image.open(os.path.join(data_dir, model_name, file_name))
    # image_inference = Image.open(os.path.join(data_dir, file_name))
    image_inference = image_inference.resize((224,224))
    axes[i, 5].imshow(image_inference)
    axes[i, 5].axis('off')
    axes[i, 5].set_title(f"image")

fig.tight_layout()
fig.savefig('vis.png')