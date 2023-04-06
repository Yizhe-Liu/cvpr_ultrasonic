from tqdm import tqdm
import numpy as np
import open3d as o3d

X, Y, Z = np.meshgrid(np.array(range(768))*0.49479, np.array(range(768))*0.49479, np.array(range(1280))*0.3125)
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)

for i in tqdm(['001','002','003','004','005']):
    mesh = o3d.io.read_triangle_mesh(f'data/meshes/scan_{i}.ply')
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    distance = scene.compute_distance(pts).numpy().reshape(768, 768, 1280)
    np.save(f'data/distance_field/gt_{i}.npy', distance)
    np.save(f'data/occ_field/gt_{i}.npy', distance < 1)