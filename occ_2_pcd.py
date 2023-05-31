import torch 
import numpy as np
import open3d as o3d
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser('Convert Occupancy Field to Point Cloud')
    parser.add_argument('--input', '-i', type=str, help='Occupancy filed path') 
    args = parser.parse_args()

    occ = torch.load(args.input)
    idx = (occ.permute((1, 2, 0)).nonzero() + torch.Tensor([[0.5, 0.5, 0.5]]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(idx*torch.Tensor([[0.49479, 0.49479, 0.3125]]))
    o3d.io.write_point_cloud(args.input[:-3] + '.xyz', pcd)

            