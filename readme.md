# CVPR 2023 Ultrasonic Data Challenge

## Dataset
- Unzip the dataset under data/
- Run mesh_2_occ.py to convert the meshes for the ground truth scan into occupancy fields: gt_001.npy, ..., gt_005.npy, which is the sames shape as the input scan. 

## Training
```bash
python train.py --slicing=xy
python train.py --slicing=yz
python train.py --slicing=zx
```

You can also download pretrained weights and unzip it under pretrained/
Modify gen_logits.sh to use the correct checkpoints.  
Copy/soft link ground truth scans: gt_001.npy, ..., gt_005.npy from data/occ_field to output/.  
```bash
cd learned_voter
python train.py
```

Modify gen_predictions.sh to generate final occupancy predictions.

## Environment
Please use the Dockerfile in this repo to build a docker image and mount the project folder. 

## Evaluation 
Run occ_2_pcd.py to convert occupancy field (.pt) to point cloud (.xyz).  
The output will be in the same folder of the input.  