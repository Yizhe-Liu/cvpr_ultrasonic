# CVPR 2023 Ultrasonic Data Challenge

## Dataset
- Unzip the dataset under data/
- Run mesh_2_occ.py to convert the meshes for the ground truth scan into occupancy fields, which is the sames shape as the input scan. 

## Training
```bash
python train.py --slicing=xy
python train.py --slicing=yz
python train.py --slicing=zx
```
Modify gen_logits.sh to use the correct checkpoints.  
Copy/soft link ground truth scans: gt_001.npy, ..., gt_005.npy from data/occ_field to output/.  
```bash
cd learned_voter
python train.py
python predict.py 
```

## Evaluation 

