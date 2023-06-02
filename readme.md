# CVPR 2023 Ultrasonic Data Challenge
Authors:
Yizhe Liu Email: y2549liu@uwaterloo.ca
Nick Torenvliet Email: yihze.liu@uwaterloo.ca

## Environment
Please use the Dockerfile in this repo to build a docker image and mount the project folder. 
```bash
sudo docker build . --tag yizhe/ultrasonic_env
sudo docker run -it --gpus all -v.:/project --shm-size=64gb yizhe/ultrasonic_env
cd /project
```

## Dataset
- Unzip the dataset under data/ without the enclosing training folder. There folder strusture should look like: 
```
data
├── distance_field
├── meshes
├── occ_field
├── README.md
└── volumes
```

- Run mesh_2_occ.py to convert the meshes for the ground truth scan into occupancy fields: gt_001.npy, ..., gt_005.npy, which is the sames shape as the input scan. This process is very slow. 


## Training
```bash
python train.py --slicing=xy
python train.py --slicing=yz
python train.py --slicing=zx
# You can adjust the batch size
```
After training, you can find your checkpoints under trained/ and then run gen_logits_train.sh. 
Alternatively, you can also download pretrained weights and unzip it under pretrained/, and then run gen_logits.sh.

Copy/soft link ground truth scans: gt_001.npy, ..., gt_005.npy from data/occ_field to output/ and train the learned voter using the following commands. 

```bash
cp data/occ_field/gt*.npy output/
cd learned_voter
python train.py
```
After training, you can find your checkpoints under trained/ and then run gen_predictions_train.sh. 
Alternatively, you can use the pretrained weights and run gen_predictions.sh.
The final predicts can be found under output/



## Evaluation 
Run occ_2_pcd.py to convert occupancy field (.pt) to point cloud (.xyz).  
For example
```bash
python occ_2_pcd.py -i output/pred_001_voted.pt
```
The output will be in the same folder of the input.  


## Evaluation (Appendix)
Suppose the testing file is 'testing/volume/scan_001.raw'
You can run the following commands to generates 3 predictions in 3 slicing direction (pred_001_xy_2d_soft_5layers.pt, pred_001_yz_2d_soft_5layers.pt, pred_001_zx_2d_soft_5layers.pt) under the output folder. 

```bash
python predict.py --ckpt pretrained/xy.ckpt --path testing/ --slicing=xy --idx 1 --label=soft
python predict.py --ckpt pretrained/yz.ckpt --path testing/ --slicing=yz --idx 1 --label=soft
python predict.py --ckpt pretrained/zx.ckpt --path testing/ --slicing=zx --idx 1 --label=soft
```

Then, you can use the learned voter to aggregate them. The output file pred_001_voted.pt will be stored under the output foler.
```bash
cd learned_voter
python predict.py --ckpt ../pretrained/voter.ckpt --idx 1
```

Finally, you can convert the occupancy file to pointcloud using the following command. 
You will find the resulting pointcloud at output/pred_001_voted.xyz

```bash
cd .. # go back to project root
python occ_2_pcd.py -i output/pred_001_voted.pt
```