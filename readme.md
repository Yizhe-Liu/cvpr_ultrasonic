# CVPR 2023 Ultrasonic Data Challenge


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

```bash
# cd .. 
python mesh_2_occ.py -i output/pred_001_voted.pt
```

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