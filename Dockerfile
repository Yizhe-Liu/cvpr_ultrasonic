FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt update && apt install -y git libgomp libGL1 && apt-get clean
RUN pip install opencv-python open3d tdqm lightning && pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
CMD bash