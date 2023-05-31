FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install git libgl1 -y && apt-get clean
RUN pip install open3d tdqm lightning && pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
CMD bash