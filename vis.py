import numpy as np
from matplotlib import pyplot as plt
import cv2

raw_img = np.fromfile('data/volumes/scan_005.raw', '<u2').reshape(1280, 768, 768)
out = cv2.VideoWriter('scan_005.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (768, 768), 0)
for i in raw_img:
    img = (i//256).astype(np.uint8)[..., None]
    #print(img.shape)
    out.write(img)


out.release()