import os 
import sys
import cv2
import numpy as np


# /mnt/f/dataset/OCTA_500_BOTH_MODAL/OCTA_Projection
root_path = '/mnt/f/dataset/OCTA_500_BOTH_MODAL/OCTA_Projection'
files = os.listdir(root_path)                                                                                                                           
sample = cv2.imread(os.path.join(root_path, files[0]), cv2.IMREAD_GRAYSCALE)
print(sample.shape)
# 400 400 只有一个channel