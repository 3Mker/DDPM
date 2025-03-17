# import torch
# import torchvision
# from PIL import Image
# import cv2
# import os
# import numpy as np


# class OCTADataset(torch.utils.data.Dataset):

#     def __init__(self):
#         self.root_path = '/mnt/f/dataset/OCTA_500_BOTH_MODAL'
#         self.files_path = 'OCTA_Projection'
#         self.files = os.listdir(os.path.join(self.root_path, self.files_path))
#         self.transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((224, 224)),
#             torchvision.transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         img = cv2.imread(os.path.join(self.root_path, self.files_path, self.files[index]), cv2.IMREAD_GRAYSCALE)
#         img = Image.fromarray(img)
#         return self.transform(img)


import torch
import torchvision
from PIL import Image
import os

class OCTADataset(torch.utils.data.Dataset):

    def __init__(self, root_path='/mnt/f/dataset/OCTA_500_BOTH_MODAL', files_path='OCTA_Projection'):
        self.root_path = root_path
        self.files_path = files_path
        self.files = [f for f in os.listdir(os.path.join(self.root_path, self.files_path)) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        # 缓存图像
        self.images = [self._load_image(f) for f in self.files]

    def _load_image(self, file_name):
        try:
            img_path = os.path.join(self.root_path, self.files_path, file_name)
            img = Image.open(img_path).convert('L')  # 转换为灰度图像
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")
            return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.images[index]