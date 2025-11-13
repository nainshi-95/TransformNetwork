import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

class YDomainImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', crop_size=256):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.image_paths = glob.glob(os.path.join(root_dir, "*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return torch.zeros(1, self.crop_size, self.crop_size)

        # 1. Y 도메인 추출
        y_img = img.convert('YCbCr').split()[0]

        # 2. 패딩 로직 (이미지가 crop_size보다 작을 경우)
        w, h = y_img.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)

        if pad_w > 0 or pad_h > 0:
            # (left, top, right, bottom) 순서로 패딩 계산
            # 이미지를 중앙에 두기 위해 양쪽으로 나눔
            padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
            # fill=0 (검은색 패딩), padding_mode='constant'
            y_img = F.pad(y_img, padding, fill=0, padding_mode='constant')

        # 3. Mode에 따른 Crop
        if self.mode == 'train':
            cropper = transforms.RandomCrop(self.crop_size)
        else:
            cropper = transforms.CenterCrop(self.crop_size)
        
        y_img_cropped = cropper(y_img)

        return self.to_tensor(y_img_cropped)
