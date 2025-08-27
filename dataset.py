import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose

class AnimalDataset(Dataset):
    def __init__(self, root, train = True,transforms=None):
        self.transforms = transforms
        self.images = []
        self.labels = []
        
        self.folders = os.listdir(root)
        self.class_to_idx = {cls :idx for idx, cls in enumerate(self.folders)}
        
        for folder in self.folders:
            folder_path = os.path.join(root, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png','.jpg','.jpeg')):
                    file_path = os.path.join(folder_path, file)
                    try:
                        image = Image.open(file_path).convert('RGB')
                        #image = image.resize((64, 64))  # Resize to 64x64
                        self.images.append(image)
                        self.labels.append(self.class_to_idx[folder]) # Lưu label theo index
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label

if __name__ =='__main__':
    transforms = Compose([
        Resize((64,64)),
        ToTensor()
    ])
    root =r'./dataset/train'
    dataset = AnimalDataset(root= root, transforms=transforms)
    for iter, (image , label) in enumerate(dataset):
        print(f"Image shape: {image.shape}, Label: {label}")
        break


    
        