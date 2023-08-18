import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MultiAttribDataset(Dataset):
    def __init__(self, csv, size=(224,224)):
        self.csv = csv
        self.image_names = self.csv[:]['Filenames']
        self.labels = np.array(self.csv.drop(['Filenames'], axis=1))

        # Set the training data images and labels
        print(f"Number of images: {self.image_names}")

        # Define the training transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
        ])
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(f"/content/multi-label-classification-pytorch/dataset/images/{self.image_names[index]}")
        
        # Convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
