import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import MultiAttribDataset
from torch.utils.data import DataLoader

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Intialize the model
model = models.resnet50(pretrained=False, requires_grad=False, num_classes=3).to(device)

# Load the model checkpoint
checkpoint = torch.load('./outputs/model.pth')

# Load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load annotation CSV file
test_csv = pd.read_csv('/content/multi-label-classification-pytorch/dataset/test.csv')

# Prepare the test dataset and dataloader
test_data = MultiAttribDataset(
    test_csv, size=(224,224)
)

test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)


for counter, data in enumerate(test_loader):
    image, label = data['image'].to(device), data['label']
    
    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    pred = np.round(outputs[0])

    print(f"Prediction: {pred} | Label: {label}")

    # image = image.squeeze(0)
    # image = image.detach().cpu().numpy()
    # image = np.transpose(image, (1, 2, 0))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    # plt.savefig(f"../outputs/inference_{counter}.jpg")
    # plt.show()
