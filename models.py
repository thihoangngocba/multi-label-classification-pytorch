from torchvision import models as models
import torch.nn as nn

def resnet50(pretrained, requires_grad, num_classes=None):
    model = models.resnet50(progress=True, pretrained=pretrained)
    
    # Freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # Unfreeze the hidden layers to make them trainable
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    
    # Prediction head
    model.fc = nn.Linear(2048, num_classes)

    return model
