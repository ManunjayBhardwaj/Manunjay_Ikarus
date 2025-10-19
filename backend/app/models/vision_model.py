import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

def get_vision_model(num_classes):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def extract_image_embedding(image_path):
    if not os.path.exists(image_path):
        return None
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    x = preprocess(img).unsqueeze(0)
    model = models.resnet18(pretrained=True)
    # remove final layer
    feat = nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        out = feat(x).squeeze().numpy()
    return out

def infer_category_from_title(title):
    t = title.lower()
    if 'chair' in t:
        return 'Chairs'
    if 'sofa' in t or 'couch' in t:
        return 'Sofas'
    if 'table' in t:
        return 'Tables'
    return 'Other'
