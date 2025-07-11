import torch
from torchvision import models, transforms
from PIL import Image

resnet = models.resnet101(weights='IMAGENET1K_V1').eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("cat.jpg")  # Replace with your image path
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f]

with torch.no_grad():
    out = resnet(batch_t)
    probabilities = torch.nn.functional.softmax(out[0], dim=0) * 100
    top5_prob, top5_idx = torch.topk(probabilities, 5)

for i in range(5):
    print(f"{labels[top5_idx[i]]}: {top5_prob[i].item():.2f}%")
