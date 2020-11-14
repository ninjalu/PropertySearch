import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import yaml
import urlli
import numpy as np


# model = models.resnext101_32x8d(pretrained=True)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])

image = Image.open(
    '/Users/luluo/projects/PropertySearch/propertyimages/image-43725919-0.jpg')

file = urllib. request. urlopen(
    "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
classes = ''
for f in file:
    classes = classes + f.decode("utf-8")
classes = yaml.load(classes, Loader=yaml.Loader)

img = transform(image)
img = img.unsqueeze(0)
model.eval()
pred = model(img)
print(pred)
labels = pred[0]['labels'].detach().numpy().tolist()
labels = [classes[i] for i in labels]
scores = pred[0]['scores'].detach().numpy().tolist()
pred = [(labels[i], scores[i]) for i in range(len(labels))]
print(pred)


# pred = torch.nn.functional.softmax(pred, dim=1)[0]
# labels = [(pred[i].item(), classes[i]) for i in range(len(classes))]
# labels.sort(reverse=True)
# print(labels[:10])

# print(classes[torch.argmax(pred).numpy().tolist()])
