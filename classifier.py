# %%
from torchvision import transforms
import os
import pandas as pd
from fastai.vision.widgets import *
from fastbook import *
import fastbook
import numpy as np
fastbook.setup_book()
# %%
learn = load_learner('models/rnclassifer.pkl')
# %%


# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, csv, transform=None):
#         super().__init__()
#         self.csv = pd.read_csv(csv)  # read the data csv
#         self.transform = transform  # save the transform variable as part of the class object

#     def __len__(self):
#         return len(self.csv)

#     def __getitem__(self, idx):
#         # get the image filepath and label from at that index from the csv
#         filepath = self.csv['file_path'][idx]
#         # open with PIL and convert to rgb
#         img = Image.open(filepath).convert("RGB")
#         if self.transform:
#             img = self.transform(img)  # apply transforms
#         return img


# images = ImageDataset(csv='propertyimages/image_list.csv',
#                       transform=transforms.Compose([
#                           transforms.Resize((256, 256)),
#                           transforms.ToTensor()
#                       ])
#                       )


# images[6]

# %%
images = pd.read_csv('propertyimages/image_list.csv')
images.head()
# images = np.array(images).squeeze()
# images[0]
# # %%
# labels = []
# for idx, image in enumerate(images):
#     label, _, _ = learn.predict(image)
#     labels.append(label)
# %%
def get_x(r): return r['file_path']


image_block = DataBlock(
    get_x=get_x,
    item_tfms=RandomResizedCrop(256, min_scale=0.7)
)
image_data = image_block.dataloaders(images)
# %%
preds, _ = learn.get_preds(image_data)
print(preds)
# %%
pred = zip(images, labels)
print(pred)
