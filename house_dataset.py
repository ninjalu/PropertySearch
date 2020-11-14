# %%
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
# %%


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None):
        super().__init__()
        self.csv = pd.read_csv(csv)  # read the data csv
        self.transform = transform  # save the transform variable as part of the class object

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # get the image filepath and label from at that index from the csv
        filepath, label = self.csv['file_path'][idx], self.csv['label'][idx]
        # open with PIL and convert to rgb
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)  # apply transforms
        return img, label


# %%
# data_without_resize = ClassificationDataset(
# )
# data_without_resize[0][0].show()

# data_with_resize = ClassificationDataset(
#     transform=transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.ToPILImage()
#     ])
# )
# data_with_resize[0][0].show()
# %%


if __name__ == '__main__':
    data = ClassificationDataset(
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    )


# %%
# %%
