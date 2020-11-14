# %%
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
from time import time
import numpy as np
from modules import AEDataset, SimpleAE
import pandas as pd
# %%
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405],
                         std=[0.229, 0.224, 0.225])
])

data_dir = 'propertyimages/garden/image_list.csv'
batch_size = 4
data = AEDataset(data_dir, transform=data_transform)
data_loader = torch.utils.data.DataLoader(
    data, batch_size=batch_size, shuffle=True)

# %%


def train(model, data_loader, device, epochs=3, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_train_idx = 0
    for epoch in range(epochs):
        for img, label in data_loader:
            # img = img.to(device)
            # label = label.to(device)
            pred = model(img)
            # print(pred.shape)
            loss_train = F.mse_loss(pred, label)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Loss train:', loss_train.item(), batch_train_idx)
            batch_train_idx += 1
            break


# %%
# model = ResNet_VAE()
model = SimpleAE()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train(model, data_loader, device)

# %%
model = SimpleAE()
model.load_state_dict(torch.load(
    '/Users/luluo/projects/propertySearch/models/ae.pt', map_location=torch.device('cpu')))
# %%
model = model.eval()

# %%


def get_latent_rep(model, csv, transform):
    df = pd.read_csv(csv)
    latent = []
    for i in range(len(df)):
        file_path = df['file_path'][i]
        img = Image.open(file_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        # print(img.shape)
        features = model.encode(img).squeeze().flatten()
        latent.append((file_path, features.detach().numpy()))
    return latent


latent = get_latent_rep(model, data_dir, data_transform)
# %%
len(latent[0][1])
# %%
img_paths = [latent[i][0] for i in range(len(latent))]
latent_rep = [latent[i][1] for i in range(len(latent))]
img_paths[:5]

# # %%
# df_lat = pd.DataFrame(features)
# df_lat.head()
# # %%
# # %%
# img = Image.open(latent[0][0]).convert('RGB')
# img.show()
# %%
tene = TSNE().fit_transform(latent_rep[:10])
x = tene[:, 0]
y = tene[:, 1]
# %%


def getImage(path, size):
    image = Image.open(path)
    image = image.resize(size)
    return OffsetImage(image)


# %%
fig, ax = plt.subplots()
ax.scatter(x, y)
for x0, y0, path in zip(x, y, img_paths):
    ab = AnnotationBbox(getImage(path, (60, 60)), (x0, y0), frameon=False)
    ax.add_artist(ab)

# %%
