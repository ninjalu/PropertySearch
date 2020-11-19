# %%


from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
from time import time
import numpy as np
from modules import AEDataset, SimpleAE, vis_tsne, vis_tb
import pandas as pd
import pickle
# %%
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405],
                         std=[0.229, 0.224, 0.225])
])

data_dir = 'propertyimages/interior/image_list.csv'
batch_size = 4
data = AEDataset(data_dir, transform=data_transform)
data_loader = torch.utils.data.DataLoader(
    data, batch_size=batch_size, shuffle=True)


# def train(model, data_loader, device, epochs=3, lr=0.0001):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     batch_train_idx = 0
#     for epoch in range(epochs):
#         for img, label in data_loader:
#             # img = img.to(device)
#             # label = label.to(device)
#             pred = model(img)
#             # print(pred.shape)
#             loss_train = F.mse_loss(pred, label)
#             loss_train.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             print('Loss train:', loss_train.item(), batch_train_idx)
#             batch_train_idx += 1


# model = SimpleAE()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train(model, data_loader, device)

# %%
model = SimpleAE()
model.load_state_dict(torch.load(
    '/Users/luluo/projects/propertySearch/models/ae-all.pt', map_location=torch.device('cpu')))
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
with open('latent.pickle', 'wb') as f:
    pickle.dump(latent, f)
# %%
with open('latent.pickle', 'rb') as f:
    latent = pickle.load(f)
# %%
# Visualising in tSNE
vis_tsne(latent[:200], 80, 60)

# %%
# Visualisation in tensorboard
vis_tb(latent[:500], 120)

# %%
