# %%
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, models
from house_dataset import ClassificationDataset
import torch.nn.functional as F
import torch
from PIL import Image
from time import time
import numpy as np

# %%
trans_model = models.resnet50(pretrained=True)

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405],
                         std=[0.229, 0.224, 0.225])
])

train_dir = 'propertyimages/train'
val_dir = 'propertyimages/validation'
batch_size = 32

train_data = ClassificationDataset(
    csv='./propertyimages/train/labels.csv',
    transform=data_transform
)

val_data = ClassificationDataset(
    csv='./propertyimages/validation/labels.csv',
    transform=data_transform
)

train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size)

val_loader = torch.utils.data.DataLoader(
    val_data, shuffle=True, batch_size=batch_size)

# img = train_data[0][0]
# topil = transforms.ToPILImage()
# topil(img).show()

# %%


class TransClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = trans_model
        self.layers.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def freeze(self):
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.layers.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.layers.parameters():
            param.requires_grad = True


def train(model, data_loader, device, epochs=3, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=f'runs/trans_classifier/{time()}')
    batch_train_idx = 0
    batch_train_idx = 0
    acc = []
    for epoch in range(epochs):
        for img, label in data_loader:
            # img = img.to(device)
            # label = label.to(device)
            pred_train = model(img)
            loss_train = F.cross_entropy(pred_train, label)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss/trans_train',
                              loss_train.item(), batch_train_idx)
            batch_train_idx += 1
            print('Loss train:', loss_train.item())

        for img, label in val_loader:
            model.eval()
            pred_val = model(img)
            loss_val = F.cross_entropy(pred_val, label)
            writer.add_scalar('Loss/trans_val',
                              loss_val.item(), batch_val_idx)
            print('Loss val :', loss_val.item())
            acc.append(cal_acc(pred_val, label))
            batch_val_idx += 1

    print('Validation acc is', np.mean(acc))


def cal_acc(pred, y):
    pred_label = torch.argmax(pred, dim=1).numpy()
    print(pred_label, y)
    return np.mean(pred_label == y.numpy())


tcnn = TransClassifier()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tcnn = tcnn.to(device)
tcnn.freeze()
train(tcnn, train_loader, device, 3)

# %%

# %%
