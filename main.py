# %%
import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from house_dataset import ClassificationDataset
import torch.nn.functional as F
from time import time
# %%
data = ClassificationDataset(
    transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
)
# %%
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True)

# %%
kernel_size = 7
stride = 2


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size, stride),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size),
            torch.nn.Flatten(),
            torch.nn.Linear(3200, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        # print(x.shape)
        return x


def train(model, data_loader, device, epochs=10, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=f'runs/classifier/{time()}')
    batch_idx = 0
    for epoch in range(epochs):
        for img, label in data_loader:
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss/no_trans_train', loss.item(), batch_idx)
            batch_idx += 1
            print('Loss:', loss.item())

        print(cal_acc(pred, label))


def cal_acc(pred, y):
    pred_label = torch.argmax(pred, dim=1)
    print(pred_label, y)
    return torch.sum(pred_label == y)/len(y)


cnn = Classifier()
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = cnn.to(device)

train(cnn, train_loader, device)

# %%
