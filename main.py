# %%
import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from house_dataset import ClassificationDataset
import torch.nn.functional as F
# %%
data = ClassificationDataset(
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)
# %%
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True)

# %%


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 7),
            torch.nn.Flatten(),
            torch.nn.Linear(3625216, 3),
            torch.nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


def train(model, data_loader, epochs=1, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir='runs')
    batch_idx = 0
    for epoch in range(epochs):
        for img, label in data_loader:
            pred = model(img)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss/no_trans_train', loss.item(), batch_idx)
            batch_idx += 1
            print('Loss:', loss.item())


cnn = Classifier()
cnn.
train(cnn, train_loader)

# %%
