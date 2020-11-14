from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import torch
from PIL import Image

model = models.resnet50(pretrained=True)

print(model)
aadadf

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405],
                         std=[0.229, 0.224, 0.225])
])

train_dir = 'propertyimages'

train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)

train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=8)


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrain = models.resnet152(pretrained=True)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU()
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.
            torch.nn.ConvTranspose2d()
        )
        self.encode = None

    def forward(self, x):
        self.encode = self.encoder(x)
        x = self.decoder(self.encode)
        return x


encoder = Encoder()


def train(model, train_loader, epochs):
    optimiser = torch.optim.Adam(model.parameters())
    losses = []
    writer = SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            data, _ = batch
            # print(data.shape)
            data = data.view(-1, 784)
            pred = model(data)
            loss = F.mse_loss(pred, data)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            losses.append(loss)
            writer.add_scalar('Loss:', loss.item(), batch_idx)
            batch_idx += 1
    plt.plot(losses)
    plt.show()
    return model

#encoder = train(encoder, train_loader, 1)
#torch.save(encoder, './encoder.pth')


encoder = torch.load('./encoder.pth')

# print(train_data[0][0].shape)
img = train_data[0][0]
img_flat = img.view(-1, 784)
# print(img.shape)
#plt.imshow(img[0].numpy(), cmap='gray')
# plt.show()

pred = encoder(img_flat)
pred = pred[0].view(1, 28, 28)
# print(pred)
trans = transforms.ToPILImage()
pred_img = trans(pred)
pred_img.show()
#pred = pred.detach().numpy().reshape((1,28,28))
# print(pred.shape)
#plt.imshow(pred, cmap='gray')
# plt.show()
#accuracy = test(cnn, test_loader)
#print('Acurracy:', accuracy)
