import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MLP import MLP
from CNN import CNN
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=tf), batch_size=128,shuffle=True)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 100
    for i in range(1, epochs + 1):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), labels).backward()
            opt.step()
        print(f"第{i}轮ok")
        torch.save(model.state_dict(), f"mlp_weights/mlp_{i}rd.pth")
if __name__ == "__main__":
    train()