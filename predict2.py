import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN import CNN
from MLP import MLP


def plot_confusion_matrix(model, test_loader, device, title='Confusion Matrix'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=False,
                fmt='d',
                cmap='Blues',
                vmax= 10,
                cbar=True,
                square=True,
                xticklabels=range(10),
                yticklabels=range(10),
                linewidths=0.5,
                linecolor='lightgray',
                cbar_kws={"shrink": .8}
                )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST('./data', train=False, download=True, transform=tf)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    cnn_model = CNN().to(device)
    cnn_model.load_state_dict(torch.load("cnn_weights/cnn_100rd.pth", map_location=device,weights_only=True))
    plot_confusion_matrix(cnn_model, test_loader, device, title='CNN Confusion Matrix')
    #
    # mlp_model = MLP().to(device)
    # mlp_model.load_state_dict(torch.load("mlp_weights/mlp_100rd.pth", map_location=device,weights_only=True))
    # plot_confusion_matrix(mlp_model, test_loader, device, title='MLP Confusion Matrix')