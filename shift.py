import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN import CNN
from MLP import MLP


def test_robustness():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tf), batch_size=256,
                             shuffle=False)
    cnn_model = CNN().to(device)
    cnn_model.load_state_dict(torch.load("cnn_weights/cnn_100rd.pth", map_location=device, weights_only=True))
    cnn_model.eval()
    mlp_model = MLP().to(device)
    mlp_model.load_state_dict(torch.load("mlp_weights/mlp_100rd.pth", map_location=device, weights_only=True))
    mlp_model.eval()
    shifts = [0, 1, 2, 3]
    cnn_accs, mlp_accs = [], []
    for shift in shifts:
        cnn_correct, mlp_correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                if shift > 0:
                    imgs = TF.affine(imgs, angle=0, translate=[shift, 0], scale=1.0, shear=0)

                cnn_pred = cnn_model(imgs).argmax(dim=1)
                cnn_correct += (cnn_pred == labels).sum().item()

                mlp_pred = mlp_model(imgs).argmax(dim=1)
                mlp_correct += (mlp_pred == labels).sum().item()

                total += labels.size(0)

        cnn_accs.append(cnn_correct / total * 100.0)
        mlp_accs.append(mlp_correct / total * 100.0)

    plt.figure(figsize=(8, 6))
    plt.plot(shifts, cnn_accs, marker='o', linewidth=2, color='blue', label='CNN Robustness')
    plt.plot(shifts, mlp_accs, marker='o', linewidth=2, linestyle='-', color='red', label='MLP Robustness')
    plt.title('Robustness to Image Translation (CNN vs MLP)', fontsize=14)
    plt.xlabel('Shift Pixels (Right)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(shifts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_robustness()