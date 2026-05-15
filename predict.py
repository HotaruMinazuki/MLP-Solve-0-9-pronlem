import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from CNN import CNN
from MLP import MLP
def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=tf),
        batch_size=256, shuffle=False
    )
    model = MLP().to(device)
    accuracies_mlp = []
    for num in range(1,100 + 1):
        model_path = f"mlp_weights/mlp_{num}rd.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100.0
        accuracies_mlp.append(acc)
        print(f'mlp{num}的准确率是 {acc}%')
    model2 = CNN().to(device)
    accuracies_cnn = []
    for num in range(1, 100 + 1):
        model_path = f"cnn_weights/cnn_{num}rd.pth"
        model2.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model2(imgs)
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100.0
        accuracies_cnn.append(acc)
        print(f'cnn{num}的准确率是 {acc}%')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 100 + 1),accuracies_mlp, color='r', linestyle='-', linewidth=2,label='MLP Test Accuracy')
    plt.plot(range(1, 100 + 1), accuracies_cnn, color='b', linestyle='-', linewidth=2,label='CNN Test Accuracy')
    plt.title('CNN vs MLP')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
if __name__ == "__main__":
    predict()