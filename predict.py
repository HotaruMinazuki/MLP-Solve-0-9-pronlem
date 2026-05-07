import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
from model import MLP


def predict():
    # 1. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    model.load_state_dict(torch.load("mlp.pth", map_location=device,weights_only = True))
    model.eval()
    img_path = os.path.join("input", os.listdir("input")[0])
    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY_INV)
    img_pil = Image.fromarray(binary)
    tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        print(torch.argmax(output, dim=1).item())
if __name__ == "__main__":
    predict()