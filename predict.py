import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from model import MLP


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    model.load_state_dict(torch.load("mlp_mnist.pth", weights_only=True))
    model.eval()

    input_dir = "input"
    files = os.listdir(input_dir)
    img_path = os.path.join(input_dir, files[0])

    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        padding = 20
        digit_crop = binary[max(0, y - padding):y + h + padding, max(0, x - padding):x + w + padding]
        kernel = np.ones((3, 3), np.uint8)
        digit_crop = cv2.dilate(digit_crop, kernel, iterations=1)

        side = max(digit_crop.shape)
        canvas = np.zeros((side, side), dtype=np.uint8)
        oy, ox = (side - digit_crop.shape[0]) // 2, (side - digit_crop.shape[1]) // 2
        canvas[oy:oy + digit_crop.shape[0], ox:ox + digit_crop.shape[1]] = digit_crop
        final_img = canvas
    else:
        final_img = binary

    img_pil = Image.fromarray(final_img)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    print(prediction)


if __name__ == "__main__":
    predict()