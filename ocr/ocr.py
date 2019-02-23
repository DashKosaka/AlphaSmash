from ocr_train_pytorch import CNN
import torch
import torchvision.transforms as transforms
import cv2

model = torch.load("./alphasmash_ocr.pth")

def torchify_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = torch.Tensor(img)
    img = img.view(1, 1, img.size(0), img.size(1))
    return(img)

print(model(torchify_image("./numbers/0/img0001.jpeg")))