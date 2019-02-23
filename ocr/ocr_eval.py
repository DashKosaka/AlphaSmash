from ocr_train_pytorch import CNN
import torch
import torchvision.transforms as transforms
import cv2
import json
import random
import numpy as np


ocr_model = CNN()
ocr_model.load_state_dict(torch.load("./alphasmash_ocr.pth"))

testing = cv2.imread("./numbers/5/img223478.jpeg")
testing = cv2.resize(testing, (14, 25))
testing = np.true_divide(testing, 255)
testing = torch.Tensor(testing)
testing = testing.unsqueeze(0)
print(testing.size())
print(ocr_model(testing).max(1)[1])