import cv2
from ocr.ocr import *

image = cv2.imread('./ocr/numbers/1/img2.jpeg')
print(image.shape)

image_tensor = torch.Tensor(image).view(1, 3, 25, 14)
print(ocr_model(image_tensor))