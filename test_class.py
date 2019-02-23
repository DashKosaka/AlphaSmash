import cv2
from ocr.ocr import *

image = cv2.imread('./testing.jpeg')
print(image.shape)

cv2.imshow('normal', image)
cv2.waitKey(10)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('BW', image)
cv2.waitKey(10)

image[image<255//2] = 0
image[image>=255//2] = 255
cv2.imshow('masked', image)
cv2.waitKey(10)

while(True):
	pass
# image_tensor = torch.Tensor(image).view(1, 3, 25, 14)
# print(ocr_model(image_tensor))