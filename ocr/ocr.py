from ocr_train_pytorch import CNN
import torch
import torchvision.transforms as transforms
import cv2
import json

with open('cropping.json') as f:
    HP_NUM_COORDS = json.load(f)

ocr_model = torch.load("./alphasmash_ocr.pth")

INPUT_DIMENSIONS = (50, 50)
HP_BOX_WIDTH = 15

def torchify_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_DIMENSIONS)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = torch.Tensor(img)
    img = img.view(1, 1, img.size(0), img.size(1))
    return(img)

def torchify_tensor(tensors):
	for tensor in tensors:
	    img = cv2.resize(tensor, INPUT_DIMENSIONS)
	    img = torch.Tensor(img)
	    img = img.view(1, 1, img.size(0), img.size(1))
    return(tensors)

def get_health(frame):
	parent_health = HP_NUM_COORDS["hp"]
	top_coord, bottom_coord, right_coord, left_coord = \
		parent_health["top"], parent_health["bottom"], parent_health["right"], parent_health["left"]

	enemy_health = HP_NUM_COORDS["enemy_hp"]
	enemy_top, enemy_bottom, enemy_right, enemy_left = \
		enemy_health["top"], enemy_health["bottom"], enemy_health["right"], enemy_health["left"]

	all_digits = []

	digit_1 = frame[top_coord : bottom_coord, left_coord : left_coord + HP_BOX_WIDTH]

	center = (left_coord + right_coord)//2
	digit_2 = frame[top_coord : bottom_coord, (center - HP_BOX_WIDTH//2) : (center + HP_BOX_WIDTH//2)]

	digit_3  = frame[top_coord: bottom_coord, right_coord - HP_BOX_WIDTH : right_coord]
	
	all_digits.append(digit_1)
	all_digits.append(digit_2)
	all_digits.append(digit_3)

	### ENEMY DIGITS

	enemy_digit_1 = frame[enemy_top : enemy_bottom, enemy_left : enemy_left + HP_BOX_WIDTH]

	enemy_center = (enemy_left + enemy_right)//2
	enemy_digit_2 = frame[enemy_top : enemy_bottom, (enemy_center - HP_BOX_WIDTH//2) : (enemy_center + HP_BOX_WIDTH//2)]

	enemy_digit_3  = frame[enemy_top: enemy_bottom, enemy_right - HP_BOX_WIDTH : enemy_right]

	all_digits.append(enemy_digit_1)
	all_digits.append(enemy_digit_2)
	all_digits.append(enemy_digit_3)

	return (ocr_model(torchify_tensor(all_digits)).max(1))