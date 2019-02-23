from ocr.ocr_train_pytorch import CNN
import torch
import torchvision.transforms as transforms
import cv2
import json
import random

with open('cropping.json') as f:
    HP_NUM_COORDS = json.load(f)

ocr_model = CNN()
ocr_model.load_state_dict(torch.load("./ocr/alphasmash_ocr.pth"))

INPUT_DIMENSIONS = (50, 50)
HP_BOX_WIDTH = 15

def torchify_tensors(tensors):
    for idx in range(len(tensors)):
        tensors[idx] = cv2.resize(tensors[idx], INPUT_DIMENSIONS)
        # img = torch.Tensor(img)
    tensors = torch.Tensor(tensors)
    return tensors

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
    
    cv2.imshow("digit_1", digit_1)
    cv2.imshow("digit_2", digit_2)
    cv2.imshow("digit_3", digit_3)

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

    cv2.imshow("enemy_digit_1", enemy_digit_1)
    cv2.imshow("enemy_digit_2", enemy_digit_2)
    cv2.imshow("enemy_digit_3", enemy_digit_3)

    while(True):
        key = cv2.waitKey(100) & 0xFF

        if key == ord('f'):
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", digit_1)
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", digit_2)
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", digit_3)
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", enemy_digit_1)
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", enemy_digit_2)
            cv2.imwrite("./images/img" + str(random.randint(1,1000000)) + ".jpeg", enemy_digit_3)

        elif key == ord('d'):
            break

    digit_tensors = torchify_tensors(all_digits).view(-1, 3, INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1])
    # ret = ocr_model(digit_tensors).max(1)[1]
    # print(ret)
    # return ret