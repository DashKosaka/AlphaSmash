import cv2, time
import torch
import numpy as np
from collections import deque
from webcam.utils import *
from ddqn.agent import *
from config import *
from player import Player
from ocr.ocr import *

# Cropping coords
screen_top = HP_NUM_COORDS["screen"]["top"]
screen_left = HP_NUM_COORDS["screen"]["left"]
screen_bottom = HP_NUM_COORDS["screen"]["bottom"]
screen_right = HP_NUM_COORDS["screen"]["right"]
# torch.Size([6, 3, 14, 25])
# Setup video capture
cap = cv2.VideoCapture(0)

# Setup model
agent = Agent(epsilon=0, load_model=False)
fps = deque(maxlen=100)
combo_player = Player()

quit = False

curr_time = time.time()

# Reset history
history = np.zeros([5, INPUT_Y, INPUT_X])

# Reset counters and life
action_timer = time.time()

while True:
    # Show the frame real-time
    rval, frame = cap.read()
    frame = frame[screen_top:screen_bottom, screen_left:screen_right]
    # print(frame.shape)
    cv2.imshow("Game", frame)

    # FPS Counter 
    fps.append(1/(time.time() - curr_time))

    # Reset time counter
    curr_time = time.time()

    # Option to quit    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

    # Check if we are allowed to make an action
    idle_time = time.time() - action_timer
    if idle_time < ACTION_WAIT:
        continue
    action_timer = time.time()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    get_health(frame)
    frame = cv2.resize(frame, dsize=(INPUT_X, INPUT_Y))
    # print(frame.shape)
    cv2.imshow("Input", frame)
    history[:4] = history[-4:]
    history[-1] = frame

    
    # Get action
    action = agent.get_action(history[:4])
    # print(action)

    # Perform action
    combo_player.do(action)

finish(cap)