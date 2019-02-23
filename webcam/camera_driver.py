import cv2, time
import torch
import numpy as np
from agent import Agent
from ddqn.models import ODQN, DQN

cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)
#model = Agent(10, 0, False)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    rval = False

curr_time = time.time()

#history = np.random.rand(5, 84, 84)
#dqn = ODQN(10)
#dqn(torch.Tensor(np.float32(history[:4, :, :]) / 255.).unsqueeze(0))


model = DQN()

fps = []
while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) 

    fps.append(1/(time.time() - curr_time))
    print(1/(time.time() - curr_time), (sum(fps) / len(fps)))
#    print('FPS: ', str(1/(time.time() - curr_time)))
    
    curr_time = time.time()
    
    # Testing DQN forward
#    inp = frame[:300, :540].reshape(1, 300, 540)
#    history = np.random.rand(5, 300, 540)
#    history = np.concatenate((inp, inp, inp, inp))
#    state = torch.Tensor(np.float32(history[:4, :, :]) / 255.).unsqueeze(0)
#    pred = model(state)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")