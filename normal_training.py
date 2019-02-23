import cv2, time
import torch
import numpy as np
from collections import deque
from webcam.utils import *
from ddqn.agent import *
from config import *
from player import Player

# Setup video capture
cap = cv2.VideoCapture(0)

# Setup model
agent = StateAgent(False)
frame_count = 0
fps = deque(maxlen=100)
combo_player = Player()

quit = False
#evaluation_reward = deque(maxlen=EVALUATION_REWARD_LENGTH)

curr_time = time.time()
# Training loop
while quit is False:
    # Both players are alive again
    alive = True

    # Reset history
    history = np.zeros([5, INPUT_Y, INPUT_X])
    
    # Reset counters and life
    ### TODO ###
    score = 0
    step = 0
    own_health = 0
    enemy_health = 0

    action_timer = time.time()

    while alive:
        # Show the frame real-time
        rval, frame = cap.read()
        cv2.imshow("Game", frame)

        # FPS Counter 
        fps.append(1/(time.time() - curr_time))
#        print(1/(time.time() - curr_time), (sum(fps) / len(fps)))
#        print('FPS: ', str(1/(time.time() - curr_time)))

        # Reset time counter
        curr_time = time.time()

        # Option to quit    
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            quit = True
            break

        # Check if we are allowed to make an action
        idle_time = time.time() - action_timer
        if idle_time < ACTION_WAIT:
            continue
        action_timer = time.time()

        # Increment counters
        frame_count += 1
        step += 1

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, dsize=(INPUT_X, INPUT_Y))
        history[:4] = history[-4:]
        history[-1] = frame
        
        # Get action
        action = agent.get_action(history)

        # Perform action
        agent.do(action)

'''
        # Generate reward
        ### TODO ###

        # Check for changes in health
#        healths = get_health(frame)
        healths = (0, 0)

        if healths[0] == own_health and healths[1] == enemy_health:
            # Reward from damage
            reward = (healths[0] - own_health) * REWARD_DMG_TAKEN
            reward += (healths[1] - enemy_health) * REWARD_DMG_GIVEN

            # Update healths
            own_health = healths[0]
            enemy_health = healths[1]
        # elif: # Somebody has died
            # reward = REWARD_KILL_
        else:
            # Calculate idle penalty
            reward = idle_time * REWARD_IDLE

        # Check if health reset
        ### TODO ###

        # Check if both players are alive
        ### TODO ###
#        alive = is_alive(frame)
        alive = True
'''

        # Add whole current state to network memory
        agent.memory.push(np.copy(frame), None, reward, alive)
        output = agent.policy_net(torch.Tensor(history[:4]).unsqueeze(0).to(device))
        print('Forwarding:', output)

        # # Take a break to update network
        # if frame_count >= FRAMES_REQUIRED and frame_count % UPDATE_FREQUENCY == 0:
        #     # First pause the game
        #     ### TODO ###
        #     pause()
        #     agent.train_policy_net(frame_count)
        #     if frame_count % TRANSFER_FREQUENCY == 0:
        #         # Save a checkpoint of the network
        #         agent.update_target_net()

    # Save checkpoint of model
    print('Saving checkpoint')
    torch.save(agent.policy_net, STATE_DQN_PATH)

# Save the final network

finish(cap)