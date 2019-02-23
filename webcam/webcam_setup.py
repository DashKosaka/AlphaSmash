# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 01:28:53 2019

@author: Dash
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# 5.4 x 3.0 Screen size

# 60 degrees of camera vision

# 0.5 x 0.25 Health size
# - 0.5 from bottom
# - 0.875 & 2.0 from left

def draw_hline(arr, y, size):
    arr[y-1, size[0]:size[1]] = 255
    arr[y, size[0]:size[1]] = 0 
    arr[y+1, size[0]:size[1]] = 255
    
def draw_vline(arr, x, size):
    arr[size[0]:size[1], x-1] = 255
    arr[size[0]:size[1], x] = 0 
    arr[size[0]:size[1], x+1] = 255

def draw_screen_border(frame, x_tuple, y_tuple):
    x1, x2 = x_tuple
    y1, y2 = y_tuple
    
    # Horizontal borders
    draw_hline(frame, x1, (y1, y2))
    draw_hline(frame, x2, (y1, y2))

    # Vertical borders
    draw_vline(frame, y1, (x1, x2))
    draw_vline(frame, y2, (x1, x2))

def draw_health_border(screen, x_tuple, y_tuple):
    x1, x2 = x_tuple
    y1, y2 = y_tuple

    width = int((x2 - x1) / 3)
    
    draw_hline(screen, y1, (x1, x2))
    draw_hline(screen, y2, (x1, x2))

    draw_vline(screen, x1, (y1, y2))
    draw_vline(screen, x1 + width, (y1, y2))
    draw_vline(screen, x1 + 2*width, (y1, y2))
    # draw_vline(screen, 132, (y1, y2))
    draw_vline(screen, x2, (y1, y2))
    

frame_up = 65
frame_down = 333
frame_left = 92
frame_right = 572

h1_side = 65
h1_up = 225
h1_down = 250

h2_side = 170
h2_up = 225
h2_down = 250

while(True):
    h1_left = h1_side
    h1_right = h1_side + 40
    h2_left = h2_side
    h2_right = h2_side + 40

    # Capture frame-by-frame
    ret, frame = cap.read()
    original = np.copy(frame)

    # Draw borders of screen crop
    draw_screen_border(frame, (frame_up, frame_down), (frame_left, frame_right))

    # Display the resulting frame
    cv2.imshow('Cropping Screen', frame)

    # Crop out the screen    
    screen = original[frame_up:frame_down, frame_left:frame_right]
    health = np.copy(screen)
    
    # Own health borders
    draw_health_border(screen, (h1_left, h1_right), (h1_up, h1_down))

    # Enemy health borders
    draw_health_border(screen, (h2_left, h2_right), (h2_up, h1_down))
    
    # Display the captured screen
    cv2.imshow('Game Screen', screen)

    cv2.imshow('Number', health[h2_up:h2_down, h2_side:h2_side+15])
    
    key = cv2.waitKey(100) & 0xFF
    if key == 27: # Escape to quit
        break
    # Adjust top
    elif key == ord('q'):
        frame_up += -1
    elif key == ord('a'):
        frame_up += 1
    # Adjust bottom
    elif key == ord('w'):
        frame_down += -1
    elif key == ord('s'):
        frame_down += 1
    # Adjust left
    elif key == ord('e'):
        frame_left += -1
    elif key == ord('r'):
        frame_left += 1
    # Adjust right
    elif key == ord('d'):
        frame_right += -1
    elif key == ord('f'):
        frame_right += 1

    elif key == ord('u'):
        h1_side += -1
    elif key == ord('i'):
        h1_side += 1
    elif key == ord('j'):
        h2_side += -1
    elif key == ord('k'):
        h2_side += 1
    
        
print('Final Values')
print(frame_up)
print(frame_down)
print(frame_left)
print(frame_right)

print(h1_left)
print(h1_right)
print(h1_up)
print(h1_down)

print(h2_left)
print(h2_right)
print(h2_up)
print(h2_down)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
