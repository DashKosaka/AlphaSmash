from pyfirmata import Arduino, util
from controller.arduino_driver import ArduinoDriver
import sys, os, json, time, msvcrt, random
import time

# board = Arduino('COM3')
driver = ArduinoDriver('COM4')

while True:
    init_time = time.time()
        
    # Capture the user input
    user_char = None
    print('waiting')
    while user_char is None:
        if msvcrt.kbhit():
            char_bin = msvcrt.getch()
            user_char = char_bin.decode('utf-8')
    # Check if we can print the character
    if char_bin is b'\x08' or user_char is '\b':
        pass
#    else:
#        print(user_char)

    # Skip the current
    elif char_bin is b'\r':
        print('\nSkipping...')
    
    # Regular character
    else:
        if user_char is ' ':
            pass
        elif user_char is 'a':
            driver.attack()

        elif user_char is 's':
            driver.special()

        elif user_char is 'd':
            driver.shield()

        elif user_char is 'f':
            driver.grab()

        elif user_char is 'g':
            driver.tilt_left()

        elif user_char is 'h':
            driver.tilt_right()

    time.sleep(0.1)

    driver.clear()

