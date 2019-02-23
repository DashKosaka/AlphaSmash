from pyfirmata import Arduino, util
from controller.arduino_driver import ArduinoDriver
from config import *
import sys, os, json, time, msvcrt, random
import time

print('Connecting')
driver = ArduinoDriver('COM4')

print('Reading')
while True:
    init_time = time.time()
        
    # Capture the user input
    user_char = None
    # print('waiting')
    while user_char is None:
        if msvcrt.kbhit():
            char_bin = msvcrt.getch()
            user_char = char_bin.decode('utf-8')

        # time.sleep(0.01)
        # second_char

    # Check if we can print the character
    if char_bin is b'\x08' or user_char is '\b':
        pass

    # Skip the current
    elif char_bin is b'\r':
        print('\nSkipping...')
    
    # Regular character
    else:
        # Movement
        if user_char is ' ':
            driver.shield()
            save=[PIN_SHIELD]

        elif user_char is 'w':
            driver.move_up()
            save=[PIN_VMOVE_ENBL, PIN_VMOVE]
        elif user_char is 'a':
            driver.move_left()
            save=[PIN_HMOVE_ENBL, PIN_HMOVE]
        elif user_char is 's':
            driver.move_down()
            save=[PIN_VMOVE_ENBL, PIN_VMOVE]
        elif user_char is 'd':
            driver.move_right()
            save=[PIN_HMOVE_ENBL, PIN_HMOVE]

        elif user_char is 'e':
            driver.grab()
            save=[PIN_GRAB]
        elif user_char is 'j':
            driver.attack()
            save=[PIN_ATTACK]
            curr_time = time.time()
            while curr_time - time.time() < 0.1:
                if msvcrt.kbhit():
                    char_bin = msvcrt.getch()
                    user_char = char_bin.decode('utf-8')
                    driver.move_left()
                    save.extend([PIN_HMOVE_ENBL, PIN_HMOVE])
                    break
        elif user_char is 'k':
            driver.special()
            save=[PIN_SPECIAL]


            # save=[PIN_VTILT_ENBL, PIN_VTILT]
            # save=[PIN_HTILT_ENBL, PIN_HTILT]
            # save=[PIN_VTILT_ENBL, PIN_VTILT]
            # save=[PIN_HTILT_ENBL, PIN_HTILT]
            # save=[PIN_VMOVE_ENBL, PIN_VMOVE, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_VMOVE_ENBL, PIN_VMOVE, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_SHIELD, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_SHIELD, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_SHIELD, PIN_VMOVE_ENBL, PIN_VMOVE]
            # save=[PIN_ATTACK, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_ATTACK, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_ATTACK, PIN_VMOVE_ENBL, PIN_VMOVE]
            # save=[PIN_ATTACK, PIN_VMOVE_ENBL, PIN_VMOVE]
            # save=[PIN_SPECIAL, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_SPECIAL, PIN_HMOVE_ENBL, PIN_HMOVE]
            # save=[PIN_SPECIAL, PIN_VMOVE_ENBL, PIN_VMOVE]
            # save=[PIN_SPECIAL, PIN_VMOVE_ENBL, PIN_VMOVE]


        # elif user_char is 'd':
        #     driver.shield()

        # elif user_char is 'f':
        #     driver.grab()

        # elif user_char is 'g':
        #     driver.tilt_left()

        # elif user_char is 'h':
        #     driver.tilt_right()


        time.sleep(0.1)
        driver.clear(save=save)


