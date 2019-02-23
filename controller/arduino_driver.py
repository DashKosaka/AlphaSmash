# Arduino Driver
import time
from pyfirmata import Arduino, util
from config import *

'''
PIN_VMOVE_ENBL 
PIN_VMOVE 
PIN_HMOVE_ENBL 
PIN_HMOVE
PIN_ATTACK
PIN_SPECIAL
PIN_HTILT_ENBL
PIN_HTILT
PIN_VTILT_ENBL
PIN_VTILT
PIN_GRAB
PIN_SHIELD
'''

class ArduinoDriver(Arduino):

    def __init__(self, port):
        super(ArduinoDriver, self).__init__(port)

        self.clear()

    def clear(self, save=[]):
        for pin in range(2, 14):
            if pin not in save:
                self.digital[pin].write(OFF)        

    '''
    *** Attack Functions
    '''
    def attack(self):
        self.digital[PIN_ATTACK].write(ON)

    def attack_(self):
        self.digital[PIN_ATTACK].write(OFF)

    '''
    *** Special Functions
    '''
    def special(self):
        self.digital[PIN_SPECIAL].write(ON)

    def special_(self):
        self.digital[PIN_SPECIAL].write(OFF)

    '''
    *** Shield Functions
    '''
    def shield(self):
        self.digital[PIN_SHIELD].write(ON)

    def shield_(self):
        self.digital[PIN_SHIELD].write(OFF)

    '''
    *** Grab Functions
    '''
    def grab(self):
        self.digital[PIN_GRAB].write(ON)

    def grab_(self):
        self.digital[PIN_GRAB].write(OFF)

    '''
    *** Move Functions
    '''
    def move_left(self):
        self.digital[PIN_HMOVE].write(LEFT)
        self.digital[PIN_HMOVE_ENBL].write(ON)        

    def move_left_(self):
        self.digital[PIN_HMOVE_ENBL].write(OFF)

    def move_right(self):
        self.digital[PIN_HMOVE].write(RIGHT)
        self.digital[PIN_HMOVE_ENBL].write(ON)

    def move_right_(self):
        self.digital[PIN_HMOVE_ENBL].write(OFF)

    def move_up(self):
        self.digital[PIN_VMOVE].write(UP)
        self.digital[PIN_VMOVE_ENBL].write(ON)

    def move_up_(self):
        self.digital[PIN_VMOVE_ENBL].write(OFF)

    def move_down(self):
        self.digital[PIN_VMOVE].write(DOWN)
        self.digital[PIN_VMOVE_ENBL].write(ON)

    def move_down_(self):
        self.digital[PIN_VMOVE_ENBL].write(OFF)
        
    '''
    *** Tilt Functions
    '''
    def tilt_left(self):
        self.digital[PIN_HTILT].write(LEFT)
        self.digital[PIN_HTILT_ENBL].write(ON)        

    def tilt_left_(self):
        self.digital[PIN_HTILT_ENBL].write(OFF)

    def tilt_right(self):
        self.digital[PIN_HTILT].write(RIGHT)
        self.digital[PIN_HTILT_ENBL].write(ON)

    def tilt_right_(self):
        self.digital[PIN_HTILT_ENBL].write(OFF)

    def tilt_up(self):
        self.digital[PIN_VTILT].write(UP)
        self.digital[PIN_VTILT_ENBL].write(ON)

    def tilt_up_(self):
        self.digital[PIN_VTILT_ENBL].write(OFF)

    def tilt_down(self):
        self.digital[PIN_VTILT].write(DOWN)
        self.digital[PIN_VTILT_ENBL].write(ON)

    def tilt_down_(self):
        self.digital[PIN_VTILT_ENBL].write(OFF)


