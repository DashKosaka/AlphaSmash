from controller.arduino_driver import ArduinoDriver
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

class Player(ArduinoDriver):

    def __init__(self, port):
        super(Player, self).__init__(port)
        self.is_idle = False

    '''
    *** Do Nothing
    '''
    def idle(self):
        self.clear()

    '''
    *** Base Actions
    '''
    def shield(self):
        self.clear(save=[PIN_SHIELD])
        super().shield()

    def grab(self):
        self.clear(save=[PIN_GRAB])
        super().grab()

    def attack(self):
        self.clear(save=[PIN_ATTACK])
        super().attack()

    def special(self):
        self.clear(save=[PIN_SPECIAL])
        super().special()

    def move_up(self):
        self.clear(save=[PIN_VMOVE_ENBL, PIN_VMOVE])
        super().move_up()

    def move_left(self):
        self.clear(save=[PIN_HMOVE_ENBL, PIN_HMOVE])
        super().move_left()

    def move_down(self):
        self.clear(save=[PIN_VMOVE_ENBL, PIN_VMOVE])
        super().move_down()

    def move_right(self):
        self.clear(save=[PIN_HMOVE_ENBL, PIN_HMOVE])
        super().move_right()

    def tilt_up(self):
        self.clear(save=[PIN_VTILT_ENBL, PIN_VTILT])
        super().tilt_up()

    def tilt_left(self):
        self.clear(save=[PIN_HTILT_ENBL, PIN_HTILT])
        super().tilt_left()

    def tilt_down(self):
        self.clear(save=[PIN_VTILT_ENBL, PIN_VTILT])
        super().tilt_down()

    def tilt_right(self):
        self.clear(save=[PIN_HTILT_ENBL, PIN_HTILT])
        super().tilt_right()

    '''
    *** Combo Movement
    '''
    def up_left(self):
        self.clear(save=[PIN_VMOVE_ENBL, PIN_VMOVE, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().move_up()
        super().move_left()

    def up_right(self):
        self.clear(save=[PIN_VMOVE_ENBL, PIN_VMOVE, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().move_up()
        super().move_right()
        
    def dodge_left(self):
        self.clear(save=[PIN_SHIELD, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().shield()
        super().move_left()
        
    def dodge_right(self):
        self.clear(save=[PIN_SHIELD, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().shield()
        super().move_right()
        
    def dodge_down(self):
        self.clear(save=[PIN_SHIELD, PIN_VMOVE_ENBL, PIN_VMOVE])
        super().shield()
        super().move_down()

    '''
    *** Combo Attacks
    '''
    def left_smash(self):
        self.clear(save=[PIN_ATTACK, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().attack()
        super().move_left()

    def right_smash(self):
        self.clear(save=[PIN_ATTACK, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().attack()
        super().move_right()

    def down_smash(self):
        self.clear(save=[PIN_ATTACK, PIN_VMOVE_ENBL, PIN_VMOVE])
        super().attack()
        super().move_down()

    def up_smash(self):
        self.clear(save=[PIN_ATTACK, PIN_VMOVE_ENBL, PIN_VMOVE])
        super().attack()
        super().move_up()
        
    def left_special(self):
        self.clear(save=[PIN_SPECIAL, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().special()
        super().move_left()

    def right_special(self):
        self.clear(save=[PIN_SPECIAL, PIN_HMOVE_ENBL, PIN_HMOVE])
        super().special()
        super().move_right()

    def down_special(self):
        self.clear(save=[PIN_SPECIAL, PIN_VMOVE_ENBL, PIN_VMOVE])
        super().special()
        super().move_down()

    def up_special(self):
        self.clear(save=[PIN_SPECIAL, PIN_VMOVE_ENBL, PIN_VMOVE])
        super().special()
        super().move_up()
