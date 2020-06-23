import numpy as np
import random


class Easy21:
    def __init__(self):
        self.cardValMin, self.cardValMax = 1, 10
        self.validValMin, self.validValMax = 1, 21
        self.dealerValPass = 17

    def setup(self):
        # Initializes game. Deals out starting cards. s0, s1 = dealer, player.
        card_dealer = random.randint(self.cardValMin, self.cardValMax)
        card_player = random.randint(self.cardValMin, self.cardValMax)
        s = (card_dealer, card_player)
        return s

    def check_terminal(self, s):
        # Checks to see if terminal state has been reached.
        if s[0] < self.validValMin or s[0] > self.validValMax:
            r = 1
            return True, r
        elif s[1] < self.validValMin or s[1] > self.validValMax:
            r = -1
            return True, r
        elif s[0] >= self.dealerValPass:
            if s[0] == s[1]:
                r = 0
            elif s[0] > s[1]:
                r = -1
            else:
                r = 1
            return True, r
        else:
            r = 0
            return False, r

    def draw(self):
        # Draws a card according to an equal distribution across values 1-10
        # and a 1 to 2 distribution across red and black.
        value = random.randint(1, 10)
        if np.random.random() <= 1 / 3:
            return -value
        else:
            return value

    def step(self, s, a):
        # Performs one step of the game and returns the reward and next state.
        # 0 = Hit, 1 = Stick
        if a == 0:
            val = s[1] + self.draw()
            s = (s[0], val)
            terminal, r = self.check_terminal(s)
            if terminal:
                s = 'terminal'
        else:
            terminal = False
            while not terminal:
                val = s[0] + self.draw()
                s = (val, s[1])
                terminal, r = self.check_terminal(s)
            s = 'terminal'
        return r, s
