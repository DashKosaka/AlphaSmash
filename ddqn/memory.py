from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
    
    def push(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= MEMORY_CAPACITY:
            sample_range = MEMORY_CAPACITY
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), BATCH_SIZE)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    def __len__(self):
        return len(self.memory)