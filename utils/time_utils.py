# -*- coding: utf-8 -*-

import time


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception("{} is not in the clock.".format(key))
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


def time_since(last_time):
    time_elapsed = time.time() - last_time
    current_time = time.time()
    return current_time, time_elapsed
