import time


class TimeProfiler(object):
    def __init__(self):
        self.t0 = None
        self.t1 = None

    def start_measurement(self):
        self.t0 = time.time()

    def end_measurement(self):
        self.t1 = time.time()

    def get_time(self):
        assert self.t0 is not None and self.t1 is not None
        delta_t = self.t1 - self.t0
        return delta_t
