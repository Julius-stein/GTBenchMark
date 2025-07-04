import time
import numpy as np
from contextlib import contextmanager


class Timer(object):

    def __init__(self):
        super(Timer, self).__init__()
        self._time = {}

    @contextmanager
    def timer(self, name):
        if not name in self._time:
            self._time[name] = []
        t0 = time.time()
        yield
        t1 = time.time()
        self._time[name].append(t1-t0)
    
    @contextmanager
    def accumulate_timer(self, name):
        name += '_temp'
        if not name in self._time:
            self._time[name] = []
        t0 = time.time()
        yield
        t1 = time.time()
        self._time[name].append(t1-t0)
    
    def add_time(self, name, num):
        if not name in self._time:
            self._time[name] = []
        
        self._time[name].append(num)

    def reduce_time(self):
        re = []
        for (k, tlist) in self._time.items():
            if len(tlist) == 0: continue
            if '_temp' in k:
                name = k[:-5]
                re.append((name, np.sum(tlist)))
                self._time[k] = []
        for (x,y) in re:
            self.add_time(x,y)
    

    def tot_time(self):
        tot = 0
        for tlist in self._time.values():
            if len(tlist) == 0: continue
            tot += np.sum(tlist)
        return tot

    def print_time(self):
        for (k, tlist) in self._time.items():
            if len(tlist) == 0: continue
            print(f'time of {k}: {np.mean(tlist)} seconds.')

    def clear(self):
        self._time = {}