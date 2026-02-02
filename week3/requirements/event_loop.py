import heapq

class SimulationKernel:
    def __init__(self):
        self.time = 0.0
        self.events = [] # Heap: (timestamp, sequence, function)
        self.seq = 0

    def schedule(self, delay, func):
        timestamp = self.time + delay
        heapq.heappush(self.events, (timestamp, self.seq, func))
        self.seq += 1

    def run(self, duration):
        while self.events:
            t, _, func = heapq.heappop(self.events)
            
            if t < self.time: raise RuntimeError("Time Travel detected!")
            
            self.time = t
            if self.time > duration: break
            
            func()