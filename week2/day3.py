import heapq
import itertools
import random

class Event:
    _id_counter = itertools.count()

    def __init__(self, timestamp, priority, action, description):
        self.id = next(self._id_counter)
        self.timestamp = timestamp
        self.priority = priority # Lower number = Higher priority 
        self.action = action     
        self.description = description

    def __lt__(self, other):
        # Sorting by Time -> Priority -> Creation Order
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.id < other.id

class SimulationKernel:
    def __init__(self):
        self.current_time = 0.0
        self.event_queue = [] 

    def schedule(self, delay, priority, action, description):
        execution_time = self.current_time + delay
        event = Event(execution_time, priority, action, description)
        heapq.heappush(self.event_queue, event)
        print(f"[SCHEDULER] Scheduled: {description} @ t={execution_time:.3f}")

    def run(self, max_time=None):
        print(f"\n--- SIMULATION START (t={self.current_time}) ---")
        while self.event_queue:
            next_event = self.event_queue[0]
            
            if max_time and next_event.timestamp > max_time:
                print(f"--- TIME LIMIT REACHED (t={max_time}) ---")
                break

            heapq.heappop(self.event_queue)
            self.current_time = next_event.timestamp
            
            print(f"[t={self.current_time:.3f}] Executing: {next_event.description}")
            next_event.action()

        print(f"--- SIMULATION END (t={self.current_time:.3f}) ---\n")

class SimpleEngine:
    def process(self, order_name):
        print(f"    -> ENGINE: Processed {order_name}")

# --- LATENCY ARBITRAGE ---
# Scenario:
# Trader A is a Slow Institution. They decide to buy at t=0.
# Trader B is a Fast HFT. They detect A's intent and decide to buy at t=0.1.
# But, A has 200ms latency, B has 5ms latency.
# Trying to find, Who gets filled first?

kernel = SimulationKernel()
engine = SimpleEngine()

def send_order(trader_name, latency_ms):

    delay = latency_ms / 1000.0 
    
    action = lambda: engine.process(f"Order from {trader_name}")
    
    kernel.schedule(delay, priority=1, action=action, description=f"{trader_name} Order Arrival")
    print(f"[{trader_name}] Sent Order (Latency: {latency_ms}ms)")


kernel.current_time = 0.0
send_order("Trader A", latency_ms=200)

kernel.current_time = 0.1
send_order("Trader B", latency_ms=5)

kernel.run()