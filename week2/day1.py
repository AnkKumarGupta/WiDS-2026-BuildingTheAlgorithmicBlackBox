import time
import heapq
import random
import bisect

# --- THE "BAD" ORDER BOOK (List-Based) ---
class ListOrderBook:
    def __init__(self):
        self.bids = [] 
        self.asks = []  

    def add_order(self, price, side):
        if side == 'Buy':
            self.bids.append(price)
            self.bids.sort(reverse=True) 
        else:
            self.asks.append(price)
            self.asks.sort()

# --- 2. THE "GOOD" ORDER BOOK (Heap-Based) ---
class HeapOrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []

    def add_order(self, price, side):
        if side == 'Buy':
            heapq.heappush(self.bids, -price)
        else:
            heapq.heappush(self.asks, price)

def run_benchmark(n_orders=10000):
    print(f"--- BENCHMARKING: Inserting {n_orders} Orders ---")
    
    orders = []
    for _ in range(n_orders):
        side = 'Buy' if random.random() < 0.5 else 'Sell'
        price = random.uniform(100, 200)
        orders.append((side, price))

    list_book = ListOrderBook()
    start_time = time.time()
    for side, price in orders:
        list_book.add_order(price, side)
    list_duration = time.time() - start_time
    print(f"List Implementation: {list_duration:.4f} seconds")

    heap_book = HeapOrderBook()
    start_time = time.time()
    for side, price in orders:
        heap_book.add_order(price, side)
    heap_duration = time.time() - start_time
    print(f"Heap Implementation: {heap_duration:.4f} seconds")

    # RESULTS
    if heap_duration > 0:
        speedup = list_duration / heap_duration
        print(f"\nWinner: HEAP is {speedup:.1f}x faster!")
    else:
        print("\nHeap was too fast to measure!")

if __name__ == "__main__":
    # run_benchmark(n_orders=10000)
    # run_benchmark(n_orders=50000)
    run_benchmark(n_orders=100000)
