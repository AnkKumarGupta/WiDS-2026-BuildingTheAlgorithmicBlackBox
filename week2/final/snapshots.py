class SnapshotRecorder:
    def __init__(self):
        self.records = []
        
    def record(self, timestamp, best_bid, best_ask):
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
        spread = best_ask - best_bid if (best_bid and best_ask) else None
        
        self.records.append({
            'timestamp': timestamp,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid,
            'spread': spread
        })

    def get_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)