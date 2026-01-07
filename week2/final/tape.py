class Tape:
    def __init__(self):
        self.records = []
    
    def record(self, trade):
        self.records.append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'qty': trade.qty
        })
    
    def get_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)