import numpy as np
from src.dataGenius.process_data import FinancialProcessor

class DataFrameBuilder3000:
    def __init__(self, tickers, start_year, end_year):
        self.tickers = tickers
        self.start_year = start_year
        self.end_year = end_year

    def process_historical_ticker(self, ticker):
        history = []
        for year in range(self.start_year, self.end_year + 1):
            try:
                processor = FinancialProcessor(ticker, year)
                print(f"Processing {ticker} {year}...")
                result = processor.process_ticker()
                if result and result['tensor'] is not None:
                    history.append(result['tensor'])
            except Exception as e:
                print(f"Failed on {ticker} {year}: {e}")
        return np.array(history)

    def build_seq2seq_dataset(self, window_size=3):
        """
        Converts raw history into [Input: 3 Years] -> [Target: 3 Years]
        """
        X, Y = [], []
        
        for ticker in self.tickers:
            history = self.process_historical_ticker(ticker)
            if len(history) < (window_size * 2):
                print(f"Not enough data for {ticker} to build sequences.")
                continue
                
            for i in range(len(history) - (window_size * 2) + 1):
                input_seq = history[i : i + window_size]
                target_seq = history[i + window_size : i + (window_size * 2)]
                
                X.append(input_seq)
                Y.append(target_seq)
                
        return np.array(X), np.array(Y)