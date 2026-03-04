import numpy as np
from dataGenius.process_data import FinancialProcessor
from scripts.collect_data import DataCollector
import edgar

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


if __name__ == "__main__":
    start_year = 2017
    end_year = 2025
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example tickers - replace with actual S&P 500 list
    print("Input Name (for SEC identity): ", end="")
    Name = input()
    print("Input Email (for SEC identity): ", end="")
    Email = input()
    print("Starting data collection and dataset building...")
    edgar.set_identity(Name + " " + Email)  # Set identity for SEC access
    for ticker in tickers:
        print(f"Processing {ticker} from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            collector = DataCollector(ticker, year)
            if collector.save_data():
                print(f"  [+] Data collected for {ticker} {year}")
            else:
                print(f"  [-] Failed to collect data for {ticker} {year}")
    
    builder = DataFrameBuilder3000(tickers, start_year, end_year)

    # Generate 3-year inputs -> 3-year targets
    X_train, Y_train = builder.build_seq2seq_dataset(window_size=3)
    
    print(f"✅ Dataset Built!")
    print(f"X_train shape: {X_train.shape} -> (Samples, {X_train.shape[1]} Years, {X_train.shape[2]} Features)")
    print(f"Y_train shape: {Y_train.shape} -> (Samples, {Y_train.shape[1]} Years, {Y_train.shape[2]} Features)")
    
    # Save to disk for Keras
    np.save("X_train.npy", X_train)
    np.save("Y_train.npy", Y_train)