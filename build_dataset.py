import numpy as np
import edgar
from scripts.collect_data import DataCollector
from src.dfmaker3000 import DataFrameBuilder3000

if __name__ == "__main__":
    start_year = 2017
    end_year = 2025
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] 
    
    print("Input Name (for SEC identity): ", end="")
    Name = input()
    print("Input Email (for SEC identity): ", end="")
    Email = input()
    
    print("Starting data collection and dataset building...")
    edgar.set_identity(Name + " " + Email)  
    
    # 1. Collect Data using the script
    for ticker in tickers:
        print(f"Processing {ticker} from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            collector = DataCollector(ticker, year)
            if collector.save_data():
                print(f"  [+] Data collected for {ticker} {year}")
            else:
                print(f"  [-] Failed to collect data for {ticker} {year}")
    
    # 2. Build the Dataset using the src class
    builder = DataFrameBuilder3000(tickers, start_year, end_year)
    X_train, Y_train = builder.build_seq2seq_dataset(window_size=3)
    
    print(f"✅ Dataset Built!")
    print(f"X_train shape: {X_train.shape} -> (Samples, {X_train.shape[1]} Years, {X_train.shape[2]} Features)")
    print(f"Y_train shape: {Y_train.shape} -> (Samples, {Y_train.shape[1]} Years, {Y_train.shape[2]} Features)")
    
    # 3. Save to the new data/ directory
    np.save("data/X_train.npy", X_train)
    np.save("data/Y_train.npy", Y_train)