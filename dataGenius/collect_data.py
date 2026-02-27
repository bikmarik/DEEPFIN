import edgar
import pandas as pd
import os

# Set identity - Required by SEC
edgar.set_identity("Marat Bikbaev bikbaevmarik55555@gmail.com")

class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = f"data/{ticker}"
        os.makedirs(self.data_path, exist_ok=True)
        
        #import 
        self.company = edgar.Company(ticker)
        self.financials = self.company.get_financials()
        self.tenk = self.company.latest_tenk

    def save_data(self):
        if not self.financials:
            print(f"  [-] No financials found for {self.ticker}")
            return False

        try:
            # SAVE BALANCE SHEET (QUANTIVESTA Constraints)
            # FIX: Call the method balance_sheet() first!
            bs_obj = self.financials.balance_sheet()
            if bs_obj:
                bs_obj.to_dataframe().to_csv(f"{self.data_path}/balance_sheet.csv")
                print(f"  [+] Saved Balance Sheet")

            # SAVE INCOME STATEMENT (PFAI Performance Drivers)
            # FIX: Call income_statement() as a function
            is_obj = self.financials.income_statement()
            if is_obj:
                is_obj.to_dataframe().to_csv(f"{self.data_path}/income_statement.csv")
                print(f"  [+] Saved Income Statement")

            # SAVE CASH FLOW (The Reality Check)
            # FIX: Call cashflow_statement() as a function
            cf_obj = self.financials.cashflow_statement()
            if cf_obj:
                cf_obj.to_dataframe().to_csv(f"{self.data_path}/cashflow.csv")
                print(f"  [+] Saved Cash Flow")

            # SAVE ITEM 7 (MD&A Text for LLM processing)
            if self.tenk:
                # Use management_discussion for direct access in 2026
                mda = self.tenk.management_discussion
                if mda:
                    with open(f"{self.data_path}/mda.txt", "w", encoding="utf-8") as f:
                        f.write(str(mda))
                    print(f"  [+] Saved MD&A Text")
            
            return True
        except Exception as e:
            print(f"  [-] Error saving {self.ticker}: {e}")
            return False
        
    def transform_data(self):
        # Placeholder for future data transformation logic

        pass






def main():
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for ticker in companies:
        print(f"🚀 Collecting data for {ticker}...")
        collector = DataCollector(ticker)
        if collector.save_data():
            print(f"✅ {ticker} complete.")
        else:
            print(f"❌ {ticker} failed.")

if __name__ == "__main__":
    main()