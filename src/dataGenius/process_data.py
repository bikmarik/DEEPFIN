import pandas as pd
import calculate_data
import yfinance as yf

class FinancialProcessor:
    def __init__(self, ticker, year):
        self.engine = calculate_data.DataCalculator()
        self.ticker = ticker
        self.year = year
        self.path = f"data/{ticker}/{year}"

    def get_value(self, df, concept_name):
        """
        Finds the primary value for a GAAP concept.
        Ensures we don't get 'dimension' breakdowns (like segments or levels).
        """
        try:
            # Filter for the concept and where dimension is False (the 'Total' line)
            row = df[(df['concept'] == concept_name) & (df['dimension'] == False)]
            if row.empty:
                # Fallback: just try to find the concept if dimension column is missing
                row = df[df['concept'] == concept_name]
            
            if not row.empty:
                # Columns 4 and 5 are typically the most recent and previous dates
                # Using .iloc[0] gets the value from the first matching row
                current_val = float(row.iloc[0, 4]) 
                prev_val = float(row.iloc[0, 5]) if len(row.columns) > 5 else 0.0
                return current_val, prev_val
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    def process_ticker(self):
        
        try:
            bs_df = pd.read_csv(f"{self.path}/balance_sheet.csv")
            is_df = pd.read_csv(f"{self.path}/income_statement.csv")
            cf_df = pd.read_csv(f"{self.path}/cashflow.csv")


            # 1. BALANCE SHEET (Robust Extraction)

            assets, _ = self.get_value(bs_df, "us-gaap_Assets")
            assets_curr, _ = self.get_value(bs_df, "us-gaap_AssetsCurrent")
            liab_curr, _ = self.get_value(bs_df, "us-gaap_LiabilitiesCurrent")
            retained, _ = self.get_value(bs_df, "us-gaap_RetainedEarningsAccumulatedDeficit")
            inv, _ = self.get_value(bs_df, "us-gaap_InventoryNet")
            # Robust Liabilities: Fallback if us-gaap_Liabilities is missing
            liab_total, _ = self.get_value(bs_df, "us-gaap_Liabilities")
            if liab_total == 0:
                equity, _ = self.get_value(bs_df, "us-gaap_StockholdersEquity")
                liab_total = assets - equity


            # 2. INCOME STATEMENT (Robust Revenue)
            rev_curr, rev_prev = self.get_value(is_df, "us-gaap_Revenues")
            if rev_curr == 0:
                rev_curr, rev_prev = self.get_value(is_df, "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax")
            op_inc, _ = self.get_value(is_df, "us-gaap_OperatingIncomeLoss")
            net_inc, _ = self.get_value(is_df, "us-gaap_NetIncomeLoss")
            cogs, _ = self.get_value(is_df, "us-gaap_CostOfGoodsAndServicesSold")
            if cogs == 0:
                cogs, _ = self.get_value(is_df, "us-gaap_CostOfRevenue")            
            sga, _ = self.get_value(is_df, "us-gaap_SellingGeneralAndAdministrativeExpense")
            sga, _ = self.get_value(is_df, "us-gaap_SellingGeneralAndAdministrativeExpense")
            if sga == 0:
                marketing, _ = self.get_value(is_df, "us-gaap_SellingAndMarketingExpense")
                admin, _ = self.get_value(is_df, "us-gaap_GeneralAndAdministrativeExpense")
                sga = marketing + admin
            rd, _ = self.get_value(is_df, "us-gaap_ResearchAndDevelopmentExpense")

            
            # 3. CASH FLOW
            ocf, _ = self.get_value(cf_df, "us-gaap_NetCashProvidedByUsedInOperatingActivities")
            capex, _ = self.get_value(cf_df, "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment")
            if capex == 0:
                capex, _ = self.get_value(cf_df, "us-gaap_PaymentsToAcquireProductiveAssets")
            capex = abs(capex)


            # 4. PREPARE MAP FOR C++
            raw_data = {
                "Revenue": rev_curr,
                "COGS": cogs,
                "SGA": sga,
                "RD": rd,
                "Inventory": inv,
                "CAPEX": capex,
                "Assets": assets,
                "AssetsCurrent": assets_curr,
                "LiabilitiesCurrent": liab_curr,
                "RetainedEarnings": retained,
                "OperatingIncome": op_inc,
                "Revenue": rev_curr,
                "PrevRevenue": rev_prev,
                "NetIncome": net_inc,
                "Liabilities": liab_total,
                "CashFlowOps": ocf,
                "MarketCap": yf.Ticker(self.ticker).info.get("marketCap", 2000000000000.0) 
            }
            z_score = self.engine.calculate_z_score(raw_data)
            tensor = self.engine.get_tensor(raw_data)
            return {
                "ticker": self.ticker,
                "z_score": z_score,
                "tensor": tensor,
                "raw_data": raw_data
            }

        except Exception as e:
            print(f"Error processing {self.ticker}: {e}")
            return None