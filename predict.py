import numpy as np
import keras
from dataGenius.process_data import FinancialProcessor

# 1. LOAD THE MODEL AND SCALER
model = keras.models.load_model("deepfin_v1.keras")
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")


# 2. THE INPUT: APPLE 2022, 2023, 2024
history = []
print("🔍 Extracting Apple's 3-year historical state via C++ Engine...")
for year in [2022, 2023, 2024]:
    processor = FinancialProcessor("AAPL", year)
    # This must be the 12-feature Unified Tensor from your C++ code
    result = processor.process_ticker()
    history.append(result['tensor'])
input_seq = np.array(history).reshape(1, 3, 12)

# 3. SCALE AND FORECAST
input_scaled = (input_seq - scaler_mean) / scaler_scale
prediction_raw = model.predict(input_scaled)

# 4. REVERSE SCALE THE PREDICTION
prediction = (prediction_raw * scaler_scale) + scaler_mean

# 5. PRINT THE 2026 RESULTS (Index 1 of the 3-year forecast)
feature_names = [
    "Revenue", "COGS", "SGA", "RD", "CAPEX", "Inventory", 
    "Z-Score", "Solvency", "Rev Velocity", "Op Margin", 
    "MarketCap", "Total Assets"
]

print("\n" + "="*40)
print("🚀 APPLE 2026 AI PROJECTION")
print("="*40)

aapl_2026 = prediction[0, 1] 

for i, name in enumerate(feature_names):
    val = aapl_2026[i]
    if i in [0, 1, 2, 3, 4, 5, 10, 11]:
        print(f"{name:15}: ${val / 1e9:>7.2f} B")
    else:
        print(f"{name:15}: {val:>10.4f}")

print("="*40)
# 1. Get the 'Normal' 2026 Revenue
base_revenue = aapl_2026[0]
sim_input = input_scaled.copy()
sim_input[0, 2, 4] += (200000 / scaler_scale[4]) 

# 2. Predict the 'Simulated' 2026
sim_prediction_raw = model.predict(sim_input)
sim_prediction = (sim_prediction_raw * scaler_scale) + scaler_mean
sim_revenue = sim_prediction[0, 1, 0]

# 3. Result
impact = sim_revenue - base_revenue
print(f"💰 Result: $200,000 extra CAPEX resulted in ${impact:,.2f} change in 2026 Revenue.")