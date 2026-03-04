import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from dataGenius.process_data import FinancialProcessor
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE MODEL AND SCALER
# ==========================================
# Notice the updated paths pointing to your new folders!
model = keras.models.load_model("weights/deepfin_v2.keras")
scaler_mean = np.load("data/scaler_mean.npy")
scaler_scale = np.load("data/scaler_scale.npy")

# ==========================================
# 2. GATHER HISTORICAL (INPUT) AND ACTUAL FUTURE DATA
# ==========================================
history = []
actual_future = []

print("🔍 Extracting Apple's financial states via C++ Engine...")

# T_in: 3 years of input sequence
for year in [2019, 2020, 2021]:
    res = FinancialProcessor("AAPL", year).process_ticker()
    history.append(res['tensor'])

# T_out: 3 years of actual realized data to compare against predictions
for year in [2022, 2023, 2024]:
    res = FinancialProcessor("AAPL", year).process_ticker()
    actual_future.append(res['tensor'])

input_seq = np.array(history).reshape(1, 3, 12)
actual_seq = np.array(actual_future).reshape(1, 3, 12)

# ==========================================
# 3. BASELINE AI PREDICTION (JAX JIT-COMPILED)
# ==========================================
input_scaled = (input_seq - scaler_mean) / scaler_scale
prediction_raw = model.predict(input_scaled)
prediction = (prediction_raw * scaler_scale) + scaler_mean

# ==========================================
# 4. QUANTIVESTA™ $5 BILLION CAUSAL PERTURBATION
# ==========================================
# Inject +$5,000,000,000 to CAPEX (Index 4) in the final input year 2021 (Index 2)
sim_input = input_scaled.copy()
sim_input[0, 2, 4] += (5000000000 / scaler_scale[4]) 

sim_prediction_raw = model.predict(sim_input)
sim_prediction = (sim_prediction_raw * scaler_scale) + scaler_mean

# ==========================================
# 5. GENERATE THE COMPARATIVE TIME-SERIES GRAPH
# ==========================================

# Extract Revenue (Index 0) and convert to Billions
hist_rev = input_seq[0, :, 0] / 1e9
actual_rev = actual_seq[0, :, 0] / 1e9
base_pred_rev = prediction[0, :, 0] / 1e9
sim_pred_rev = sim_prediction[0, :, 0] / 1e9

historical_years = [2019, 2020, 2021]
forecast_years = [2021, 2022, 2023, 2024]

# Connect the forecast and actual lines back to the final historical point (2021)
last_hist_val = hist_rev[-1]
actual_line = [last_hist_val, actual_rev[0], actual_rev[1], actual_rev[2]]
baseline_line = [last_hist_val, base_pred_rev[0], base_pred_rev[1], base_pred_rev[2]]
simulated_line = [last_hist_val, sim_pred_rev[0], sim_pred_rev[1], sim_pred_rev[2]]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 4 lines
ax.plot(historical_years, hist_rev, marker='o', color='#4a4a4a', 
        linewidth=2.5, markersize=8, label='Historical Revenue (Input)')

ax.plot(forecast_years, actual_line, marker='D', color='#d62728', 
        linewidth=2.5, markersize=8, label='Actual Realized Revenue')

ax.plot(forecast_years, baseline_line, marker='s', color='#1f77b4', 
        linewidth=2.5, linestyle='--', markersize=8, label='Baseline AI Forecast')

ax.plot(forecast_years, simulated_line, marker='^', color='#2ca02c', 
        linewidth=2.5, linestyle='--', markersize=8, label='Simulated (+$5B CAPEX)')

# Inference Point Divider
ax.axvline(x=2021, color='black', linestyle=':', linewidth=2, label='Inference Point (T_0 = 2021)')

# Formatting
ax.set_title('QUANTIVESTA™: AI Forecast vs Actuals vs Causal Perturbation', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Fiscal Year', fontsize=12)
ax.set_ylabel('Revenue (Billions $)', fontsize=12)
ax.set_xticks(historical_years + forecast_years[1:]) 
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)

# Save and Show (Saving directly to the assets folder)
plt.tight_layout()
plt.savefig('assets/actual_vs_predicted_simulation.png', dpi=300)
print("📈 Graph successfully saved as 'assets/actual_vs_predicted_simulation.png'")
plt.show()