# DEEPFIN & QUANTIVESTA™ 📈
**A Prescriptive Financial Intelligence and Causal Simulation Engine**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![C++20](https://img.shields.io/badge/C%2B%2B-20-purple)
![Keras 3](https://img.shields.io/badge/Keras-3.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
Traditional Financial Planning and Analysis (FP&A) models are predominantly descriptive, relying on linear historical extrapolations. **DEEPFIN** is a hybrid computational framework designed to transition enterprise planning from retrospective variance analysis to proactive, prescriptive decision intelligence. 

By integrating a high-performance **C++ extraction engine** with a **Keras 3 Sequence-to-Sequence (Seq2Seq) LSTM network**, DEEPFIN processes high-dimensional SEC XBRL regulatory filings into actionable forecasts. Its proprietary prescriptive logic layer, **QUANTIVESTA™**, allows executive management to perform causal perturbation simulations (e.g., injecting +$200,000 into CAPEX) to quantitatively measure the long-term Marginal Revenue Return on Investment (MR-ROI).

## Key Features
* **High-Performance XBRL Ingestion:** A `C++20` processing pipeline (`DataCalculator`) linked via `Pybind11` for low-latency parsing of highly sparse SEC EDGAR XML data.
* **Structural Health Bounding:** Mathematically integrates bankruptcy risk constraints (Altman Z-Score) and liquidity thresholds (Solvency Ratios) directly into the feature space.
* **The Unified Master Tensor:** Condenses raw operational knobs and financial scale factors into a standardized 12-dimensional tensor.
* **Deep Temporal Forecasting:** Utilizes a Seq2Seq LSTM architecture optimized with AdamW to map 3-year historical states to 3-year multi-variable forecasts.
* **QUANTIVESTA™ Causal Simulation:** A prescriptive "What-If" engine that isolates the downstream effects of specific budget reallocations across a neural latent space.

---

## ️ System Architecture

1. **Data Ingestion:** Automated querying of the SEC EDGAR API for 10-K filings (focusing on S&P 500 tech constituents: AAPL, MSFT, META, etc.).
2. **C++ Harmonization:** Fallback-mapping logic resolves taxonomy inconsistencies (e.g., `CostOfGoodsSold` vs. `CostOfRevenue`).
3. **Sliding Window:** 9-year historical horizons are transformed into 3-year overlapping sequential windows, creating tensors of shape `(Samples, 3, 12)`.
4. **Seq2Seq Inference:** The Encoder condenses historical financial momentum; the Decoder projects future financial states.

---

## ⚙️ Installation & Prerequisites

To build and run this project, you need a C++20 compiler and Python 3.10+.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/deepfin.git](https://github.com/yourusername/deepfin.git)
cd deepfin
