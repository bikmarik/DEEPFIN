#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

class DataCalculator {
public:
    /**
     * FULL ALTMAN Z-SCORE
     * Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
     */
    double calculate_full_z_score(const std::map<std::string, double>& data) {
        try {
            double total_assets = data.at("Assets");
            if (total_assets <= 0) return 0.0;

            // X1: Working Capital / Total Assets (Liquidity)
            double x1 = (data.at("AssetsCurrent") - data.at("LiabilitiesCurrent")) / total_assets;
            
            // X2: Retained Earnings / Total Assets (Cumulative Profitability)
            double x2 = data.at("RetainedEarnings") / total_assets;
            
            // X3: EBIT / Total Assets (Operating Efficiency)
            double x3 = data.at("OperatingIncome") / total_assets;
            
            // X4: Market Value of Equity / Total Liabilities (Solvency/leverage)
            double x4 = data.at("MarketCap") / data.at("Liabilities");
            
            // X5: Sales / Total Assets (Asset Turnover)
            double x5 = data.at("Revenue") / total_assets;

            return (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5);
        } catch (...) {
            return -1.0; // Error indicator for missing tags
        }
    }
    /**
     * PFAI PREDICTIVE DRIVERS
     * Calculating the "Drift" and "Volatility" of Revenue and Net Income
     */
    std::map<std::string, double> calculate_pfai_drivers(double current_rev, double prev_rev, double net_income) {
        std::map<std::string, double> drivers;
        // Revenue Velocity (The 'Drift')
        drivers["rev_velocity"] = (prev_rev != 0) ? (current_rev - prev_rev) / std::abs(prev_rev) : 0.0;
        // Operating Margin (Efficiency Driver)
        drivers["op_margin"] = (current_rev > 0) ? net_income / current_rev : 0.0;
        
        return drivers;
    }

    double calculate_solvency_ratio(const std::map<std::string, double>& data) {
    try {
        double total_liabilities = data.at("Liabilities");
        if (total_liabilities <= 0) return 0.0;
        
        // Cash Flow from Ops / Total Liabilities (Standard Solvency Metric)
        return data.at("CashFlowOps") / total_liabilities;
    } catch (...) {
        return 0.0;
    }
    }

    std::vector<double> get_tensor(const std::map<std::string, double>& data) {
        std::vector<double> tensor;
        
        // --- GROUP A: THE KNOBS (Raw Dollars for Simulation) ---
        tensor.push_back(data.at("Revenue"));        // [0] Top line
        tensor.push_back(data.at("COGS"));           // [1] Production Costs (What-If knob 1)
        tensor.push_back(data.at("SGA"));            // [2] Salaries/Admin (What-If knob 2)
        tensor.push_back(data.at("RD"));             // [3] Innovation
        tensor.push_back(data.at("CAPEX"));          // [4] Investment (What-If knob 3)
        tensor.push_back(data.at("Inventory"));      // [5] Capacity
        
        // --- GROUP B: THE HEALTH (Ratios for Logic) ---
        double z = calculate_full_z_score(data);
        double solvency = calculate_solvency_ratio(data);
        tensor.push_back(z);                         // [6] Risk Floor
        tensor.push_back(solvency);                  // [7] Cash Safety
        
        // --- GROUP C: THE MOMENTUM (PFAI Drivers) ---
        double prev_rev = data.at("PrevRevenue");
        double velocity = (prev_rev != 0) ? (data.at("Revenue") - prev_rev) / std::abs(prev_rev) : 0.0;
        double margin = (data.at("Revenue") > 0) ? data.at("NetIncome") / data.at("Revenue") : 0.0;
        tensor.push_back(velocity);                  // [8] Speed of growth
        tensor.push_back(margin);                    // [9] Profit efficiency
        
        // --- GROUP D: SCALE ---
        tensor.push_back(data.at("MarketCap"));      // [10] Relative size
        tensor.push_back(data.at("Assets"));         // [11] Total footprint

        return tensor; 
    }
        
};

PYBIND11_MODULE(calculate_data, m) {
    py::class_<DataCalculator>(m, "DataCalculator")
        .def(py::init<>())
        .def("calculate_z_score", &DataCalculator::calculate_full_z_score)
        .def("calculate_solvency_ratio", &DataCalculator::calculate_solvency_ratio)
        .def("get_tensor", &DataCalculator::get_tensor);
}