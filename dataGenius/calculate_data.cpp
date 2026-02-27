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

    // Unified Feature Extractor for Keras
    std::vector<double> get_feature_tensor(std::map<std::string, double> raw_data) {
        std::vector<double> tensor;
        double z = calculate_full_z_score(raw_data);
        auto pfai = calculate_pfai_drivers(raw_data["Revenue"], raw_data["PrevRevenue"], raw_data["NetIncome"]);
        double solvency = calculate_solvency_ratio(raw_data);
        tensor.push_back(z);                      // [0] Risk Constraint
        tensor.push_back(pfai["rev_velocity"]);   // [1] Growth Driver
        tensor.push_back(pfai["op_margin"]);      // [2] Efficiency Driver
        tensor.push_back(solvency);               // [3] Solvency Driver
        tensor.push_back(raw_data["MarketCap"]);  // [4] Scale Factor
        return tensor;
    }
    // Add these to your DataCalculator class in C++
    std::vector<double> get_simulation_tensor(const std::map<std::string, double>& data) {
        std::vector<double> tensor;
        tensor.push_back(data.at("Revenue"));        // [0]
        tensor.push_back(data.at("COGS"));           // [1] Production Costs
        tensor.push_back(data.at("SGA"));            // [2] Salaries & Admin
        tensor.push_back(data.at("RD"));             // [3] Innovation/R&D
        tensor.push_back(data.at("Inventory"));      // [4] Production capacity indicator
        tensor.push_back(data.at("CAPEX"));          // [5] The "Investment" input
        
        // Efficiency Ratios (Helps the AI learn the relationship between cost and profit)
        double prod_efficiency = (data.at("Revenue") > 0) ? data.at("COGS") / data.at("Revenue") : 0.0;
        tensor.push_back(prod_efficiency);           // [6]
        
        // Traditional Health (Z-Score & Solvency)
        tensor.push_back(calculate_full_z_score(data)); // [7]
        
        return tensor;
    }
    
};

PYBIND11_MODULE(calculate_data, m) {
    py::class_<DataCalculator>(m, "DataCalculator")
        .def(py::init<>())
        .def("calculate_full_z_score", &DataCalculator::calculate_full_z_score)
        .def("calculate_solvency_ratio", &DataCalculator::calculate_solvency_ratio)
        .def("get_feature_tensor", &DataCalculator::get_feature_tensor)
        .def("get_simulation_tensor", &DataCalculator::get_simulation_tensor);
}