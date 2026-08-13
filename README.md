# 📦 Demand Forecasting with XGBoost & Streamlit

A machine learning pipeline and web application designed to forecast retail product demand, reducing inventory costs associated with **stockouts** and **overstocking**. Built using **XGBoost**, tuned with **L1/L2 regularization**, and deployed as an interactive **Streamlit** dashboard.

---

## 📊 Executive Summary

- **Problem:** Retailers lose margin on stockouts (lost revenue) and overstocking (capital tied up in excess inventory). Traditional rule-based heuristics fail to capture non-linear interactions across promotions, pricing, and category dynamics.
- **Dataset:** 76,000 clean transaction records covering 5 core product categories (*Clothing, Electronics, Furniture, Groceries, Toys*).
- **Core Results:**
  - **27.2% reduction in MAE** compared to the naive baseline.
  - Reduced average prediction error from **37.04 units to 26.97 units**.
  - Identified **Promotional Activity (58%)** and **Product Category (27%)** as the primary drivers of demand.

---

## 🎯 Model Comparison Results

Multiple algorithms were trained and evaluated on an 80/20 train-test split using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ score:

| Model | Evaluation Purpose | RMSE (Units) | MAE (Units) | $R^2$ Score | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline (Mean)** | Naive benchmark | 46.99 | 37.04 | 0.000 | Predicts overall average demand; ignores feature dynamics. |
| **Ridge Regression** | Linear modeling baseline | 40.12 | 31.85 | 0.271 | Fast execution, but underfits non-linear promo/category splits. |
| **Default XGBoost** | Basic tree ensemble | 38.45 | 29.50 | 0.330 | Captures interaction terms, but sensitive to noisy outliers. |
| **Tuned XGBoost (L1 & L2)** | **Final Selected Model** | **35.56** | **26.97** | **0.428** | Best performance; prevents overfitting via $\alpha=0.01$ and $\lambda=20.0$. |

---

## 🧠 Regularization Strategy

Hyperparameters were tuned using `RandomizedSearchCV` with 3-fold cross-validation. Regularization parameters were explicitly optimized to control tree growth and feature weights:

- **L1 Regularization (`reg_alpha = 0.01`)**: Performs subtle feature selection by shrinking uninformative feature weights.
- **L2 Regularization (`reg_lambda = 20.0`)**: Strongly penalizes extreme weight variance, preventing model over-sensitivity to individual inputs (e.g., pricing spikes) in live inference.

---

## 💡 Feature Importance Highlights

Analysis of the final XGBoost model reveals key drivers behind demand volume:

1. **Promotion (58%)** — Active marketing campaigns are the single strongest determinant of unit volume.
2. **Category (27%)** — Structural baseline demand differs drastically across product types.
3. **Price (8%)** — Elasticity plays a secondary role relative to promotion placement.
4. **Competitor Pricing (3%)**, **Discount (2%)**, **Inventory Level (2%)** — Supplementary contextual signals.

---

## 📁 Repository Structure

```text
.
├── demand_forecasting.csv       # Dataset (76,000 transaction records)
├── train_model.py              # Training script & hyperparameter tuning
├── compare_models.py           # Model benchmarking & chart generator
├── app.py                      # Streamlit interactive web application
├── visualize_results.py        # Analytics dashboard plotting script
├── XG_Boost_demand_model.pickle # Saved XGBoost model artifact
├── label_encoders.pickle       # Saved LabelEncoder mappings
├── model_comparison_chart.png  # Exported metric comparison visualization
├── demand_forecasting_dashboard.png # Multi-panel analytics dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
