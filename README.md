# 📊 Insurance Claims Risk Analyzer

An interactive **Streamlit web app** and **Python script** that analyzes insurance claims data, calculates actuarial KPIs, visualizes results, and predicts fraud risk using machine learning.

---

## 🚀 Features
- **Data Cleaning & Preparation** – handles raw insurance claims data.  
- **Key Insurance KPIs**  
  - Claim Frequency (per region)  
  - Severity (average claim amount)  
  - Loss Ratio (claims ÷ premium)  
- **Visualizations**  
  - Claim frequency by region (bar chart)  
  - Fraud vs Non-Fraud distribution (pie chart)  
  - Loss ratio distribution (histogram)  
- **Fraud Prediction Model**  
  - Logistic Regression to predict fraudulent claims  
  - Displays model accuracy  
  - Adds fraud predictions to dataset  
- **Automated Reports**  
  - Exports results to Excel (`Insurance_Claims_Report.xlsx`)  
  - Saves model metrics to `model_metrics.txt`

---

## 📂 Project Structure
```
Insurance_Claims_Risk_Analyzer/
│
├── app.py                     # Streamlit web app
├── claims_analyzer.py          # Standalone Python script
├── claims_dataset.csv          # Sample dataset
├── Insurance_Claims_Report.xlsx # Sample output report
├── chart_claims_by_region.png   # Sample chart
├── chart_fraud_pie.png
├── chart_loss_ratio_hist.png
├── model_metrics.txt           # Model performance
└── README.md                   # Project documentation
```

---

## 🛠️ Installation
1. Clone the repository or unzip the project folder.
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
   ```

---

## ▶️ Usage

### Option 1: Run as Web App
```bash
streamlit run app.py
```
- Opens in your browser at `http://localhost:8501`  
- Upload `claims_dataset.csv`  
- Explore KPIs, charts, and fraud predictions  

### Option 2: Run as Python Script
```bash
python claims_analyzer.py
```
- Generates **Excel report** and **charts** automatically  

---

## 📊 Example Output
- **KPIs**: Total Claims, Avg Premium, Avg Claim, Loss Ratio  
- **Charts**: Claim frequency, Fraud distribution, Loss ratio histogram  
- **Fraud Prediction**: Logistic Regression, accuracy score ≈ 0.90+  

---

## 🎯 Why This Project?
This project simulates real-world actuarial consulting work:
- Combines **technical coding skills** (Python, ML, Streamlit)  
- With **business understanding** (insurance KPIs, fraud detection)  
- Making it perfect for roles like **Insurance Consulting Analyst Engineer at WTW**  

---

## 👤 Author
**Harsh [Your Last Name]**  
M.Tech CSE @ Thapar Institute of Engineering & Technology (2024–26)  
