
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load data
    df = pd.read_csv("claims_dataset.csv")

    # Basic cleaning
    df = df.dropna().copy()
    df["LossRatio"] = df["ClaimAmount"] / df["Premium"].replace(0, np.nan)
    df["LossRatio"] = df["LossRatio"].fillna(0)

    # ---- KPIs ----
    # Frequency by region (number of claims per region)
    freq_by_region = df["Region"].value_counts().rename("ClaimCount").to_frame()

    # Severity by region (average claim amount)
    sev_by_region = df.groupby("Region")["ClaimAmount"].mean().round(2).rename("AvgClaimAmount").to_frame()

    # Overall Loss Ratio
    overall_lr = float((df["ClaimAmount"].sum() / df["Premium"].sum()).round(4))

    # Print KPIs to console
    print("=== KPIs ===")
    print("Overall Loss Ratio:", overall_lr)
    print("\nClaim Frequency by Region:\n", freq_by_region)
    print("\nAverage Severity by Region:\n", sev_by_region)

    # ---- Charts (one figure per chart, no specific colors) ----
    # 1) Claim Frequency by Region (bar)
    plt.figure()
    freq_by_region.plot(kind="bar", legend=False, title="Claim Frequency by Region")
    plt.xlabel("Region")
    plt.ylabel("Number of Claims")
    plt.tight_layout()
    plt.savefig("chart_claims_by_region.png")
    plt.close()

    # 2) Fraud vs Non-Fraud (pie)
    plt.figure()
    df["IsFraud"].value_counts().sort_index().plot(kind="pie", autopct="%1.1f%%", labels=["Non-Fraud", "Fraud"], title="Fraud vs Non-Fraud")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("chart_fraud_pie.png")
    plt.close()

    # 3) Loss Ratio Distribution (histogram)
    plt.figure()
    df["LossRatio"].plot(kind="hist", bins=20, title="Loss Ratio Distribution")
    plt.xlabel("Loss Ratio")
    plt.tight_layout()
    plt.savefig("chart_loss_ratio_hist.png")
    plt.close()

    # ---- Modeling ----
    # Predict "IsFraud" without using ClaimAmount to avoid leakage
    model_df = df.copy()
    X = model_df[["CustomerAge", "VehicleAge", "PolicyTenure", "Premium", "Region", "VehicleType"]]
    X = pd.get_dummies(X, columns=["Region", "VehicleType"], drop_first=True)
    y = model_df["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)

    print("\n=== Model ===")
    print("Accuracy:", round(acc, 4))
    print(report)

    # ---- Save report files ----
    summary_rows = [
        ("Total Claims", int(len(df))),
        ("Average Premium", float(df["Premium"].mean())),
        ("Average Claim Amount", float(df["ClaimAmount"].mean())),
        ("Overall Loss Ratio", overall_lr),
        ("Model Accuracy", round(acc, 4)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Write Excel with multiple sheets
    with pd.ExcelWriter("Insurance_Claims_Report.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        freq_by_region.reset_index().rename(columns={"index":"Region"}).to_excel(writer, sheet_name="FrequencyByRegion", index=False)
        sev_by_region.reset_index().to_excel(writer, sheet_name="SeverityByRegion", index=False)
        df.to_excel(writer, sheet_name="RawData", index=False)

    # Write classification report to text
    with open("model_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Accuracy: " + str(round(acc, 4)) + "\n\n")
        f.write(report)

if __name__ == "__main__":
    main()
