
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Locate project files
project_folder = Path(__file__).resolve().parent

# Load the trained model
model = joblib.load(project_folder / "best_battery_model.pkl")

# Load feature names
with open(project_folder / "battery_features.json", "r") as file:
    features = json.load(file)

# Load dataset
data = pd.read_csv(
    project_folder / "battery_cycle_data_preprocessed.csv"
)

print("===== BATTERY HEALTH MODEL EVALUATION =====")
print("Total dataset rows:", len(data))

# Select battery B0006, held out during model training
test_data = data[data["battery"] == "B0006"].copy()

if test_data.empty:
    raise ValueError("Battery B0006 was not found in the dataset.")

# Prepare inputs and actual SOH
X_test = test_data[features]
y_actual = test_data["SOH_percent"]

# Predict SOH
y_predicted = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_actual, y_predicted)
rmse = mean_squared_error(y_actual, y_predicted) ** 0.5
r2 = r2_score(y_actual, y_predicted)

# Compare predictions clipped to the valid 0–100% range
y_predicted_clipped = pd.Series(y_predicted).clip(0, 100)

mae_clipped = mean_absolute_error(y_actual, y_predicted_clipped)
rmse_clipped = mean_squared_error(y_actual, y_predicted_clipped) ** 0.5
r2_clipped = r2_score(y_actual, y_predicted_clipped)

print("\n===== CLIPPED PREDICTION METRICS (0–100%) =====")
print(f"MAE:  {mae_clipped:.4f} percentage points")
print(f"RMSE: {rmse_clipped:.4f} percentage points")
print(f"R²:   {r2_clipped:.4f}")
print("\nTest battery: B0006")
print("Number of test readings:", len(test_data))
print(f"MAE:  {mae:.4f} percentage points")
print(f"RMSE: {rmse:.4f} percentage points")
print(f"R²:   {r2:.4f}")

# Create a comparison table
results = pd.DataFrame({
    "Battery": test_data["battery"].values,
    "Cycle": test_data["cycle"].values,
    "Actual_SOH": y_actual.values,
    "Predicted_SOH": y_predicted,
    "Absolute_Error": abs(y_actual.values - y_predicted)
})

print("\n===== FIRST 10 PREDICTIONS =====")
print(results.head(10).round(2).to_string(index=False))

# Save all predictions for the project report
output_file = project_folder / "evaluation_results.csv"
results.to_csv(output_file, index=False)

print("\nFull results saved to:", output_file)
print("\nEvaluation completed successfully.")

import matplotlib.pyplot as plt

# Sort readings by battery cycle
results_sorted = results.sort_values("Cycle")

# Create actual vs predicted SOH graph
plt.figure(figsize=(10, 5))

plt.plot(
    results_sorted["Cycle"],
    results_sorted["Actual_SOH"],
    label="Actual SOH",
    linewidth=2
)

plt.plot(
    results_sorted["Cycle"],
    results_sorted["Predicted_SOH"],
    label="Predicted SOH",
    linewidth=2
)

plt.xlabel("Battery Cycle Number")
plt.ylabel("State of Health (SOH %)")
plt.title("Actual vs Predicted Battery Health - B0006")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save graph in the project folder
graph_file = project_folder / "actual_vs_predicted_soh.png"
plt.savefig(graph_file, dpi=300)

print("Graph saved to:", graph_file)

# Display the graph
plt.show()