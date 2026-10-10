
import joblib
import json
import pandas as pd

# Load our trained AI model
model = joblib.load("best_battery_model.pkl")

# Load the features used during training
with open("battery_features.json", "r") as file:
    features = json.load(file)


def predict_battery_health():
    print("\n===== AI BATTERY HEALTH PREDICTION =====")

    # Get battery readings from the user
    cycle = float(input("Enter battery cycle number: "))
    voltage_mean = float(input("Enter mean voltage (V): "))
    voltage_min = float(input("Enter minimum voltage (V): "))
    voltage_max = float(input("Enter maximum voltage (V): "))
    current_mean = float(input("Enter mean current (A): "))
    temperature_mean = float(input("Enter mean temperature (C): "))
    temperature_max = float(input("Enter maximum temperature (C): "))

    # Arrange the inputs in the same order as training
    input_data = pd.DataFrame([{
        "cycle": cycle,
        "voltage_mean_V": voltage_mean,
        "voltage_min_V": voltage_min,
        "voltage_max_V": voltage_max,
        "current_mean_A": current_mean,
        "temperature_mean_C": temperature_mean,
        "temperature_max_C": temperature_max
    }])[features]

    # Predict battery State of Health
    predicted_soh = float(model.predict(input_data)[0])
    predicted_soh = max(0, min(100, predicted_soh))

    print("\n===== PREDICTION RESULT =====")
    print(f"Predicted Battery SOH: {predicted_soh:.2f}%")

    if predicted_soh >= 80:
        print("Estimated condition: Good")
    elif predicted_soh >= 60:
        print("Estimated condition: Moderate")
    else:
        print("Estimated condition: Poor")

    print("\nNote: This is an experimental ML estimate,")
    print("not a certified battery diagnostic.")


if __name__ == "__main__":
    predict_battery_health()