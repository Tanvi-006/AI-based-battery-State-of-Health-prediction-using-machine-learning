# AI-Based Battery State of Health (SOH) Prediction

## Project Overview
This project predicts the State of Health (SOH) of lithium-ion batteries using machine learning. Linear Regression is used to estimate battery health from battery cycle and operating measurements.

## Objectives
- Predict battery State of Health (SOH).
- Evaluate model performance using MAE, RMSE, and R².
- Compare machine-learning algorithms.
- Visualize actual and predicted SOH values.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Pickle and JSON

## Project Files
- `main.py` — Main prediction program.
- `evaluate_model.py` — Model evaluation.
- `best_battery_model.pkl` — Saved trained model.
- `battery_features.json` — Model feature information.
- `battery_cycle_data_preprocessed.csv` — Preprocessed dataset.
- `evaluation_results.csv` — Evaluation results.
- `actual_vs_predicted_soh.png` — Prediction performance graph.

## Dataset
The project uses preprocessed lithium-ion battery cycling data. The dataset source and original documentation should be acknowledged in the final report.

## Evaluation
Model performance is assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

## How to Run
1. Install the required Python libraries.
2. Ensure the model and feature files are in the project directory.
3. Run `main.py` to use the prediction program.

## Future Scope
- Test the model on more batteries and operating conditions.
- Improve prediction accuracy with additional features.
- Explore other machine-learning algorithms.
