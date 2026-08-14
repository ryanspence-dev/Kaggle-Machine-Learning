import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Load the data
melbourne_file_path = r'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)


# Remove rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)


# Select target and features
y = filtered_melbourne_data.Price

melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = filtered_melbourne_data[melbourne_features]


# Split the data into training and validation data
# Remove 'random_state=1' as an argument for a different random split of houses each run
# random_state=1 makes the training / validation split reproducible as it uses a deterministic algorithm to achieve pseudo randomness
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)


# Specify the model
# Remove 'random_state=1' as an argument for different results
# random_state=1 makes the decision tree choices reproducible (it is pseudo-randomness not true randomness)
melbourne_model = DecisionTreeRegressor(random_state=1)


# Train the model using the training data
melbourne_model.fit(train_X, train_y)


# Make predictions using the validation data
val_predictions = melbourne_model.predict(val_X)


# Print some predictions and actual prices
print("\nFirst 5 Validation Predictions:")
print("--------------------------------")
for predicted, actual in zip(val_predictions[:5], val_y[:5]):
    print(f"Predicted: ${predicted:,.0f} | Actual: ${actual:,.0f}")


# Calculate Mean Absolute Error
val_mae = mean_absolute_error(val_y, val_predictions)

print(f"\nValidation MAE: ${val_mae:,.0f}")