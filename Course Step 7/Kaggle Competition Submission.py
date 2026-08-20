import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = r'melb_data.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.Price

# Create X
features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

# Select columns corresponding to features
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

# Make validation predictions
rf_val_predictions = rf_model.predict(val_X)

# Calculate validation MAE
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# Train a model on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)


# Read the test data
test_data_path = r'melb_data.csv'
test_data = pd.read_csv(test_data_path)

# Select the same features
test_X = test_data[features]

# Make predictions
test_preds = rf_model_on_full_data.predict(test_X)


# Generate a submission file
output = pd.DataFrame({
    'Id': test_data.index,
    'Price': test_preds
})

# output.to_csv('submission.csv', index=False)
# print("submission.csv created successfully.")

# Initial submission to the House Prices Competition for Kaggle Learn Users.
# Competition: https://www.kaggle.com/competitions/home-data-for-ml-course
