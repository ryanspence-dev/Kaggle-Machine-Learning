import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Loading the Data
melbourne_file_path = r'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Drop rows with missing target values (in this case, any rows with missing Prices)
melbourne_data = melbourne_data.dropna(axis=0)

# Then select price as the Target value
y = melbourne_data.Price

# Select features to use for training
melbourne_features = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude"
]

X = melbourne_data[melbourne_features]

# Split the data into training and validation data
# Ensure repeatable randomness of split with random_state
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Construct the Random Forest Model
# Ensure repeatable randomness of constructured choices with random_state
rf_model = RandomForestRegressor(random_state=1)

# Train the model
rf_model.fit(train_X, train_y)

# Make predictions
rf_predictions = rf_model.predict(val_X)

# Calculate the Mean Absolute Error
rf_val_mae = mean_absolute_error(val_y, rf_predictions)

print(f"Validation MAE for Random Forest Model: ${rf_val_mae:.2f}")

# Random Forest combines predictions from many decision trees
# Each tree is built with some randomness, making the trees different
# The predictions from each tree are combined to produce the final prediction
# 
# Random Forests generally perform better than a single decision tree
# because they reduce the effect of poor individual predictions from trees