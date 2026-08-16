import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = r'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

y = melbourne_data.Price

features = ["Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt",
            "Lattitude", "Longtitude"]

X = melbourne_data[features]

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)
# This function builds a decision tree with X amount of leaves
# Trains it, tests it with validation data, then returns the MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    scores[max_leaf_nodes] = get_mae(
        max_leaf_nodes, train_X, val_X, train_y, val_y
    )

best_tree_size = min(scores,key=scores.get)

print(scores)
print("Best tree size:", best_tree_size)
print("Leaves - Mean Absolute Error")
for leaf, mae in scores.items():
    print(f"{leaf:<3} → {mae:,.2f}")
