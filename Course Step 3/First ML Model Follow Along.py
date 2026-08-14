import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = r'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price 
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
x = filtered_melbourne_data[melbourne_features]
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(x, y)
