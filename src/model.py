import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "data" / "housing_price_dataset.csv"
df = pd.read_csv(csv_path)

print(df.head())
print(df.columns)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
csv_path = Path(__file__).resolve().parent.parent / "data" / "housing_price_dataset.csv"
df = pd.read_csv(csv_path)

# Convert categorical column to numeric (Neighborhood)
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

# Define features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model Trained Successfully ✅")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
