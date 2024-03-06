import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating random data for house size and price for demonstration
np.random.seed(0)
house_size = np.random.randint(1000, 5000, 100)  # Random house sizes
house_age = np.random.randint(1, 50, 100)  # Random house ages
house_price = 50 * house_size + 100 * house_age + np.random.normal(0, 10000, 100)  # Generating house prices with some noise

# Reshaping the data
X = np.column_stack((house_size, house_age))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, house_price, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Read input data from Excel file
input_data = pd.read_excel('input_data.xlsx')

# Use the model to make predictions on the input data
predictions = model.predict(input_data[['HouseSize', 'HouseAge']])

# Create a DataFrame to store predictions
output_df = pd.DataFrame({'HouseSize': input_data['HouseSize'], 'HouseAge': input_data['HouseAge'], 'PredictedPrice': predictions})

# Write predictions to Excel file
output_df.to_excel('predictions.xlsx', index=False)

# Making predictions on the testing data
predictions = model.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Plotting the results
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], predictions, color='red', label='Predicted')
plt.title('House Price Prediction')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()
