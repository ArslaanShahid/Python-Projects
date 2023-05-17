import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

database_file = 'CarDatabase_dataAnalytics.db'

conn = sqlite3.connect(database_file)

# Load CarSharing table into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM CarSharing", conn)

# print(df)
# Drop duplicate rows
print("Drop Dublicating.....")
check_drop = df.drop_duplicates(inplace=True)
print(check_drop)
# Prepare the data
print(df.columns)

df_2017 = df[df['timestamp'].dt.year == 2017][['timestamp', 'temp']]

# Set the 'timestamp' column as the index
df_2017.set_index('timestamp', inplace=True)
print(df_2017)
# Resample the data on a weekly basis and calculate the mean temperature
df_weekly = df_2017.resample('W').mean()

# Extract the temperature column from the weekly data
weekly_temp = df_weekly['temp']
# train_data = df_weekly.iloc[:int(len(df_weekly) * 0.7)]
# test_data = df_weekly.iloc[int(len(df_weekly) * 0.7):]

# print(train_data['temp']) 


# # Build the ARIMA model
# model = ARIMA(train_data['temp_feel'], order=(1, 0, 0))
# print(model)

# model_fit = model.fit()

# # Validate the model
# predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

# # Plot the actual and predicted values
# plt.plot(test_data.index, test_data['temp'], label='Actual')
# plt.plot(predictions.index, predictions, label='Predicted')
# plt.xlabel('Date')
# plt.ylabel('Temperature')
# plt.title('ARIMA Model - Weekly Average Temperature')
# plt.legend()
# plt.show()

# # Make predictions
# forecast = model_fit.forecast(steps=52)  # Adjust the steps as per your desired time period

# # Print the forecasted temperatures
# print("Forecasted temperatures:")
# print(forecast)