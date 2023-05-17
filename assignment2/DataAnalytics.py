import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


label_encoder = LabelEncoder()


database_file = 'CarDatabase_dataAnalytics.db'

conn = sqlite3.connect(database_file)

# Load CarSharing table into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM CarSharing", conn)
# Drop duplicate rows
print("Drop Dublicating.....")
df.drop_duplicates(inplace=True)
# Drop rows with null values
df.dropna(inplace=True)

# # Create a DataFrame with the relevant columns
# print(df.dtypes)

df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce')


data1 = df[['temp','humidity','windspeed','demand']]

#droping null
data1 = data1.dropna()

# #T-Test
# t_statistic, p_value = stats.ttest_ind(data1['temp'],data1['demand'])
# print("T-Test")
# print("T-Statistic:", t_statistic)
# print("P-Value:", p_value)

# # Correlation Test
# correlation_coef, p_value = stats.pearsonr(data1['temp'], data1['demand'])
# print("Correlation Test")
# print("Correlation Coefficient:", correlation_coef)
# print("P-Value:", p_value)

df.dropna(inplace=True)
# Calculate the correlation coefficient and p-value for temp and demand
corr_temp, p_temp = pearsonr(df['temp'], df['demand'])
# Calculate correlation and p-value between humidity and demand
corr_humidity, p_humidity = pearsonr(df['humidity'], df['demand'])
# Calculate correlation and p-value between windspeed and demand
corr_windspeed, p_windspeed = pearsonr(df['windspeed'], df['demand'])

# Print correlation coefficients and p-values
print("Correlation and p-values:")
print("Temperature vs. Demand: Correlation =", corr_temp, "p-value =", p_temp)
print("Humidity vs. Demand: Correlation =", corr_humidity, "p-value =", p_humidity)
print("Windspeed vs. Demand: Correlation =", corr_windspeed, "p-value =", p_windspeed)

#T_Test
# Assuming workingday column is binary (0 or 1)
workingday = df[df['workingday'] == "Yes"]['demand']
non_workingday = df[df['workingday'] == "No"]['demand']

# Perform independent t-test
t_statistic, p_value = ttest_ind(workingday, non_workingday)
print(t_statistic)
# Print t-statistic and p-value
print("T-test (Workingday vs. Non-workingday):")
print("t-statistic =", t_statistic, "p-value =", p_value)

#ARIMA
# Convert the 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the data for 2017
df_2017 = df[df['timestamp'].dt.year == 2017]

# Resample the data to weekly frequency and calculate the average temperature
df_weekly = df_2017.resample('W', on='timestamp')['temp'].mean().reset_index()
df_weekly['temp'].interpolate(method='linear', inplace=True)
# Split the data into train and test sets (considering 30% for testing)
train_data = df_weekly.iloc[:int(len(df_weekly) * 0.7)]
test_data = df_weekly.iloc[int(len(df_weekly) * 0.7):]

# Create and fit the ARIMA model
model = ARIMA(train_data['temp'], order=(1, 0, 0))
model_fit = model.fit()

# Forecast the temperature for the test data
predictions = model_fit.forecast(steps=len(test_data))
# Convert the predicted values to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['predicted_temp'])

# Concatenate the actual and predicted temperatures
result_df = pd.concat([test_data.reset_index(drop=True), predictions_df], axis=1)


# Plot the actual and predicted temperatures
plt.plot(result_df['timestamp'], result_df['temp'], label='Actual')
plt.plot(result_df['timestamp'], predictions, label='Predicted')
plt.xlabel('Week')
plt.ylabel('Temperature')
plt.title('ARIMA Model - Weekly Average Temperature Prediction')
plt.legend()
plt.show()

df.dropna(inplace=True)

X = df[['temp', 'humidity', 'windspeed', 'season']]

X['encoded_season'] = label_encoder.fit_transform(X['season'])
X.drop("season",inplace=True,axis=1)
y = df['weather']
encoded_y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.3, random_state=42)

#Random
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy (Random Forest):", accuracy3)

#DecsionTree
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy (Decision Tree):", accuracy2)

# 3.SVM classifier
# classifier = svm.SVC(kernel='linear')
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy (SVM classifier):", accuracy)


# Perform feature scaling
scaler = StandardScaler()
humidity_data = df['humidity'].values.reshape(-1, 1)
humidity_data_scaled = scaler.fit_transform(humidity_data)

# Apply the elbow method to find the optimal number of clusters
sse = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(humidity_data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(K, sse, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show()


model_nn = MLPRegressor(max_iter=1000)

# Define the parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd']
}

# Perform grid search to find the optimal hyperparameters
grid_search = GridSearchCV(model_nn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model_nn = grid_search.best_estimator_

# Train the best neural network model
best_model_nn.fit(X_train, y_train)

# Make predictions using the neural network model
predictions_nn = best_model_nn.predict(X_test)

# Calculate the mean squared error for the neural network model
mse_nn = mean_squared_error(y_test, predictions_nn)
print("Neural Network Mean Squared Error:", mse_nn)

# Define the random forest regressor model
model_rf = RandomForestRegressor()

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search to find the optimal hyperparameters
grid_search = GridSearchCV(model_rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model_rf = grid_search.best_estimator_
print("Best Model ",best_model_rf)
# Train the best random forest regressor model
best_model_rf.fit(X_train, y_train)

# Make predictions using the random forest regressor model
predictions_rf = best_model_rf.predict(X_test)

# Calculate the mean squared error for the random forest regressor model
mse_rf = mean_squared_error(y_test, predictions_rf)
print("Random Forest Mean Squared Error:", mse_rf)