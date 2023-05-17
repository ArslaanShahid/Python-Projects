########################### APPENDIX ################################

############################ PART I ###############################

#### Task 1
print("I am task 1 solution")
import os
import sqlite3


def connect_database(database_file):
    # Create the database and connect to it
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    return conn, cursor

def close_database(conn):
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def import_dataset(cursor):
    # Import the dataset into "CarSharing" table
    with open('CarSharing.csv', 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            values = line.strip().split(',')
            cursor.execute('''
                INSERT INTO CarSharing (id, timestamp, season, holiday, workingday, weather, temp, temp_feel, humidity, windspeed, demand)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)

def create_backup1_table(cursor):
    # Create the "Backup1" table and copy randomly selected columns from "CarSharing"
    cursor.execute('''
        CREATE TABLE Backup1 AS
        SELECT id, timestamp, season, holiday, workingday
        FROM CarSharing
        ORDER BY RANDOM()
        LIMIT (SELECT COUNT(*) FROM CarSharing) / 2
    ''')

def create_backup2_table(cursor):
    # Create the "Backup2" table and copy the remaining columns from "CarSharing"
    cursor.execute('''
        CREATE TABLE Backup2 AS
        SELECT id, weather, temp, temp_feel, humidity, windspeed, demand
        FROM CarSharing
        WHERE id NOT IN (SELECT id FROM Backup1)
    ''')

#### Task 2
print("I am task 2 solution")

def Add_humidity_category_column(cursor):
    #Create Humidity Column
    cursor.execute('ALTER TABLE CarSharing ADD COLUMN humidity_category TEXT')

    cursor.execute('PRAGMA table_info(CarSharing)')
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    if 'humidity_category' in column_names:
        print("The 'humidity_category' column already exists.")
        return

    # Update the "humidity_category" values based on the "humidity" column
    cursor.execute('''
        UPDATE CarSharing
        SET humidity_category =
            CASE
                WHEN humidity <= 55 THEN 'Dry'
                WHEN humidity > 55 AND humidity <= 65 THEN 'Sticky'
                WHEN humidity > 65 THEN 'Oppressive'
            END
    ''')

#### Task 3
print("I am task 3 solution")
#3(a)
def create_weather_table(cursor):
#3(b)
def assign_code_to_values(cursor):
    # Get distinct values of workingday and holiday columns
    cursor.execute('SELECT DISTINCT workingday FROM CarSharing')
    workingday_values = [row[0] for row in cursor.fetchall()]

    cursor.execute('SELECT DISTINCT holiday FROM CarSharing')
    holiday_values = [row[0] for row in cursor.fetchall()]

    # Assign numbers to workingday values
    workingday_codes = {value: code for code, value in enumerate(workingday_values, start=1)}

    # Assign numbers to holiday values
    holiday_codes = {value: code for code, value in enumerate(holiday_values, start=1)}

    # Check if columns already exist
    cursor.execute('PRAGMA table_info(CarSharing)')
    columns = [row[1] for row in cursor.fetchall()]

    # Add workingday_code column if it doesn't exist
    if 'workingday_code' not in columns:
        cursor.execute('ALTER TABLE CarSharing ADD COLUMN workingday_code INTEGER')

    # Add holiday_code column if it doesn't exist
    if 'holiday_code' not in columns:
        cursor.execute('ALTER TABLE CarSharing ADD COLUMN holiday_code INTEGER')

    # Update workingday_code and holiday_code values
    for value in workingday_values:
        cursor.execute('UPDATE CarSharing SET workingday_code = ? WHERE workingday = ?', (workingday_codes[value], value))

    for value in holiday_values:
        cursor.execute('UPDATE CarSharing SET holiday_code = ? WHERE holiday = ?', (holiday_codes[value], value))

    # Commit the changes
    conn.commit()
#3(c)
def create_holiday_table(cursor):
    # Create the "Holiday" table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Holiday (
            id INTEGER PRIMARY KEY,
            holiday INTEGER,
            workingday INTEGER,
            workingday_code INTEGER,
            holiday_code INTEGER
        )
    ''')

    # Copy the specified columns from "CarSharing" to "Holiday" table
    cursor.execute('''
        INSERT INTO Holiday (id, holiday, workingday, workingday_code, holiday_code)
        SELECT id, holiday, workingday, workingday_code, holiday_code
        FROM CarSharing
    ''')

    # Drop the specified columns from "CarSharing" table
    cursor.execute('''
        CREATE TABLE CarSharingNew AS
        SELECT id, season, timestamp, weather, temp, temp_feel, humidity, windspeed, demand
        FROM CarSharing
    ''')

    # Rename the new table to "CarSharing"
    cursor.execute('DROP TABLE CarSharing')
    cursor.execute('ALTER TABLE CarSharingNew RENAME TO CarSharing')

    # Commit the changes
    conn.commit()
#3(d)
def create_time_table(cursor):
    # Create the "Time" table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Time (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            hour INTEGER,
            weekday_name TEXT,
            month TEXT,
            season_name TEXT
        )
    ''')

    # Populate the "Time" table by selecting required columns from "CarSharing"
    cursor.execute('''
        INSERT INTO Time (id, timestamp, hour, weekday_name, month, season_name)
        SELECT id,
               strftime('%Y-%m-%d %H:%M:%S', timestamp) AS timestamp,
               strftime('%H', timestamp) AS hour,
               strftime('%w', timestamp) AS weekday_name,
               strftime('%m', timestamp) AS month,
               CASE
                   WHEN season = 1 THEN 'Spring'
                   WHEN season = 2 THEN 'Summer'
                   WHEN season = 3 THEN 'Fall'
                   WHEN season = 4 THEN 'Winter'
               END AS season_name
        FROM CarSharing
    ''')

    # Drop the "timestamp" and "season" columns from "CarSharing" table
    cursor.execute('''
        CREATE TABLE CarSharingNew AS
        SELECT id, weather, temp, temp_feel, humidity, windspeed, demand
        FROM CarSharing
    ''')

    # Rename the new table to "CarSharing"
    cursor.execute('DROP TABLE CarSharing')
    cursor.execute('ALTER TABLE CarSharingNew RENAME TO CarSharing')

    # Commit the changes
    conn.commit()

    # Create the "Weather" table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Weather (
            id INTEGER PRIMARY KEY,
            weather INTEGER,
            temp REAL,
            temp_feel REAL,
            humidity INTEGER,
            windspeed REAL,
            humidity_category TEXT
        )
    ''')

    # Copy the required columns from "CarSharing" to "Weather" table
    cursor.execute('''
        INSERT INTO Weather (id, weather, temp, temp_feel, humidity, windspeed, humidity_category)
        SELECT id, weather, temp, temp_feel, humidity, windspeed, humidity_category
        FROM CarSharing
    ''')

    # Get the column names from the CarSharing table
    cursor.execute("PRAGMA table_info('CarSharing')")
    columns = [column[1] for column in cursor.fetchall()]

    # Generate the DROP statements for columns other than 'id' and 'demand'
    drop_statements = [
        f"ALTER TABLE CarSharing DROP COLUMN {column}"
        for column in columns if column not in ['id', 'demand']
    ]

    # Execute the DROP statements
    for drop_statement in drop_statements:
        cursor.execute(drop_statement)

    # Commit the changes
    conn.commit()

#### Task 4
print("I am Task 4 Solution")
#A) The date and time (timestamp) when we had the lowest temperature and the corresponding demand rate.
Query_A = '''
    SELECT Time.timestamp, CarSharing.demand
    FROM Time
    JOIN CarSharing ON Time.id = CarSharing.id
    JOIN Weather ON Weather.id = CarSharing.id
    WHERE Weather.temp = (SELECT MIN(temp) FROM Weather)
'''

#B) The average, highest, and lowest windspeed and humidity for working days (i. e., workingday=“Yes”) and non-working days ((i. e., workingday=“No”) in 2017 andthe corresponding windspeed and humidity values.
Query_B = '''
    SELECT h.workingday,
           AVG(w.windspeed) AS average_windspeed,
           MAX(w.windspeed) AS highest_windspeed,
           MIN(w.windspeed) AS lowest_windspeed,
           AVG(w.humidity) AS average_humidity,
           MAX(w.humidity) AS highest_humidity,
           MIN(w.humidity) AS lowest_humidity
    FROM Weather AS w
    JOIN Time AS t ON w.id = t.id
    JOIN Holiday AS h ON w.id = h.id
    WHERE h.workingday = 'Yes' OR h.workingday = 'No'
        AND t.timestamp LIKE '2017-%'
    GROUP BY h.workingday
'''
#C) The weekday, month, and season when we had the highest average demand ratesthroughout 2017 and the corresponding average demand rates. 
Query_C = '''
SELECT
    Time.weekday_name,
    Time.month,
    Time.season_name,
    AVG(CarSharing.demand) AS average_demand
FROM
    Time
    JOIN CarSharing ON Time.id = CarSharing.id
WHERE
    strftime('%Y', Time.timestamp) = '2017'
GROUP BY
    Time.weekday_name, Time.month, Time.season_name
HAVING
    AVG(CarSharing.demand) = (
        SELECT
            MAX(avg_demand)
        FROM
            (SELECT
                AVG(CarSharing.demand) AS avg_demand
            FROM
                Time
                JOIN CarSharing ON Time.id = CarSharing.id
            WHERE
                strftime('%Y', Time.timestamp) = '2017'
            GROUP BY
                Time.weekday_name, Time.month, Time.season_name)
        )
'''
#D) The average demand rates for each Dry, Sticky, and Oppressive humidity in 2017 sorted in descending order based on their average demand rates.
Query_D = '''
SELECT Weather.humidity_category, AVG(CarSharing.demand) AS average_demand
FROM CarSharing
JOIN Time ON CarSharing.id = Time.id
JOIN Weather ON CarSharing.id = Weather.id
WHERE strftime('%Y', Time.timestamp) = '2017'
GROUP BY Weather.humidity_category
ORDER BY average_demand DESC
'''

############################ PART 2 ###############################

####Task 1
print("I am Task 1 Solution")

database_file = 'CarDatabase_dataAnalytics.db'
conn = sqlite3.connect(database_file)
# Load CarSharing table into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM CarSharing", conn)
# Drop duplicate rows
print("Drop Dublicating.....")
df.drop_duplicates(inplace=True)
# Drop rows with null values
df.dropna(inplace=True)

####Task 2
print("I am Task 2 Solution")

df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce')


data1 = df[['temp','humidity','windspeed','demand']]

#droping null
data1 = data1.dropna()
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

####Task 3
print("I am Task 3 Solution")

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



####Task 4
print("I am Task 4 Solution")

df.dropna(inplace=True)

X = df[['temp', 'humidity', 'windspeed', 'season']]

X['encoded_season'] = label_encoder.fit_transform(X['season'])
X.drop("season",inplace=True,axis=1)
y = df['weather']
encoded_y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.3, random_state=42)

#1. Random
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy (Random Forest):", accuracy3)

#2. DecsionTree
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy (Decision Tree):", accuracy2)

# 3.SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (SVM classifier):", accuracy)


####Task 5
print("I am Task 5 Solution")

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

####Task 6
print("I am Task 6 Solution")

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