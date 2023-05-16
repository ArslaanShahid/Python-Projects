import sqlite3

import pandas as pd
import scipy.stats as stats

database_file = 'CarDatabase_dataAnalytics.db'

conn = sqlite3.connect(database_file)

# Load CarSharing table into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM CarSharing", conn)

print(df)
# Drop duplicate rows
print("Drop Dublicating.....")
check_drop = df.drop_duplicates(inplace=True)
print(check_drop)


# # Handle missing values
# df.dropna(inplace=True)

# # Convert data types if necessary
# df['temp'] = df['temp'].astype(float)
# df['humidity'] = df['humidity'].astype(float)
# df['windspeed'] = df['windspeed'].astype(float)

# # Perform statistical tests
# columns = ['temp', 'humidity', 'windspeed', 'workingday']
# results = {}

# for column in columns:
#     # Perform the statistical test
#     test_statistic, p_value = stats.pearsonr(df[column], df['demand'])
#     results[column] = (test_statistic, p_value)

# # Print the results
# for column, (test_statistic, p_value) in results.items():
#     print(f"Test for {column}:")
#     print(f"Test statistic: {test_statistic}")
#     print(f"P-value: {p_value}")
#     print("")

# # Close the connection
# conn.close()
