import os
import sqlite3

# Step 2: Connect to the SQLite database
database_file = 'CarDatabase.db'
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Step 1: Check if the database file exists
if not os.path.exists(database_file):
    print(f"The database file '{database_file}' does not exist.")
    exit()

else:
    # Step 3: Create the "CarSharing" table if it doesn't exist
    cursor.execute('''
            CREATE TABLE CarSharing (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                season INTEGER,
                holiday INTEGER,
                workingday INTEGER,
                weather INTEGER,
                temp REAL,
                temp_feel REAL,
                humidity REAL,
                windspeed REAL,
                demand INTEGER
            )
        ''')
# Step 3: Import the dataset into "CarSharing" table
with open('CarSharing.csv', 'r') as file:
    next(file)  # Skip the header row
    for line in file:
        values = line.strip().split(',')
        cursor.execute('''
            INSERT INTO CarSharing (id, timestamp, season, holiday, workingday, weather, temp, temp_feel, humidity, windspeed, demand)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)

# Step 4: Create the "Backup1" table and copy randomly selected columns from "CarSharing"
cursor.execute('''
    CREATE TABLE Backup1 AS
    SELECT id, timestamp, season, holiday, workingday
    FROM CarSharing
    ORDER BY RANDOM()
    LIMIT (SELECT COUNT(*) FROM CarSharing) / 2
''')

# Step 5: Create the "Backup2" table and copy the remaining columns from "CarSharing"
cursor.execute('''
    CREATE TABLE Backup2 AS
    SELECT id, weather, temp, temp_feel, humidity, windspeed, demand
    FROM CarSharing
    WHERE id NOT IN (SELECT id FROM Backup1)
''')

# Step 2: Add the "humidity_category" column to the "CarSharing" table
cursor.execute('''
    ALTER TABLE CarSharing ADD COLUMN humidity_category TEXT
''')

# Step 3: Update the "humidity_category" values based on the "humidity" column
cursor.execute('''
    UPDATE CarSharing SET humidity_category = 
    CASE
        WHEN humidity <= 55 THEN 'Dry'
        WHEN humidity > 55 AND humidity < 65 THEN 'Sticky'
        WHEN humidity >= 65 THEN 'Oppressive'
    END
''')
# Commit the changes and close the connection


conn.commit()

# Step 4: Execute the SELECT queries and retrieve the results
cursor.execute('SELECT * FROM CarSharing')
car_sharing_results = cursor.fetchall()
print("CarSharing Table:")
for row in car_sharing_results:
    print(row)

cursor.execute('SELECT * FROM Backup1')
backup1_results = cursor.fetchall()
print("\nBackup1 Table:")
for row in backup1_results:
    print(row)

cursor.execute('SELECT * FROM Backup2')
backup2_results = cursor.fetchall()
print("\nBackup2 Table:")
for row in backup2_results:
    print(row)
conn.close()