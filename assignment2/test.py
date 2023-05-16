import csv
import os
import sqlite3

database_file = 'CarDatabase.db'
dataset_file = 'CarSharing.csv'

# Check if the database file exists
if os.path.exists(database_file):
    print(f"The database file '{database_file}' already exists. Skipping database creation.")
else:
    # Step 1: Create the database and connect to it
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Step 2: Create the "CarSharing" table and import the dataset
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

    with open(dataset_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists
        for row in reader:
            cursor.execute('''
                INSERT INTO CarSharing (id, timestamp, season, holiday, workingday, weather, temp, temp_feel, humidity, windspeed, demand)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)

    # Commit the changes
    conn.commit()

    print(f"The database file '{database_file}' created successfully.")

# Define the cursor variable
conn = sqlite3.connect(database_file)
cursor = conn.cursor() 
# Check if the backup tables already exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Backup1'")
backup1_table_exists = cursor.fetchone()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Backup2'")
backup2_table_exists = cursor.fetchone()

# Create the backup tables if they don't exist
if not backup1_table_exists:
    cursor.execute('''
        CREATE TABLE Backup1 AS SELECT id, timestamp, season, holiday FROM CarSharing
    ''')

if not backup2_table_exists:
    cursor.execute('''
        CREATE TABLE Backup2 AS SELECT workingday, weather, temp, temp_feel, humidity, windspeed, demand FROM CarSharing
    ''')

# Commit the changes
conn.commit()

#humidty Tabled Named:

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(CarSharing)")
columns = cursor.fetchall()
column_names = [column[1] for column in columns]
if 'humidity_category' not in column_names:
    # Add the "humidity_category" column to the "CarSharing" table
    cursor.execute('''
        ALTER TABLE CarSharing
        ADD COLUMN humidity_category TEXT
    ''')

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



# Commit the changes
conn.commit()


# Execute a SELECT query to retrieve and print the updated results
cursor.execute('SELECT * FROM CarSharing')
car_sharing_results = cursor.fetchall()
for row in car_sharing_results:
    print(row)


# Step 3: Execute the SELECT queries and retrieve the results
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

# Close the connection
conn.close()
