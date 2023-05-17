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
#3(a)
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

def create_weather_table(cursor):
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
