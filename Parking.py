import csv
import datetime
from datetime import datetime

parking_spaces = {}



def read_parking_records_csv(file_path):
    parking_records = []
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Skip the header row
        next(reader)
        # Loop over each row in the CSV file
        for row in reader:
            # Process the row
            ticket_number = row[0]
            registration_number = row[1]
            entry_time = row[2]
            exit_time = row[3]
            parking_space_id = row[4]
            # Add the record to the parking records list
            parking_records.append({
                'ticket_number': ticket_number,
                'registration_number': registration_number,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'parking_space_id': parking_space_id,
            })
    return parking_records

def enter_car_park(parking_records, parking_spaces):
    if len(parking_spaces) == 0:
        print("Sorry, the car park is full.")
        return

    reg_num = input("Enter the vehicle's registration number: ")
    entry_time = datetime.datetime.now()
    formatted_time = entry_time.strftime('%m/%d/%Y %I:%M:%S %p')
    parking_space_id = parking_spaces.pop(0)
    ticket_number = len(parking_records) + 1

    record = {'ticket_number': ticket_number,
                'registration_number': reg_num,
                'entry_time': formatted_time,
                'exit_time': None,
                'parking_space_id': parking_space_id }

    parking_records.append(record)
    print("Vehicle has been assigned parking space ID", parking_space_id,
          "and ticket number", ticket_number)
    print("Number of available parking spaces:", len(parking_spaces))   

def find_record_by_reg_num(reg_num, parking_records):
    for record in parking_records:
        if record['registration_number'] == reg_num:
            return record
    print(f"No record found for vehicle with registration number {reg_num}")
    return None

def view_available_parking_spaces(parking_records):
    """Counts the number of available parking spaces based on the current parking records.
    Returns an integer representing the number of available parking spaces."""
    total_parking_spaces = 9  # Total number of parking spaces
    used_parking_spaces = len(parking_records)  # Counting the number of records in the parking_records list
    available_parking_spaces = total_parking_spaces - used_parking_spaces
    return available_parking_spaces


def exit_car_park(parking_records, parking_spaces):
    reg_num = input("Please enter the vehicle's registration number: ")
    print(reg_num)
    record = find_record_by_reg_num(reg_num, parking_records)
    print(record)
    if record is None:
        print(f"No record found for vehicle with registration number {reg_num}")
        return
    if record['exit_time'] != '':
        print(f"Vehicle with registration number {reg_num} has already exited the car park")
        return
    entry_time = datetime.strptime(record['entry_time'], '%m/%d/%Y %I:%M:%S %p')
    exit_time = datetime.now()
    duration = exit_time - entry_time
    hours_parked = duration.total_seconds() / 3600
    parking_fee = round(2 * hours_parked, 2)
    record['exit_time'] = exit_time
    print(f"Vehicle with registration number {reg_num} has parked for {hours_parked:.2f} hours and needs to pay £{parking_fee:.2f}")
    parking_space_id = record['parking_space_id']
    parking_spaces[parking_space_id] = True
    print(f"The parking space {parking_space_id} has been freed up")
    print(f"There are {view_available_parking_spaces(parking_spaces)} parking spaces available")


def query_parking_record_by_ticket_number(parking_records):
    ticket_number = input("Enter ticket number: ")
    for record in parking_records:
        if record["ticket_number"] == ticket_number:
            print(f"Ticket Number: {record['ticket_number']}")
            print(f"Registration Number: {record['registration_number']}")
            print(f"Parking Space ID: {record['parking_space_id']}")
            print(f"Entry Time: {record['entry_time']}")
            print(f"Exit Time: {record['exit_time'] if record['exit_time'] else 'Not Exited Yet'}")
            return
    print("Ticket number not found.")

def save_parking_records_to_csv(parking_records, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ticket Number','Registration Number', 'Entry Time', 'Exit Time', 'Parking Space ID'])
        for record in parking_records:
            writer.writerow(record.values())
            
parking_records = read_parking_records_csv('parking_records.csv')

while True:
    print("\nCar Park Menu")
    print("1. Enter the car park")
    print("2. Exit the car park")
    print("3. View available parking spaces")
    print("4. Query parking record by ticket number")
    print("5. Quit")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        enter_car_park(parking_records, parking_spaces)
    elif choice == "2":
        exit_car_park(parking_records, parking_spaces)
    elif choice == "3":
        view_available_parking_spaces(parking_spaces)
    elif choice == "4":
        query_parking_record_by_ticket_number(parking_records)
    elif choice == "5":
        save_parking_records_to_csv(parking_records,'parking_records.csv')
        break
    else:
        print("Invalid choice. Please try again.")
        

# test = view_available_parking_spaces(parking_spaces)
# print(test)


# def show_csv_data(file_path):
#     with open(file_path, 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             print(row)
# data = read_parking_records_csv('parking_records.csv')
# reg = 'ABC-123'
# record = find_record_by_reg_num(reg, data)

# exit_car_park(data, parking_spaces)
# print(exit_car_park)
# if record:
#     print(record)
# else:
#     print('Record not found')