import csv
import datetime
from datetime import datetime

parking_spaces = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]



def read_parking_records_csv(file_path):
    parking_records = []
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Skip the header row
        next(reader, None)
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
    entry_time = datetime.now()
    formatted_time = entry_time.strftime('%m/%d/%Y %I:%M:%S %p')
    
    assigned_spaces = [record['parking_space_id'] for record in parking_records]
    available_spaces = set(parking_spaces) - set(assigned_spaces)
    if len(available_spaces) == 0:
        print("Sorry, the car park is full.")
        return
    
    parking_space_id = sorted(available_spaces)[0]
    ticket_number = len(parking_records) + 1
    record = {'ticket_number': ticket_number,
                'registration_number': reg_num,
                'entry_time': formatted_time,
                'exit_time': None,
                'parking_space_id': parking_space_id }

    parking_records.append(record)
    
    # Update available parking spaces
    if parking_space_id not in parking_spaces:
        print(f"Parking space {parking_space_id} is not valid or has already been freed up.")
    else:
        parking_spaces.remove(parking_space_id)
        print(f"The parking space {parking_space_id} has been assigned")
    # Update available parking spaces
    view_available_parking_spaces(parking_spaces, parking_records)

def find_record_by_reg_num(reg_num, parking_records):
    # convert reg_num to integer
    for record in parking_records:
        if record['registration_number'] == reg_num:
            print(record)
            return record
    print(f"No record found for vehicle with registration number {reg_num}")
    return None

def view_available_parking_spaces(parking_spaces, parking_records):
    """
    This function displays the available parking spaces and the number of available spaces.

    Args:
        parking_spaces (list): A list of parking space IDs
        parking_records (list): A list of dictionaries representing parking records

    Returns:
        int: The number of available parking spaces
    """

    # Create a set of all parking spaces that have been assigned
    assigned_spaces = set(record['parking_space_id'] for record in parking_records if record['exit_time'] == '')

    # Calculate the number of available parking spaces
    available_spaces = len(parking_spaces) - len(assigned_spaces)
    print(available_spaces)
    # Display the available parking spaces
    print("Available parking spaces:")
    for space in parking_spaces:
        if space not in assigned_spaces:
            print([space])

    # Display the number of available parking spaces
    # print("There are", available_spaces, "parking spaces available")
    count = available_spaces
    print(f'Available Parking Space',count)
    return count

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
    index = parking_spaces.index(parking_space_id)
    parking_spaces[index] = True
    print(f"The parking space {parking_space_id} has been freed up")
    print(f"There are {view_available_parking_spaces(parking_spaces,parking_records)} parking spaces available")

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

def main():
    # Read the initial parking records from the CSV file
    parking_records = read_parking_records_csv('parking_records.csv')
    
    # Get the list of parking spaces
    # parking_spaces = view_available_parking_spaces(parking_spaces, parking_records)
    
    # Update the parking spaces list with the initial parking records
    for record in parking_records:
        if record['parking_space_id'] in parking_spaces:
            parking_spaces.remove(record['parking_space_id'])

    #Car park menu loop
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
        view_available_parking_spaces(parking_spaces,parking_records)
    elif choice == "4":
        query_parking_record_by_ticket_number(parking_records)
    elif choice == "5":
        save_parking_records_to_csv(parking_records,'parking_records.csv')
        break
    else:
        print("Invalid choice. Please try again.")
