import csv
import datetime

# Define the data structure for parking records
parking_records = {}
# Read in the parking record data from the CSV file at the start of the program
with open('parking_records.csv') as csvfile:
    reader = csv.reader(csvfile)
    parking_records = {}
    for row in reader:
        record_id = int(row[0])
        start_time_str = row[1]
        exit_time_str = row[2]
        registration_number = row[3]
        parking_space = row[4]
        
        start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        try:
            exit_time = datetime.datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S') if exit_time_str else None
        except ValueError:
            exit_time = None
            print(f"Invalid exit time for vehicle {registration_number}")
        
        parking_records[registration_number] = [start_time, exit_time, parking_space, record_id]


# Define the number of parking spaces and available parking spaces
num_parking_spaces = 100
available_parking_spaces = num_parking_spaces - len(parking_records)

while True:
    # Display the number of available parking spaces and prompt the user to enter their choice
    print(f'There are {available_parking_spaces} available parking spaces.')
    choice = input('Enter "enter" or "exit" to park or remove your vehicle, or "query" to view your parking record: ')

    if choice == 'enter':
    # Generate a new ticket number, record the entry time, registration number and assign a parking space identifier to the vehicle
        ticket_number = str(len(parking_records) + 1)
        entry_time = datetime.datetime.now()
        registration_number = input('Enter the registration number of your vehicle: ')
        parking_space_id = str(len(parking_records) % num_parking_spaces + 1)
        parking_records = [{'ticket_number': ticket_number, 'entry_time': entry_time, 'exit_time': None, 'parking_space_id': parking_space_id, 'registration_number': registration_number}]
        print(f'Ticket number {ticket_number} has been assigned to your vehicle with registration number {registration_number}. Please park in space {parking_space_id}.')
        available_parking_spaces -= 1


    elif choice == "exit":
        registration_number = input("Please enter the vehicle's registration number: ")
        if registration_number in parking_records:
            record = parking_records[registration_number]
            start_time = record[0]
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).seconds // 3600
            if duration == 0:
                duration = 1
            fee = duration * 2
            print(f"Your parking fee is £{fee}.")
            record[2] = end_time
            record[3] = fee
            available_parking_spaces.append(record[1])
            del parking_records[registration_number]
            print(f"The parking space {record[1]} is now available.")
            print(f"There are {len(available_parking_spaces)} parking spaces available.")
        else:
            print("Vehicle not found in the parking lot.")


    elif choice == 'query':
    # Prompt the user to enter their ticket number and display their parking record
        ticket_number = input('Please enter your ticket number: ')
        for record in parking_records:
            if record['ticket_number'] == ticket_number:
                print(f'Parking record for ticket number {ticket_number}:')
                print(f'Registration number: {record["registration_number"]}')
                print(f'Entry time: {record["entry_time"].strftime("%Y-%m-%d %H:%M:%S")}')
                if record['exit_time']:
                    print(f'Exit time: {record["exit_time"].strftime("%Y-%m-%d %H:%M:%S")}')
                else:
                    print('Exit time: Not recorded')
                print(f'Parking space ID: {record["parking_space_id"]}')
                break
        else:
            print('Invalid ticket number. Please try again.')


    # Save the parking record data to the CSV file before closing the program
    with open('parking_records.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for record in parking_records.values():  # Iterate over the dictionary's values
            entry_time_str = record[0].strftime('%Y-%m-%d %H:%M:%S')
            exit_time_str = record[1].strftime('%Y-%m-%d %H:%M:%S') if record[1] else ''
            writer.writerow([record[3], entry_time_str, exit_time_str, record[2], record[2]])

