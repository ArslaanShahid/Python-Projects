import csv
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

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

parking_records = read_parking_records_csv('parking_records.csv')

print(parking_records)

def enter_car_park():
    reg_num = reg_num_entry.get()
    if len(parking_spaces) == 0:
        status_label.config(text="Sorry, car park is full.")
    else:
        space_num = parking_spaces.pop(0)
        parking_records.append({"registration_number": reg_num, "space_number": space_num})
        status_label.config(text="Number of available parking spaces: {}".format(len(parking_spaces)))
        reg_num_entry.delete(0, tk.END)

def find_record_by_reg_num():
    reg_num = reg_num_entry.get()
    for record in parking_records:
        if record['registration_number'] == reg_num:
            status_label.config(text="Vehicle with registration number {} is parked in space number {}".format(reg_num, record['space_number']))
            reg_num_entry.delete(0, tk.END)
            return
    status_label.config(text="No record found for vehicle with registration number {}".format(reg_num))
    reg_num_entry.delete(0, tk.END)
    

def view_available_parking_spaces(parking_records):
    """Counts the number of available parking spaces based on the current parking records.
    Returns an integer representing the number of available parking spaces."""
    total_parking_spaces = 9  # Total number of parking spaces
    used_parking_spaces = len(parking_records)  # Counting the number of records in the parking_records list
    available_parking_spaces = total_parking_spaces - used_parking_spaces
    return available_parking_spaces

def query_parking_record(parking_records):
    pr = parking_records
    print(pr)   
    ticket_number = ticket_num_entry.get().strip()
    print(f"Ticket number: '{ticket_number}'")
    print(type(ticket_number))
    for record in parking_records:
        if record["ticket_number"] == str(ticket_number):
            status_label.config(text=f"Ticket Number: {record['ticket_number']}\n"
                                      f"Registration Number: {record['registration_number']}\n"
                                      f"Parking Space ID: {record['parking_space_id']}\n"
                                      f"Entry Time: {record['entry_time']}\n"
                                      f"Exit Time: {record['exit_time'] if record['exit_time'] else 'Not Exited Yet'}")
            return
    status_label.config(text="Ticket number not found.")


root = tk.Tk()
root.title("Car Park Entry and Find Car by Registration Number")

# Create a frame for the entry form
form_frame = tk.Frame(root)
form_frame.pack(padx=10, pady=10)

# Add form labels and entry fields
tk.Label(form_frame, text="Registration Number:").pack(side=tk.LEFT)
reg_num_entry = tk.Entry(form_frame)
reg_num_entry.pack(side=tk.LEFT)
ticket_num_label = tk.Label(root, text="Enter Ticket Number:")
ticket_num_label.pack(padx=5, pady=5)
ticket_num_entry = tk.Entry(root)
ticket_num_entry.pack(padx=5, pady=5)
# Add submit buttons for entry and find
tk.Button(form_frame, text="Enter Car Park", command=enter_car_park).pack(side=tk.LEFT, pady=10)
tk.Button(form_frame, text="Find Car", command=find_record_by_reg_num).pack(side=tk.LEFT, pady=10)
query_button = tk.Button(root, text="Query", command=lambda: query_parking_record(parking_records))
query_button.pack(padx=5, pady=5)


# Initialize parking records and spaces
parking_records = read_parking_records_csv('parking_records.csv')
parking_spaces = list(range(1, 11))
# Add status label
status_label = tk.Label(root, text="Number of available parking spaces: {}".format(view_available_parking_spaces(parking_records)))
status_label.pack(padx=5, pady=5)
# Create the label to display the query status
query_status_label = tk.Label(root, text="")
query_status_label.pack(padx=5, pady=5)


# Start the main loop
root.mainloop()




# def exit_car_park(parking_records, parking_spaces):
#     reg_num = input("Please enter the vehicle's registration number: ")
#     print(reg_num)
#     record = find_record_by_reg_num(reg_num, parking_records)
#     print(record)
#     if record is None:
#         print(f"No record found for vehicle with registration number {reg_num}")
#         return
#     if record['exit_time'] != '':
#         print(f"Vehicle with registration number {reg_num} has already exited the car park")
#         return
#     entry_time = datetime.strptime(record['entry_time'], '%m/%d/%Y %I:%M:%S %p')
#     exit_time = datetime.now()
#     duration = exit_time - entry_time
#     hours_parked = duration.total_seconds() / 3600
#     parking_fee = round(2 * hours_parked, 2)
#     record['exit_time'] = exit_time
#     print(f"Vehicle with registration number {reg_num} has parked for {hours_parked:.2f} hours and needs to pay £{parking_fee:.2f}")
#     parking_space_id = record['parking_space_id']
#     parking_spaces[parking_space_id] = True
#     print(f"The parking space {parking_space_id} has been freed up")
#     print(f"There are {view_available_parking_spaces(parking_spaces)} parking spaces available")

# def query_parking_record_by_ticket_number(parking_records):
#     ticket_number = input("Enter ticket number: ")
#     for record in parking_records:
#         if record["ticket_number"] == ticket_number:
#             print(f"Ticket Number: {record['ticket_number']}")
#             print(f"Registration Number: {record['registration_number']}")
#             print(f"Parking Space ID: {record['parking_space_id']}")
#             print(f"Entry Time: {record['entry_time']}")
#             print(f"Exit Time: {record['exit_time'] if record['exit_time'] else 'Not Exited Yet'}")
#             return
#     print("Ticket number not found.")

# [def save_parking_records_to_csv(parking_records, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Ticket Number','Registration Number', 'Entry Time', 'Exit Time', 'Parking Space ID'])
#         for record in parking_records:
#             writer.writerow(record.values())]
            

# while True:
#     print("\nCar Park Menu")
#     print("1. Enter the car park")
#     print("2. Exit the car park")
#     print("3. View available parking spaces")
#     print("4. Query parking record by ticket number")
#     print("5. Quit")

#     choice = input("Enter your choice (1-5): ")

#     if choice == "1":
#         enter_car_park(parking_records, parking_spaces)
#     elif choice == "2":
#         exit_car_park(parking_records, parking_spaces)
#     elif choice == "3":
#         view_available_parking_spaces(parking_spaces)
#     elif choice == "4":
#         query_parking_record_by_ticket_number(parking_records)
#     elif choice == "5":
#         save_parking_records_to_csv(parking_records,'parking_records.csv')
#         break
#     else:
#         print("Invalid choice. Please try again.")
        

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