import csv
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

parking_spaces = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]



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

# Define the function to handle entering the car park
def enter_car_park(parking_records, parking_spaces):
    # Check if there are any available parking spaces
    if len(parking_spaces) == 0:
        status_label.config(text="Sorry, car park is full.")
    else:
        # Get the registration number from the entry field
        reg_num = reg_num_entry.get()
        
        # Get the current date and time
        now = datetime.now()
        formatted_time = now.strftime('%m/%d/%Y %I:%M:%S %p')
        
        # Assign the first available parking space and ticket number
        parking_space_id = parking_spaces.pop(0)
        ticket_number = len(parking_records) + 1
        
        # Create a new parking record with the registration number, entry time, space number, and ticket number
        record = {'ticket_number': ticket_number,
                'registration_number': reg_num,
                'entry_time': formatted_time,
                'exit_time': None,
                'parking_space_id': parking_space_id }
        
        # Add the parking record to the list
        parking_records.append(record)
        
        # Update the status label with the remaining available parking spaces and space number assigned to the vehicle
        status_label.config(text="Number of available parking spaces: {}\nSpace number assigned to the vehicle: {}".format(len(parking_spaces), parking_space_id))
        
        # Clear the entry field
        reg_num_entry.delete(0, tk.END)
        reg_num_entry.focus()

def view_available_parking_spaces(parking_spaces):
    """Counts the number of available parking spaces based on the current parking_spaces list.
    Returns an integer representing the number of available parking spaces."""
    return sum(1 for space in parking_spaces if space)


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

def find_record_by_reg_num(registration_number, parking_records):
    """
    Find a parking record in the list of parking records by registration number
    """
    print(registration_number)
    for record in parking_records:
        if record['registration_number'] == registration_number:
            return record
    return None


def exit_car_park(registration_number, parking_records, parking_spaces):
    record = find_record_by_reg_num(registration_number, parking_records)
    if record is None:
        print(f"No record found for vehicle with registration number {registration_number}")
        return
    if record['exit_time'] != '':
        print(f"Vehicle with registration number {registration_number} has already exited the car park")
        return
    entry_time = datetime.strptime(record['entry_time'], '%m/%d/%Y %I:%M:%S %p')
    exit_time = datetime.now()
    duration = exit_time - entry_time
    hours_parked = duration.total_seconds() / 3600
    parking_fee = round(2 * hours_parked, 2)
    record['exit_time'] = exit_time
    print(f"Vehicle with registration number {registration_number} has parked for {hours_parked:.2f} hours and needs to pay £{parking_fee:.2f}")
    parking_space_id = record['parking_space_id']
    parking_space_index = parking_spaces.index(parking_space_id)
    parking_spaces[parking_space_index] = parking_space_id
    print(f"The parking space {parking_space_id} has been freed up")
    print(f"There are {len(parking_spaces)} parking spaces available")

def save_parking_records_to_csv(filename,parking_records):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ticket Number','Registration Number', 'Entry Time', 'Exit Time', 'Parking Space ID'])
        for record in parking_records:
            writer.writerow(record.values())

def on_closing():
    # Save the parking records to the CSV file
    save_parking_records_to_csv('parking_records.csv', parking_records)
    root.destroy()


root = tk.Tk()
root.title("Car Parking")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the entry form
form_frame = tk.Frame(root)
form_frame.pack(padx=10, pady=10)

# Add form labels and entry fields
tk.Label(form_frame, text="Registration Number:").pack(side=tk.LEFT)
reg_num_entry = tk.Entry(form_frame)
reg_num_entry.pack(side=tk.LEFT)

# Add submit button for entry
enter_button = tk.Button(form_frame, text="Enter Car Park", command=lambda: enter_car_park(parking_records, parking_spaces))
enter_button.pack(side=tk.LEFT, padx=5)
# Exit Button
exit_button = tk.Button(root, text="Exit Car Park", command=lambda: exit_car_park(reg_num_entry.get(), parking_records, parking_spaces))
exit_button.pack(padx=5, pady=5)


# Add form labels and entry fields for finding a parked car
ticket_num_label = tk.Label(root, text="Enter Ticket Number:")
ticket_num_label.pack(padx=5, pady=5)
ticket_num_entry = tk.Entry(root)
ticket_num_entry.pack(padx=5, pady=5)

# Add submit button for finding a parked car
query_button = tk.Button(root, text="Query", command=lambda: query_parking_record(parking_records))
query_button.pack(padx=5, pady=5)

# Initialize parking records and spaces
parking_records = read_parking_records_csv('parking_records.csv')
# parking_spaces = list(range(1, 11))

# Add status label
status_label = tk.Label(root, text="Number of available parking spaces: {}".format(view_available_parking_spaces(parking_records)))
status_label.pack(padx=5, pady=5)

# Create the label to display the query status
query_status_label = tk.Label(root, text="")
query_status_label.pack(padx=5, pady=5)

# Start the main loop
root.mainloop()


