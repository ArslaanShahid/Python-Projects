import csv

# Example data
parking_records = [
    {'ticket_number': '1', 'registration_number': 'ABC-123', 'entry_time': '2022-05-01 10:00:00', 'exit_time': '2022-05-01 12:30:00', 'parking_space_id': 'A1'},
    {'ticket_number': '2', 'registration_number': 'DEF-456', 'entry_time': '2022-05-01 12:30:00', 'exit_time': '2022-05-01 14:00:00', 'parking_space_id': 'B2'},
    {'ticket_number': '3', 'registration_number': 'GHI-789', 'entry_time': '2022-05-02 09:00:00', 'exit_time': '022-05-01 14:00:00', 'parking_space_id': 'C3'}
]

# Open a new CSV file for writing
with open('parking_records.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Ticket Number', 'Registration Number', 'Entry Time', 'Exit Time', 'Parking Space ID'])

    # Write each parking record as a row in the CSV file
    for record in parking_records:
        writer.writerow([record['ticket_number'], record['registration_number'], record['entry_time'], record['exit_time'], record['parking_space_id']])
