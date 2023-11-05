import csv

# Specify the CSV file path
csv_file = 'test_features.csv'

# Open the CSV file for reading and writing
with open(csv_file, mode='r+') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    
    # Create a list to hold the rows with the new column
    rows = []
    
    # Read the header row
    header = next(csv_reader)
    
    # Add a new column header to the existing headers
    header.append('Malware')
    rows.append(header)
    
    # Initialize a row counter
    row_count = 0
    
    # Iterate through the rows in the input file
    for row in csv_reader:
        if 'clean' in row[1]:
            row.append('0')
            print("condition1")
        elif 'malicious' in row[1]:
            row.append('1')
            print("condition2")
        # Append the row to the list
        rows.append(row)
        
    # Move the file pointer to the beginning of the file
    csv_file.seek(0)
    
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)
    
    # Write all the rows back to the file
    csv_writer.writerows(rows)
    
    # If the new data is shorter than the old data, truncate the file
    # csv_file.truncate()
csv_file.close()

