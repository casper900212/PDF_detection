import csv

# Specify the input and output CSV file paths
input_file = 'features.csv'
output_file1 = 'traindata.csv'
output_file2 = 'testdata.csv'

# Open the input CSV file for reading
with open(input_file, mode='r') as input_csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(input_csv_file)
    
    # Read the header row
    header = next(csv_reader)
    
    # Create a list to store the train rows
    trainrows = [header]  # Include the header in the new CSV
    # Create a list to store the test rows
    testrows = [header]  # Include the header in the new CSV
    
    # Initialize a row counter
    row_count = 1
    
    # Iterate through the rows in the input file
    for row in csv_reader:
        if row_count < 3199:
            if row_count % 3 == 0:
                testrows.append(row)
            else:
                trainrows.append(row)
        else:
            trainrows.append(row)
        row_count += 1
        
input_csv_file.close()

# Open the output CSV file for writing
with open(output_file1, mode='w', newline='') as output1_csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(output1_csv_file)
    
    # Write the train rows to the output file
    csv_writer.writerows(trainrows)

output1_csv_file.close()

with open(output_file2, mode='w', newline='') as output2_csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(output2_csv_file)
    
    # Write the test rows to the output file
    csv_writer.writerows(testrows)

output2_csv_file.close()

print("finished")
