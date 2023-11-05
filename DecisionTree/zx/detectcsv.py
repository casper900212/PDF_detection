import pandas as pd

# Load your CSV file into a Pandas DataFrame
df = pd.read_csv(input("Input csv file: "))

# Iterate through the DataFrame to check for NaN values in the "Malware" column
for index, row in df.iterrows():
    if pd.isna(row['Malware']):
        print(f"NaN value found in 'Malware' column at row {index + 1}.")
