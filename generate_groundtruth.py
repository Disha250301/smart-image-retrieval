import pandas as pd
import os

# Path to CSV file
csv_path = 'monuments.csv'  # Adjust if placed elsewhere

# Output groundtruth folder
groundtruth_folder = 'dataset/groundtruth'
os.makedirs(groundtruth_folder, exist_ok=True)

# Read the CSV
df = pd.read_csv(csv_path)

# Ensure consistent naming
df['FileName'] = 'image' + df['Image ID'].astype(str) + '.png'

# Group by Monument Name
grouped = df.groupby('Monument Name')

# Generate groundtruth files
for monument, group in grouped:
    relevant_files = group['FileName'].tolist()
    for file in relevant_files:
        file_id = os.path.splitext(file)[0]
        with open(os.path.join(groundtruth_folder, f'{file_id}.txt'), 'w') as f:
            for rel_file in relevant_files:
                f.write(rel_file + '\n')

print(" Groundtruth files generated successfully.")
