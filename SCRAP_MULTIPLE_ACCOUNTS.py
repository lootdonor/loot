import requests
import json
import pandas as pd
from tqdm import tqdm
import time

api_key = 'YOUR DATA HERE'  # API key

input_file_path = "YOUR DATA HERE"
output_file_path = "YOUR DATA HERE"
backup_file_path = "YOUR DATA HERE"  # Backup file path

# Read account_ids from the input file and remove duplicates
with open(input_file_path, "r") as f:
    account_ids = set([line.strip().strip("'") for line in f.readlines()])

# Set to store valid account IDs
valid_account_ids = set()

# Split account_ids into batches of 50
batch_size = 50
account_id_batches = [list(account_ids)[i:i+batch_size] for i in range(0, len(account_ids), batch_size)]

# Make API requests for each batch of account_ids and save the data in a list
match_history_data = []
for batch in tqdm(account_id_batches):
    batch_data = []
    for account_id in batch:
        url = f'https://fortnite-api.com/v2/stats/br/v2/{account_id}'
        headers = {'Authorization': api_key}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            match_history = response.json()
            batch_data.append(match_history)
            valid_account_ids.add(account_id)  # Add to valid account IDs

        # Introduce a delay of 0.33 seconds between API requests
        time.sleep(0.33)

    match_history_data.extend(batch_data)

# Create a DataFrame from the match_history_data list
df = pd.json_normalize(match_history_data)

# Save the DataFrame in a CSV file
df.to_csv(output_file_path, index=False)

# Write valid_account_ids back to file
with open(input_file_path, "w") as f:
    for id in valid_account_ids:
        f.write(f"{id}\n")

# Write valid_account_ids to backup file
with open(backup_file_path, "w") as f:
    for id in valid_account_ids:
        f.write(f"{id}\n")

print("Match history data has been saved.")
