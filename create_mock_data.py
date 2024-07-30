import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Define the time delta to add
years_to_add = 8
months_to_add = 0
days_to_add = 6

# Load the data
data_file = 'data/household_data_1min_singleindex.csv'
data = pd.read_csv(data_file, parse_dates=['cet_cest_timestamp'])  # Assuming the column with dates is named 'timestamp'

# Add the time delta to each date
data['cet_cest_timestamp'] = data['cet_cest_timestamp'] + relativedelta(years=years_to_add, months=months_to_add) + timedelta(days=days_to_add)
data['cet_cest_timestamp'] = pd.to_datetime(data['cet_cest_timestamp'], utc=True)
# Get today's date
today = datetime.now().date()  # Normalize to midnight for accurate comparison

# Filter out rows where the cet_cest_timestamp is before today
filtered_data = data[data['cet_cest_timestamp'].dt.date >= today]

# Select only the desired columns
columns_to_keep = [
    'cet_cest_timestamp',
    'DE_KN_residential2_circulation_pump',
    'DE_KN_residential2_dishwasher',
    'DE_KN_residential2_freezer',
    'DE_KN_residential2_washing_machine',
    'DE_KN_residential2_grid_import',
    'DE_KN_residential1_pv'
]
filtered_data = filtered_data[columns_to_keep]

# delete rows from 2025-01-01 onwards
filtered_data = filtered_data[filtered_data['cet_cest_timestamp'].dt.year < 2025]

# Save the modified data to a new file
mock_data_file = 'data/mock_data.csv'
filtered_data.to_csv(mock_data_file, index=False)

print(f"Modified data saved to {mock_data_file}")