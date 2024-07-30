from flask import Flask, jsonify
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil import tz

app = Flask(__name__)
DATA_FILE = 'data/mock_data.csv'

def load_real_time_data(file_path: str) -> pd.DataFrame:
    """
    Load the latest real-time data from a CSV file.

    :param file_path: Path to the CSV file containing real-time data.
    :return: DataFrame with the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path, parse_dates=['cet_cest_timestamp'], index_col='cet_cest_timestamp', low_memory=False)

@app.route('/energy-data', methods=['GET'])
def get_energy_data():
    """
    Endpoint to get the energy consumption and generation data for the current date and time.

    :return: JSON response with the data for the current date and time.
    """
    try:
        # Load real-time data
        data = load_real_time_data(DATA_FILE)
        
        # Get current date and time
        now = datetime.now()
        
        # Filter data for the current date and time
        current_data = data.loc[now.strftime('%Y-%m-%d %H:%M:00')]
        current_data['cet_cest_timestamp'] = now.strftime('%Y-%m-%d %H:%M:00')
        
        # Convert the data to JSON
        response = current_data.to_dict()
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/past-data', methods=['GET'])
def get_past_data():
    """
    Endpoint to get the energy consumption and generation data for the past 60 minutes.

    :return: JSON response with the data for the past 60 minutes.
    """
    try:
        # Load real-time data
        data = load_real_time_data(DATA_FILE)
        
        # Get current date and time in UTC
        now = datetime.now().astimezone(tz=tz.gettz('UTC+2'))

        # Calculate the start time (60 minutes ago)
        start_time = now - timedelta(minutes=60)
        
        # Filter data for the last 60 minutes
        past_data = data.loc[start_time:now]
        
        # Reset index to include the timestamp in the JSON output
        past_data.reset_index(inplace=True)
        
        # Convert the data to JSON
        response = past_data.to_dict(orient='records')
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
