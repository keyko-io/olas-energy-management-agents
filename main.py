import os
import pandas as pd
import tensorflow as tf
import time
import logging
import coloredlogs
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import Dict
from utils.model_utils import load_models, make_prediction, preprocess_real_time_data_with_features
import matplotlib.dates as mdates

# Logging configuration
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')

SEQ_LENGTH = 60  # Sequence length used for prediction

# Path to saved models
SCALER_PATH = 'models/scaler.pkl'
PCA_PATH = 'models/pca.pkl'
MODEL_PATH = 'models/transformer_model.keras'
DATA_FILE = 'data/real_time_data.csv'
PAST_DATA_URL = 'http://localhost:8080/past-data'
API_URL = 'http://localhost:8080/energy-data'

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs")
    except RuntimeError as e:
        logger.error(e)


# Load models
logger.info("Loading models...")
scaler, pca, model = load_models(SCALER_PATH, PCA_PATH, MODEL_PATH)

# Setup for real-time plotting
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

lines = []
for i in range(3):
    line, = ax[i].plot([], [], lw=2)
    lines.append(line)

ax[0].set_title('Energy Consumed')
ax[1].set_title('Energy Produced')
ax[2].set_title('Air Conditioning ON/OFF')

# Set up date formatting for x-axis
for a in ax:
    a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    a.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    a.relim()
    a.autoscale_view()

xdata, ydata1, ydata2, ydata3 = [], [], [], []

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update_plot(frame):
    for i, ydata in enumerate([ydata1, ydata2, ydata3]):
        lines[i].set_data(xdata, ydata)
        ax[i].relim()
        ax[i].autoscale_view()
    fig.autofmt_xdate()
    return lines

ani = animation.FuncAnimation(fig, update_plot, frames=200, init_func=init, blit=True)

def fetch_data_from_api(url: str) -> Dict:
    """
    Fetches data from an external API.

    :param url: URL of the API endpoint.
    :return: Dictionary with the fetched data.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def save_data_to_csv(data: Dict, file_path: str) -> None:
    """
    Saves fetched data to a CSV file.

    :param data: Dictionary with the data to save.
    :param file_path: Path to the CSV file.
    """
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def load_real_time_data(file_path: str, n: int = 60) -> pd.DataFrame:
    """
    Load the latest real-time data from a CSV file.

    :param file_path: Path to the CSV file containing real-time data.
    :param n: Number of most recent rows to load.
    :return: DataFrame with the loaded data.
    """
    logger.info("Loading real-time data")
    data = pd.read_csv(file_path, parse_dates=['cet_cest_timestamp'], index_col='cet_cest_timestamp', low_memory=False)
    return data.tail(n)  # Return the last n rows

def main() -> None:
    """
    Main function to monitor real-time data and make predictions every minute.
    """
    #delete real_time_data.csv
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    while True:
        # Fetch data from the API
        try:
            data = fetch_data_from_api(API_URL)
            save_data_to_csv(data, DATA_FILE)
        except Exception as e:
            logger.error(f"Error fetching data from API: {e}")
            time.sleep(60)
            continue

        # Load real-time data
        data = load_real_time_data(DATA_FILE)

        # If there are less than 60 rows, fetch past data
        if len(data) < SEQ_LENGTH:
            logger.info("Not enough data, fetching past data...")
            try:
                past_data = fetch_data_from_api(PAST_DATA_URL)
                past_data_df = pd.DataFrame(past_data)
                past_data_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
                data = load_real_time_data(DATA_FILE)
            except Exception as e:
                logger.error(f"Error fetching past data: {e}")
                time.sleep(60)
                continue

        # Preprocess the data
        data_preprocessed = preprocess_real_time_data_with_features(data)
        
        ac_on = make_prediction(data_preprocessed, scaler, pca, model, SEQ_LENGTH)

        # Make prediction
        if ac_on:
            logger.info("Turn on air conditioning")
        else:
            logger.info("Do not turn on air conditioning")

        # Update real-time plot data
        current_time = datetime.now()
        xdata.append(current_time)
        ydata1.append(data['DE_KN_residential2_grid_import'].iloc[-1])
        ydata2.append(data['DE_KN_residential1_pv'].iloc[-1])
        ydata3.append(1 if ac_on else 0)

        # Keep only the last 60 points
        if len(xdata) > 60:
            xdata.pop(0)
            ydata1.pop(0)
            ydata2.pop(0)
            ydata3.pop(0)

        plt.pause(1)  # Pause to update the plot

        # Wait one minute before the next check
        time.sleep(60)

if __name__ == "__main__":
    main()
