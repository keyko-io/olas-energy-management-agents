import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from typing import Tuple
from utils.transformers import Time2Vector, TransformerBlock
import logging
import coloredlogs

# Logging configuration
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')

def load_models(scaler_path: str, pca_path: str, model_path: str) -> Tuple[StandardScaler, PCA, tf.keras.Model]:
    """
    Load scaler, PCA, and model from the given paths.

    :param scaler_path: Path to the saved scaler.
    :param pca_path: Path to the saved PCA model.
    :param model_path: Path to the saved Transformer model.
    :return: Tuple containing the loaded scaler, PCA, and model.
    """
    scaler: StandardScaler = joblib.load(scaler_path)
    pca: PCA = joblib.load(pca_path)
    model: tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects={"Time2Vector": Time2Vector, "TransformerBlock": TransformerBlock})
    return scaler, pca, model

def make_prediction(data: pd.DataFrame, scaler: StandardScaler, pca: PCA, model: tf.keras.Model, seq_length: int) -> bool:
    """
    Make a prediction on whether to turn on the air conditioning.

    :param data: DataFrame containing the preprocessed data.
    :param scaler: The loaded scaler.
    :param pca: The loaded PCA model.
    :param model: The loaded Transformer model.
    :param seq_length: The length of the sequence used for prediction.
    :return: Boolean indicating whether to turn on the air conditioning.
    """
    logger.info("Making prediction")

    # Ensure data has valid feature names
    data = data[scaler.feature_names_in_]

    # Scale the data
    data_scaled = scaler.transform(data)

    # Apply PCA
    data_pca = pca.transform(data_scaled)

    # Create data sequences
    X_data = []
    for i in range(len(data_pca) - seq_length + 1):
        X_data.append(data_pca[i:(i + seq_length)])
    X_data = np.array(X_data)

    # Make predictions
    predictions = model.predict(X_data)

    # Get the latest prediction
    latest_prediction = predictions[-1, 0]
    return latest_prediction > 0.5  # Return True if the air conditioning should be turned on, otherwise False


def preprocess_real_time_data_with_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the real-time data similarly to the training set, returning a DataFrame with feature names.

    :param data: DataFrame containing the real-time data.
    :return: DataFrame with the preprocessed data and feature names.
    """
    logger.info("Preprocessing real-time data with feature names")
    data['DE_KN_residential2_circulation_pump'] = data['DE_KN_residential2_circulation_pump'].diff().fillna(0)
    data['DE_KN_residential2_dishwasher'] = data['DE_KN_residential2_dishwasher'].diff().fillna(0)
    data['DE_KN_residential2_freezer'] = data['DE_KN_residential2_freezer'].diff().fillna(0)
    data['DE_KN_residential2_washing_machine'] = data['DE_KN_residential2_washing_machine'].diff().fillna(0)
    data['DE_KN_residential2_grid_import'] = data['DE_KN_residential2_grid_import'].diff().fillna(0)
    data['DE_KN_residential1_pv'] = data['DE_KN_residential1_pv']
    data = data.ffill()

    features = [
        'DE_KN_residential2_circulation_pump', 
        'DE_KN_residential2_dishwasher', 
        'DE_KN_residential2_freezer', 
        'DE_KN_residential2_grid_import',
        'DE_KN_residential2_washing_machine',
        'DE_KN_residential1_pv'
    ]

    return data[features]