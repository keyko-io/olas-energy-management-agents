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
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model # type: ignore

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
    data['DE_KN_residential1_pv'] = data['DE_KN_residential1_pv'].diff().fillna(0)
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

def inverse_transform_data(scaled_pca_data: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """
    Revert the PCA and StandardScaler transformations to get the original scale data.

    :param scaled_pca_data: Numpy array with the scaled and PCA-transformed data.
    :param scaler: The fitted StandardScaler.
    :param pca: The fitted PCA model.
    :return: Numpy array with the data in the original scale.
    """
    logger.info("Reverting PCA and scaling transformations")
    pca_reversed = pca.inverse_transform(scaled_pca_data)
    original_data = scaler.inverse_transform(pca_reversed)
    return original_data

def load_data(file_path):
    logger.info("Loading data")
    return pd.read_csv(file_path, parse_dates=['cet_cest_timestamp'], index_col='cet_cest_timestamp')

def preprocess_data(sub_dfs):
    logger.info("Preparing data")
    X_minutes = 60

    for key in sub_dfs.keys():
        sub_dfs[key]['grid_import'] = sub_dfs[key]['grid_import'].diff().fillna(0)
        sub_dfs[key]['pv'] = sub_dfs[key]['pv'].diff().fillna(0)

        sub_dfs[key].index = pd.to_datetime(sub_dfs[key].index, utc=True) 
        sub_dfs[key]['hour'] = sub_dfs[key].index.hour
        sub_dfs[key]['day_of_year'] = sub_dfs[key].index.dayofyear
        sub_dfs[key]['day_of_week'] = sub_dfs[key].index.dayofweek

        sub_dfs[key] = sub_dfs[key].dropna() 
        sub_dfs[key]['target'] = 0

        #total consumption = pv + grid_import
        sub_dfs[key].loc[sub_dfs[key]['pv'].rolling(window=X_minutes).sum().shift(-X_minutes) > sub_dfs[key]['grid_import'].rolling(window=X_minutes).sum().shift(-X_minutes), 'target'] = 1
        sub_dfs[key] = sub_dfs[key].dropna()
        sub_dfs[key]['target'] = sub_dfs[key]['target'].astype(int)

    return sub_dfs

def select_features(sub_dfs):
    Xs, ys = [], []
    for key in sub_dfs.keys():
        Xs.append(sub_dfs[key][['grid_import', 'pv', 'hour', 'day_of_year', 'day_of_week']].values)
        ys.append(sub_dfs[key]['target'].values)
    
    return Xs, ys

def normalize_data(models_dir, X_train_list, X_valid_list, X_test_list):
    logger.info("Normalizing data")
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info(f"Loading scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        combined_X_train = np.concatenate(X_train_list, axis=0)
        scaler.fit(combined_X_train)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saving scaler to {scaler_path}")
    X_train_list = [scaler.transform(X) for X in X_train_list]
    X_valid_list = [scaler.transform(X) for X in X_valid_list]
    X_test_list = [scaler.transform(X) for X in X_test_list]
    return X_train_list, X_valid_list, X_test_list

def apply_pca(models_dir, X_train_list, X_valid_list, X_test_list, ):
    logger.info("Applying PCA")
    pca_path = os.path.join(models_dir, 'pca.pkl')
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        logger.info(f"PCA model loaded from {pca_path}")
    else:
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        combined_X_train = np.concatenate(X_train_list, axis=0)
        pca.fit(combined_X_train)
        joblib.dump(pca, pca_path)
        logger.info(f"Saved PCA model to {pca_path}")
    X_train_list = [pca.transform(X) for X in X_train_list]
    X_valid_list = [pca.transform(X) for X in X_valid_list]
    X_test_list = [pca.transform(X) for X in X_test_list]
    return X_train_list, X_valid_list, X_test_list

def remove_nans(X, y):
    logger.info("Deleting rows with NaN values")
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y

def create_tf_dataset(X, y, seq_length, batch_size, dataset_name):
    logger.info(f"Creating dataset {dataset_name}")
    X_data, y_data = [], []
    for i in range(len(X) - seq_length - 1):
        X_data.append(X[i:(i + seq_length)])
        y_data.append(y[i + seq_length])
    X_data, y_data = np.array(X_data), np.array(y_data)
    with tf.device('/gpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
        dataset = dataset.cache().shuffle(buffer_size=len(X_data)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def build_model(seq_length, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
    inputs = layers.Input(shape=input_shape)
    time_embedding = Time2Vector(seq_length)(inputs)
    x = layers.Concatenate(axis=-1)([inputs, time_embedding])
    x = layers.Dense(head_size, dtype='float32')(x)
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout)(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def plot_loss(history):
    logger.info("Plotting training and validation loss")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# Divide data into train, validation, and test sets
def split_data(Xs, ys, train_ratio=0.7, val_ratio=0.15):
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []

    for X, y in zip(Xs, ys):
        n = len(X)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + val_ratio))

        X_train.append(X[:train_end])
        X_valid.append(X[train_end:valid_end])
        X_test.append(X[valid_end:])
        
        y_train.append(y[:train_end])
        y_valid.append(y[train_end:valid_end])
        y_test.append(y[valid_end:])

    return X_train, X_valid, X_test, y_train, y_valid, y_test