from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import torch
import os
import numpy as np
import joblib
from datetime import datetime, timedelta
from dateutil import tz
from utils.transformers import TransformerModel
from functools import wraps
import onnxruntime as ort



app = Flask(__name__)
DATA_FILE = 'data/residential4_features.csv'
MODEL_FILE = 'models/transformer_model.pth'
SCALER_FILE = 'models/scaler.pkl'
ONNX_MODEL_FILE = 'models/transformer_model.onnx'
device = torch.device("cpu")

# model = TransformerModel(
#     seq_length=60,
#     input_dim=5,  # 2 features (pv, grid_import) + 3 (hour, day_of_year, day_of_week)
#     head_size=256,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     dropout=0.1,
#     mlp_dropout=0.1
# )

# model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
# model.to(device)
# model.eval()  # Set model to evaluation mode

# # Define el nombre del archivo ONNX

# # Define un tensor de entrada de ejemplo, asegurándote de que tenga la misma forma que los datos de entrada que usas para el modelo
# example_input = torch.randn(1, 60, 5)  # 1 batch size, 60 sequence length, 5 features

# # Exportar el modelo a formato ONNX
# torch.onnx.export(
#     model,                          # El modelo PyTorch
#     example_input,                  # El tensor de entrada de ejemplo
#     ONNX_MODEL_FILE,                # Ruta donde se guardará el archivo ONNX
#     export_params=True,             # Exportar también los parámetros del modelo
#     opset_version=13,               # Versión de ONNX a usar (puedes ajustar esto según lo que necesites)
#     input_names=['energy_features'], # Nombre de entrada descriptivo
#     output_names=['prediction_output'], # Nombre de salida descriptivo
#     dynamic_axes={
#         'energy_features': {0: 'batch_size', 1: 'seq_length'},  # Tamaño de lote y longitud de secuencia como dimensiones dinámicas
#         'prediction_output': {0: 'batch_size'}  # Salida dinámica en función del tamaño del lote
#     }
# )

# print(f"Modelo exportado a ONNX y guardado en {ONNX_MODEL_FILE}")

# Cargar el modelo ONNX utilizando ONNX Runtime
onnx_session = ort.InferenceSession(ONNX_MODEL_FILE)

scaler = joblib.load(SCALER_FILE)

VALID_TOKEN = "MyToken"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            token = auth_header.split(" ")[1]
        else:
            token = None

        if token != VALID_TOKEN:
            return jsonify({"message": "Invalid or missing token!"}), 401

        return f(*args, **kwargs)
    
    return decorated

def preprocess_data_from_api(data_list):
    """
    Preprocesses the list of data received from the API to be suitable for the model.

    :param data_list: List of dictionaries containing pv, grid_import, and cet_cest_timestamp.
    :return: A torch tensor ready for the model.
    """
    if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
        raise ValueError("Input must be a list of dictionaries containing the required fields.")

    pv_values = []
    grid_import_values = []
    hours = []
    day_of_years = []
    day_of_weeks = []

    for data in data_list:
        pv_values.append(data['pv'])
        grid_import_values.append(data['grid_import'])
        timestamp = datetime.strptime(data['cet_cest_timestamp'], '%Y-%m-%d %H:%M:%S')
        hours.append(timestamp.hour)
        day_of_years.append(timestamp.timetuple().tm_yday)
        day_of_weeks.append(timestamp.weekday())

    # Create the feature array
    features = np.array([grid_import_values, pv_values, hours, day_of_years, day_of_weeks]).T

    # Normalize the features
    features_normalized = scaler.transform(features)

    # Convert to a NumPy array and add the batch dimension
    features_array = np.expand_dims(features_normalized, axis=0).astype(np.float32)  # Shape will be (1, 60, 5)
    
    return features_array

def load_real_time_data(file_path: str) -> pd.DataFrame:
    """
    Load the latest real-time data from a CSV file.

    :param file_path: Path to the CSV file containing real-time data.
    :return: DataFrame with the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path, parse_dates=['cet_cest_timestamp'], index_col='cet_cest_timestamp', low_memory=False)

def make_prediction(model, processed_data):
    """
    Makes a prediction using the model with the processed data.

    :param model: The loaded Transformer model.
    :param processed_data: Preprocessed data as a torch tensor.
    :return: Prediction result.
    """
    # Ensure the input data has the correct shape (batch_size, seq_length, num_features)
    if processed_data.ndim != 3:
        raise ValueError(f"Expected input to be 3-dimensional, but got shape {processed_data.shape}")

    # Run the model inference
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    
    prediction = model.run([output_name], {input_name: processed_data})
    
    return prediction[0].squeeze()
    
@app.route('/schema.json')
def serve_json():
    return send_from_directory(directory='.', path='schema.json')

@app.route('/predict', methods=['POST'])
@token_required
def predict_energy():
    """
    Endpoint to receive the last 60 minutes of pv, grid_import, and timestamp data and return a prediction.

    :return: JSON response with the prediction result.
    """
    try:
        data = request.json["user_input"]["data"]
        
        # Verify that the data has exactly 60 data points
        if len(data) != 60:
            return jsonify({"error": "Exactly 60 data points are required."}), 400

        # Preprocess the data
        processed_data = preprocess_data_from_api(data)

        # Make prediction
        prediction = make_prediction(onnx_session, processed_data)


        result = "The consumption will be lower than production in the next hour." if prediction > 0.5 else "The consumption will not be lower than production in the next hour."

        return jsonify({"prediction": result, "prediction_class": 1 if prediction > 0.5 else 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/energy-data', methods=['GET'])
@token_required
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
    
@app.route('/devices/<device_id>/last_60min', methods=['GET'])
@token_required
def get_past_data(device_id):
    """
    Endpoint to get the energy consumption and generation data for the past 60 minutes.

    :return: JSON response with the data for the past 60 minutes.
    """
    try:
        # Load real-time data
        data = load_real_time_data(DATA_FILE)
        
        # Get current date and time in UTC
        now = datetime.now().astimezone(tz=tz.gettz('UTC+2'))

        # Round down to the nearest minute (i.e., second 00)
        now = now.replace(second=0, microsecond=0)

        # Calculate the start time (60 minutes ago) and round to the nearest minute
        start_time = now - timedelta(minutes=60)
        start_time = start_time.replace(second=0, microsecond=0)

        # Filter data for the last 60 minutes
        past_data = data.loc[start_time:now]
        
        # Reset index to include the timestamp in the JSON output
        past_data.reset_index(inplace=True)

        # Transform "Mon, 12 Aug 2024 14:44:00 GMT" to "2024-08-12 14:44:00"
        past_data['cet_cest_timestamp'] = past_data['cet_cest_timestamp'].dt.strftime('%Y-%m-%d %H:%M:00')
        
        # Convert the data to JSON
        response = {
            "result": past_data.to_dict(orient='records')
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/devices/<device_id>/switch', methods=['POST'])
@token_required
def control_device(device_id):
    """
    Endpoint to switch on/off a device based on the prediction.

    :param device_id: Device ID to control.
    :return: JSON response with the result of the action.
    """
    try:
        data = request.form
        action = data.get('on')
        if action not in ['true', 'false']:
            return jsonify({"error": "Invalid action. Must be 'true' (on) or 'false' (off)."}), 400

        message = f"Device {device_id} switched {'on' if action == 'true' else 'off'} successfully."
        return jsonify({"message": message, "success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Healthy"})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

