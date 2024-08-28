import torch
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request

# Inicializar la app Flask
app = Flask(__name__)

# Rutas a los archivos
MODEL_FILE = '../models/transformer_model_full.pth'
SCALER_FILE = '../models/scaler.pkl'

# Cargar el modelo completo
model = torch.load(MODEL_FILE)
model.eval()  # Establecer el modelo en modo de evaluación

# Cargar el scaler
scaler = joblib.load(SCALER_FILE)

# Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_data_from_api(data_list):
    """
    Preprocesa la lista de datos recibida de la API para ser adecuada para el modelo.

    :param data_list: Lista de diccionarios que contiene pv, grid_import y cet_cest_timestamp.
    :return: Un tensor de PyTorch listo para el modelo.
    """
    # Extraer valores de la lista de diccionarios
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

    # Convertir a un array de numpy
    features = np.array([grid_import_values, pv_values, hours, day_of_years, day_of_weeks]).T

    # Normalizar las características
    features_normalized = scaler.transform(features)

    # Convertir a un tensor de PyTorch
    features_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0).to(device)  # Añadir dimensión de batch

    return features_tensor

def make_prediction(model, data):
    """
    Realiza una predicción utilizando el modelo y los datos procesados.

    :param model: El modelo de PyTorch cargado.
    :param data: El tensor de datos procesados.
    :return: La predicción realizada por el modelo.
    """
    with torch.no_grad():
        prediction = model(data)
        return prediction.item()

@app.route('/predict_energy', methods=['POST'])
def predict_energy():
    """
    Endpoint para recibir los últimos 60 minutos de datos de pv, grid_import y timestamp y devolver una predicción.

    :return: Respuesta JSON con el resultado de la predicción.
    """
    try:
        data = request.json["data"]
        
        # Verificar que los datos tengan exactamente 60 puntos
        if len(data) != 60:
            return jsonify({"error": "Exactly 60 data points are required."}), 400

        # Preprocesar los datos
        processed_data = preprocess_data_from_api(data)

        # Hacer la predicción
        prediction = make_prediction(model, processed_data)

        # Interpretar el resultado de la predicción
        result = "The consumption will be lower than production in the next hour." if prediction > 0.5 else "The consumption will not be lower than production in the next hour."

        return jsonify({"prediction": result, "prediction_class": 1 if prediction > 0.5 else 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
