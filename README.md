# Energy Consumption and Production Prediction with OLAS Agents

This project predicts whether the energy consumption of a household will be lower than its production in the next hour using a Transformer-based machine learning model. The project also includes a real-time monitoring script and a Flask server to provide data endpoints.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Real-Time Monitor](#running-the-real-time-monitor)
  - [Running the Flask Server](#running-the-flask-server)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to:
1. Predict whether the energy consumption of a household will be lower than its production in the next hour using a Transformer-based model.
2. Provide real-time monitoring to make decisions on whether to turn on the air conditioning based on the prediction.
3. Serve energy consumption and production data through a Flask API.

## Dataset

The dataset used for this project can be found in this link: 
https://data.open-power-system-data.org/household_data/

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Clone the Repository

```sh
git clone https://github.com/keyko-io/olas-energy-management-agents.git
cd olas-energy-management-agents
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

## Usage
### Running the Real-Time Monitor
The real-time monitor script fetches data from an external API, saves it to a CSV file, preprocesses the data, and makes predictions every minute.

```sh
python main.py
```

### Running the Flask Server
The Flask server provides endpoints to fetch current and past (mock) energy data in case there is no third-party API server

```sh
python server.py
```
### API Endpoints
    `/energy-data`
    - Method: GET
    - Description: Fetches the current energy consumption and production data.
    - Response: JSON with the current data.
    
    `/past-data`
    - Method: GET
    - Description: Fetches the past 60 minutes of energy consumption and production data.
    - Response: JSON array with the past 60 minutes of data.

### Project Structure

```
olas-energy-management-agents/
│
├── main.py                  # Script for real-time monitoring and prediction
├── server.py                # Flask server script
├── requirements.txt         # Project dependencies
│
├── models/                  # Directory for saved models
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── transformer_model.keras
│
├── data/                    # Directory for storing data
│   └── real_time_data.csv
│
├── utils/                   # Directory for utility modules
│   ├── __init__.py
│   ├── data_preprocessing.py  # Module for data preprocessing functions
│   ├── model_utils.py         # Module for model-related functions
│   └── transformers.py        # Module for Transformer-related classes
│
└── logs/                    # Directory for storing log files (optional)
```

## Contributing
    1. Fork the repository.
    2. Create your feature branch (git checkout -b feature/AmazingFeature).
    3. Commit your changes (git commit -m 'Add some AmazingFeature').
    4. Push to the branch (git push origin feature/AmazingFeature).
    5. Open a Pull Request.
    
## License

This project is licensed under the MIT License - see the LICENSE file for details.
