# Energy Management Agent with OLAS

## Introduction

This Proof of Concept (PoC) leverages IoT, AI, and autonomous agents to manage energy consumption in a prosumer household. The system monitors energy consumption (Grid import) and production (Solar panel production) through the Combinder API and utilizes Transformers AI models to predict whether energy production will exceed consumption in the next 60 minutes. If the prediction indicates excess production, the agent will autonomously control a physical device (e.g., Air Conditioning/Heat Pump) to optimize energy usage and maintain comfortable household conditions.

## Tech Stack
- **Olas Agents**: Autonomous agents that govern household energy management.
- **Combinder API**: Data source for energy consumption and production metrics, as well as control interface for household devices.
- **Nevermined Gateway**: A proxy service that manages API access and payments/subscriptions via Nevermined's decentralized infrastructure.
- **PEAQ blockchain**: A DePIN network that logs IoT interactions and facilitates smart contract operations through Nevermined on the PEAQ blockchain.
- **Transformers AI model**: A deep learning model trained to predict energy production and consumption dynamics based on time series data.
- **Flask server**: A lightweight server to handle API requests, model predictions, and interaction with the autonomous agent.

## Olas Agents

Each household is governed by an autonomous agent/service that follows a Finite State Machine (FSM) flow to manage energy consumption. The agent continuously monitors the household's energy dynamics and interacts with physical devices based on AI model predictions.

![plot](./images/fsm_diagram.png)

## Combinder API

The Combinder API is used to collect real-time energy consumption and production data for a household. It also provides an interface to control household devices. Each household is identified by a unique ID, and the API enables seamless integration between the autonomous agent and the physical infrastructure.

### Key API Interactions:
- **Data Collection**: Fetch real-time data on grid import and solar production.
- **Device Control**: Send commands to devices (e.g., Air Conditioning) based on the agent's decisions.

## Nevermined Gateway

The Nevermined Gateway is employed to manage access to the Combinder API, ensuring secure and decentralized payments and subscriptions. Nevermined wraps the API access within its proxy, allowing for transparent and accountable interactions logged on the blockchain.

### Integration Points:
- **Payment Management**: All API access is mediated through Nevermined, enabling decentralized payment and subscription models.
- **Secure Access**: The gateway ensures that all interactions with the Combinder API are secure and verifiable.

## PEAQ blockchain

The PEAQ blockchain is used to log IoT interactions and manage smart contracts for this project. Specifically, it tracks interactions facilitated by the Nevermined Gateway and ensures that all data and transactions are immutable and transparent.

### Blockchain Use Cases:
- **IoT Access Logging**: Every interaction with the Combinder API is recorded on the PEAQ blockchain.
- **Smart Contracts**: Smart contracts deployed on PEAQ handle the management of subscriptions, payments, and device control logic.

## Transformers AI model

The core of this project is a Transformer-based deep learning model trained to predict whether solar energy production will exceed grid energy consumption in the next 60 minutes. The model was trained on time series data from a public dataset provided by [Open Power System Data](https://data.open-power-system-data.org/household_data/)

### Model Details:
- **Architecture**: The model is based on the [Transformer architecture](https://arxiv.org/abs/1706.03762), known for its effectiveness in handling sequential data. The architecture includes self-attention mechanisms that allow the model to focus on relevant parts of the time series data.
Features: The model takes input features such as grid import, solar production, and temporal data (e.g., hour of the day, day of the year, day of the week).
- **Training**: The model was trained using the PyTorch framework. The training process included data normalization, time-based feature engineering, and the application of a custom Time2Vector layer for capturing temporal patterns.
- **Performance**: The model was evaluated on a holdout test set, achieving high accuracy in predicting whether energy production would exceed consumption. The fine-tuned model is now capable of making real-time predictions in production.

You can explore the training process in detail [here](https://github.com/keyko-io/olas-energy-management-agents/blob/main/PoC/notebooks/TRANSFORMER%20-%20Energy%20Consumption%20-%20PyTorch.ipynb).

### Inference Pipeline:
- **Data Preprocessing**: Incoming data from the Combinder API is preprocessed to match the format used during training. This includes normalizing the input features and applying temporal encoding.
- **Prediction**: The preprocessed data is passed through the Transformer model to generate a prediction.
Actionable Insights: The model's output is used by the Olas Agent to make decisions about device control, optimizing energy use in real-time.

## Flask Server
The Flask server acts as the backbone of the system, managing API requests and coordinating interactions between the AI model, the autonomous agent, and the physical devices.

### Endpoints:
- **/energy-data**: Provides the current energy consumption and production data.
- **/past-data**: Returns energy data from the past 60 minutes.
- **/predict-energy**: Receives the last 60 minutes of energy data and returns a prediction on whether production will exceed consumption in the next hour.

The server ensures that all components of the system work seamlessly together, enabling real-time energy management and optimization.

## System requirements

- Python `>=3.8`
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`
- [IPFS node](https://docs.ipfs.io/install/command-line/#official-distributions) `==0.6.0`
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Poetry](https://python-poetry.org/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

Alternatively, you can fetch this docker image with the relevant requirements satisfied:

> **_NOTE:_**  Tendermint and IPFS dependencies are missing from the image at the moment.

```bash
docker pull valory/open-autonomy-user:latest
docker container run -it valory/open-autonomy-user:latest
```

## This repository contains:
 ```bash
 TODO
 ```

## How to use

Create a virtual environment with all development dependencies:

```bash
poetry shell
poetry install
```