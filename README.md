# AI-Based Electricity Demand Forecasting for Tamil Nadu

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project provides a deep learning solution for forecasting electricity demand in Tamil Nadu, India. [cite_start]The primary objective is to accurately predict both baseline and peak electricity loads to help power distribution companies (DISCOMs) optimize power management, ensure grid stability, and reduce operational costs[cite: 5, 9]. [cite_start]The model leverages a Long Short-Term Memory (LSTM) neural network to capture complex temporal patterns from historical consumption data[cite: 7].

## 📋 Features

* **Time-Series Forecasting:** Predicts future electricity demand on an hourly or daily basis.
* [cite_start]**Peak Demand Prediction:** Identifies the expected timing and magnitude of peak load periods[cite: 53].
* [cite_start]**Hybrid Modeling:** Integrates historical load data with external factors like weather and public holidays to improve accuracy[cite: 43, 45, 46, 47].
* [cite_start]**Multi-Step Forecasting:** Capable of predicting demand for future time intervals, such as the next 24 to 48 hours[cite: 51].

## 🧠 Methodology

[cite_start]The core of this solution is a hybrid AI model architecture[cite: 32]:

1.  **LSTM Layers:** One or more layers of Long Short-Term Memory (LSTM) cells process the sequential historical electricity demand data. [cite_start]This captures underlying temporal patterns like daily cycles, weekly trends, and seasonality[cite: 34, 35].
2.  **Feed-Forward Layers:** These layers process non-sequential exogenous variables, such as:
    * [cite_start]**Weather Data:** Temperature, humidity, wind speed, etc.[cite: 46].
    * [cite_start]**Calendar Information:** Public holidays and special events in Tamil Nadu[cite: 47].
3.  [cite_start]**Dense Output Layer:** The outputs from the LSTM and feed-forward layers are combined and passed to a dense layer to produce the final demand forecast[cite: 49].

## 📊 Dataset

The model requires the following datasets for training:

* [cite_start]**Historical Electricity Load Data:** System-wide demand (MW) for a region in Tamil Nadu, recorded at regular intervals (ideally 15 or 60 minutes)[cite: 59, 61].
* [cite_start]**Historical Weather Data:** Corresponding weather data (temperature, humidity, etc.) for the same period and location[cite: 62, 64].
* [cite_start]**Temporal Data:** Timestamps and a calendar indicating public holidays specific to Tamil Nadu[cite: 66, 68].

## ⚙️ Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow
* Pandas
* Scikit-learn
* Jupyter Notebook (optional, for exploration)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/electricity-forecasting.git](https://github.com/your-username/electricity-forecasting.git)
    cd electricity-forecasting
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Data Preprocessing:**
    Place your raw data files in the `data/raw/` directory. Run the preprocessing script to clean, synchronize, and feature-engineer the data.
    ```bash
    python src/data_preprocessing.py
    ```

2.  **Model Training:**
    Train the LSTM model on the preprocessed data.
    ```bash
    python src/train.py
    ```

3.  **Generate Forecasts:**
    Use the trained model to generate future demand forecasts.
    ```bash
    python src/forecast.py --horizon 48
    ```

## 📁 Project Structure
## 🔮 Future Scope

Potential enhancements for this project include:
* [cite_start]Integrating more granular data from consumer-level smart meters[cite: 140].
* [cite_start]Incorporating real-time weather and event data for more responsive forecasts[cite: 141].
* [cite_start]Extending the model to provide probabilistic forecasts for risk assessment[cite: 144].
* [cite_start]Deploying lightweight versions of the model on edge devices within the power grid[cite: 145].

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
