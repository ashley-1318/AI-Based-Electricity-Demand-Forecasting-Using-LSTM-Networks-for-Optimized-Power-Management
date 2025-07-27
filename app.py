import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Electricity Demand Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Electricity Demand Forecasting in Tamil Nadu (2024)")
st.write("This application demonstrates an LSTM model for forecasting electricity demand using historical data and weather features.")

# --- Data Upload Section ---
st.header("1. Upload Data")
st.write("Please upload the CSV file containing the electricity demand and weather data.")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None # Initialize df to None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

# --- Raw Data Preview Section ---
if df is not None:
    st.header("2. Raw Data Preview")
    st.write("Here is a preview of the raw data.")
    st.dataframe(df)

    # --- Load Curve Visualization Section ---
    st.header("3. Electricity Load Curve")
    st.write("Visualizing the electricity load over time.")
    # Ensure the 'datetime' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='datetime', y='tamilnadu_load_MW', ax=ax)
    ax.set_title("Electricity Load Curve")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Electricity Load (MW)")
    st.pyplot(fig)
    plt.close(fig)

    # --- Data Preprocessing Section ---
    st.header("4. Data Preprocessing")
    st.write("Preprocessing the data by selecting features, scaling, and creating sequences.")

    features = ['tamilnadu_load_MW', 'temperature_C', 'humidity_%', 'hour', 'dayofweek', 'month']
    features_df = df[features]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:(i + seq_length)]
            label = data[i + seq_length, 0]
            X.append(seq)
            y.append(label)
        return np.array(X), np.array(y)

    seq_length = 24
    X, y = create_sequences(scaled_features, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    st.success("Data preprocessing completed successfully!")
    st.write(f"Shape of training features (X_train): {X_train.shape}")
    st.write(f"Shape of training targets (y_train): {y_train.shape}")
    st.write(f"Shape of testing features (X_test): {X_test.shape}")
    st.write(f"Shape of testing targets (y_test): {y_test.shape}")


    # --- Model Training Section ---
    st.header("5. Model Training")
    st.write("Training the LSTM model. This may take some time.")
    if X_train.size > 0 and y_train.size > 0:
        with st.spinner('Training LSTM model...'):
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        st.success("LSTM model training completed successfully!")
    else:
         st.warning("Cannot train model. Preprocessed data is not available or is empty.")


    # --- Prediction and Evaluation Section ---
    st.header("6. Prediction and Evaluation")
    st.write("Making predictions on the test data and evaluating the model.")
    if 'model' in locals() and model is not None and X_test.size > 0 and y_test.size > 0:
        y_pred_scaled = model.predict(X_test)

        dummy_array_pred = np.zeros((len(y_pred_scaled), scaled_features.shape[1]))
        dummy_array_test = np.zeros((len(y_test), scaled_features.shape[1]))
        dummy_array_pred[:, 0] = y_pred_scaled.flatten()
        dummy_array_test[:, 0] = y_test.flatten()

        y_pred = scaler.inverse_transform(dummy_array_pred)[:, 0]
        y_actual = scaler.inverse_transform(dummy_array_test)[:, 0]

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        predicted_actual_df = pd.DataFrame({'Actual Load': y_actual, 'Predicted Load': y_pred})

    else:
        st.warning("Cannot make predictions. Trained model or test data not found.")

    # --- Prediction Visualization Section ---
    st.header("7. Actual vs. Predicted Load")
    st.write("Visualizing the actual and predicted electricity load on the test data.")
    if 'predicted_actual_df' in locals() and predicted_actual_df is not None and not predicted_actual_df.empty:
        peak_pred_value = predicted_actual_df['Predicted Load'].max()
        peak_pred_index = predicted_actual_df['Predicted Load'].idxmax()

        fig, ax = plt.subplots(figsize=(15, 7))
        sns.lineplot(data=predicted_actual_df, ax=ax, label='Actual Load')
        sns.lineplot(data=predicted_actual_df, x=predicted_actual_df.index, y='Predicted Load', ax=ax, label='Predicted Load')

        ax.axvline(x=peak_pred_index, color='red', linestyle='--', label='Peak Predicted Demand Index')
        ax.plot(peak_pred_index, peak_pred_value, marker='o', color='red', markersize=8)
        ax.annotate(f'Peak: {peak_pred_value:.2f} MW',
                    xy=(peak_pred_index, peak_pred_value),
                    xytext=(peak_pred_index + 50, peak_pred_value + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10,
                    color='red')

        ax.set_title("Actual vs. Predicted Electricity Load with Peak Highlight")
        ax.set_xlabel("Time Steps (Test Data)")
        ax.set_ylabel("Electricity Load (MW)")
        ax.legend(title='Load Type')

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Cannot plot predictions. Predicted vs. Actual data not found or empty.")

else:
    st.info("Please upload a CSV file to get started.")