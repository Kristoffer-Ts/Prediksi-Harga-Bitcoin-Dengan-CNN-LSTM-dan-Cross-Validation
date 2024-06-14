import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8); /* Adjust the opacity here to control the brightness */
            z-index: 0;
        }}
        .stApp > div {{
            position: relative;
            z-index: 1;
        }}
        .custom {{
            position: relative;
            top: 70px
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to split data into features and target
def split_target(data, look_back=720):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(np.delete(data[i:(i + look_back)], 3, axis=1))
        y.append(data[i + look_back, 3])
    return np.array(X), np.array(y)

# Function to reverse scaling
def reverse_scaling(value, scaler, feature_index, input_features):
    dummy = np.zeros((len(value), len(input_features)))
    if len(value.shape) == 1:
        dummy[:, feature_index] = value
    else:
        dummy[:, feature_index] = value[:, 0]
    actual_value = scaler.inverse_transform(dummy)[:, feature_index]
    return actual_value

def predict_bitcoin_prices():
    data = pd.read_csv('btcpermenit05.csv')
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index(data['Datetime'])
    data1 = data.drop(['Datetime'], axis=1)
    minmax = MinMaxScaler()
    data1 = pd.DataFrame(minmax.fit_transform(data1.values), columns=data1.columns, index=data['Datetime'])
    
    x = data1['Close']
    cv = TimeSeriesSplit(n_splits=7)
    x = data1.values
    split_indices = list(cv.split(x))
    train_indices, test_indices = split_indices[-1]
    train, test = x[train_indices], x[test_indices]
    look_back = 720
    X_train, y_train = split_target(train, look_back)
    X_val, y_val = split_target(test, look_back)
    
    model = tf.keras.models.load_model('model_fold_7.h5')
    y_pred = model.predict(X_val)
    
    input_feature = data1.columns
    y_pred_orig = reverse_scaling(y_pred, minmax, input_feature.to_list().index('Close'), input_feature)
    y_val_orig = reverse_scaling(y_val, minmax, input_feature.to_list().index('Close'), input_feature)
    
    mse = mean_squared_error(y_val_orig, y_pred_orig)
    rmse = sqrt(mse)
    r2 = r2_score(y_val_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(y_val_orig, y_pred_orig)
    
    st.session_state['results'] = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_val_orig': y_val_orig,
        'y_pred_orig': y_pred_orig,
        'data1': data1.to_dict(),  # Convert DataFrame to dict to store in session state
        'model': 'model_fold_7.h5'  # Store model path
    }
    
    st.title('Hasil Prediksi Terbaik')
    st.write(f'Validation R2: {r2}')
    st.write(f'Validation MAPE: {mape}')
    
    plt.figure(figsize=(20, 8))
    plt.title("Bitcoin Price Predictions")
    plt.plot(y_val_orig, label='Real value')
    plt.plot(y_pred_orig, label='Predicted Value')
    plt.legend()
    st.pyplot(plt)

# Function to predict future prices
def future_prediction(df, model_path='model_fold_7.h5'):
    new_data = df
    new_data['Datetime'] = pd.to_datetime(new_data['Datetime'])
    data2 = new_data.set_index(new_data['Datetime'])
    data2 = new_data.drop(['Datetime', 'Unnamed: 0'], axis=1)
    scaler = MinMaxScaler()
    data2 = pd.DataFrame(scaler.fit_transform(data2.values), columns=data2.columns, index=new_data['Datetime'])

    train_scaled = np.array(data2)

    X = []
    y = []

    future = 1
    past = 720

    for i in range(past, len(train_scaled) - future+1):
        X.append(train_scaled[i - past:i])
        y.append(train_scaled[i + future - 1:i + future,0])

    X_train, y_train = np.array(X), np.array(y)

    model = tf.keras.models.load_model(model_path)

    y_pred = model.predict(X_train)
    y_pred_inverse = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], train_scaled.shape[1] - 1)))))[:, 0]
    
    st.title('Hasil Prediksi Masa Depan Dengan Data 2 Juni - 8 Juni')
    plt.figure(figsize=(20, 8))
    plt.title("Future Bitcoin Price Predictions")
    plt.plot(y_pred_inverse, label='Predicted Value')
    plt.legend()
    st.pyplot(plt)

# Main function to create the Streamlit app
def main():
    set_background('Background.jpg')
    # Check the query parameters to determine which page to display
    query_params = st.experimental_get_query_params()
    page = query_params.get('page', ['main'])[0]
    
    if page == 'main':
        st.title("Prediksi Bitcoin Prices")
        st.write("Bitcoin, sebagai mata uang kripto paling terkenal dan paling mapan, telah mengubah paradigma keuangan global sejak diperkenalkan pada tahun 2009 oleh seseorang atau kelompok yang menggunakan nama samaran Satoshi Nakamoto. Dengan dasar teknologi blockchain yang inovatif," 
                 "Bitcoin memungkinkan transaksi peer-to-peer tanpa melibatkan otoritas sentral, seperti bank atau pemerintah (Nakamoto, 2008)")
        st.image("bitcoin.png", width=100)  # Set your Bitcoin logo here
        st.write("### Profil Mahasiswa")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("FotoDiri.jpg", width=200)  # Set your profile picture here
            st.write("### Kristoffer Timoty Sugiarto")
        
        with col2:
            st.markdown('<p class="custom">NIM<br>11201049</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<p class="custom">Jurusan / Prodi<br>JMTI / Informatika</p>', unsafe_allow_html=True)
        
        st.write("---")
        st.write("## Upload CSV")
        future_uploaded_file = st.file_uploader("Choose a CSV file for Future Predictions", type="csv")
        
        if future_uploaded_file is not None:
            try:
                future_df = pd.read_csv(future_uploaded_file)
                st.session_state['future_uploaded_file'] = future_df
                st.write("Future Dataframe:")
                st.dataframe(future_df)
            except Exception as e:
                st.error(f"Error reading the future prediction file: {e}")

        if st.button("Predict Bitcoin Prices"):
            predict_bitcoin_prices()
            if 'future_uploaded_file' in st.session_state:
                future_df = st.session_state['future_uploaded_file']
                future_prediction(future_df)
            else:
                st.write("Please upload a CSV file for future predictions first.")
    elif page == 'results':
        display_results()
    
if __name__ == "__main__":
    main()

# # Load data
# data = pd.read_csv("btcpermenit05.csv")
# data['Datetime'] = pd.to_datetime(data['Datetime'])
# data = data.set_index(data['Datetime'])
# data1 = data.drop(['Datetime'], axis=1)


# # Scale data
# minmax = MinMaxScaler()
# data1 = pd.DataFrame(minmax.fit_transform(data1.values), columns=data1.columns, index=data['Datetime'])
# x = data1['Close']

# # Function to split data
# def split_target(data, look_back=720):
#     X, y = [], []
#     for i in range(len(data) - look_back):
#         X.append(np.delete(data[i:(i + look_back)], 3, axis=1))
#         y.append(data[i + look_back, 3])
#     return np.array(X), np.array(y)

# # Time series split
# cv = TimeSeriesSplit(n_splits=7)
# x = data1.values
# split_indices = list(cv.split(x))
# train_indices, test_indices = split_indices[-1] 
# train, test = x[train_indices], x[test_indices]

# # Split data into training and validation sets
# look_back = 720
# X_train, y_train = split_target(train, look_back)
# X_val, y_val = split_target(test, look_back)

# # Load model
# model = load_model('model_fold_7.h5')

# # Make predictions
# y_pred = model.predict(X_val)

# # Evaluate model
# result = model.evaluate(X_val, y_val, verbose=0)
# mse = result[2]
# rmse = sqrt(mse)
# r2 = r2_score(y_val, y_pred)
# mape = mean_absolute_percentage_error(y_val, y_pred)

# # Streamlit display
# st.title('Bitcoin Price Prediction Results')
# st.write(f'Validation MSE Score: {mse}')
# st.write(f'Validation RMSE Score: {rmse}')
# st.write(f'Validation R2 Score: {r2}')
# st.write(f'Validation MAPE Score: {mape}')

# # Plotting
# st.write('## Prediction vs Real Value')
# fig, ax = plt.subplots(figsize=(15, 5))
# ax.set_title("Hasil Prediksi model fold ke 7")
# ax.plot(y_val, label='Real value')
# ax.plot(y_pred, label='Prediction Value')
# ax.legend()
# st.pyplot(fig)
