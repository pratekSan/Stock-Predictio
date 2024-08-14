import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews

# page_bg_img = """
# <style>
# /* Main Page Background */
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://i.imgur.com/4Nj5WcN.jpg");
#     background-size: cover; /* Cover the entire page */
#     background-position: center; /* Center the background image */
#     background-repeat: no-repeat; /* Prevent repeating */
#     background-attachment: fixed; /* Fix background image in place */
#     min-height: 100vh; /* Ensure it covers the full viewport height */
# }

# [data-testid="stHeader"] {
#     background: rgba(0,0,0,0); /* Make header transparent */
# }

# [data-testid="stToolbar"] {
#     right: 2rem; /* Adjust toolbar position */
# }
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

# st.markdown(
#     """
#     <style>
#     /* Sidebar Background */
#     [data-testid="stSidebar"] {
#         background-image: url('https://i.imgur.com/LWeCp2M.jpg');
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         padding: 2rem; /* Adjust padding as needed */
#     }
#
#     /* Additional Styling */
#     .stButton > button {
#         background-color: #4CAF50; /* Example button color */
#         color: white; /* Example button text color */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Streamlit components
st.title('Stock Trend Prediction')
st.write("Welcome to the stock trend prediction app!")

# Sidebar inputs
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
st.sidebar.markdown("### Select Dates to Start Prediction")

# Fetch data from Yahoo Finance
df = yf.download(user_input, start=start_date, end=end_date)

# Tabs for different sections
tabs = st.tabs(["Current Trend", "Insights", "Predictions"])

# Current Trend Tab
with tabs[0]:
    st.header('Current Trend')
    st.subheader('Closing Price vs Time')
    fig = px.line(df, x=df.index, y='Adj Close', title=user_input)
    st.plotly_chart(fig)
    
    # Additional Information
    st.subheader('Additional Information')
    st.write('Data Summary:')
    st.write(df.describe())

    st.subheader('Rolling Averages')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig_ma = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], 'b', label='Closing Price')
    plt.plot(ma100, 'g', label='100-Day MA')
    plt.plot(ma200, 'r', label='200-Day MA')
    plt.legend()
    plt.title('Rolling Averages')
    st.pyplot(fig_ma)

# Insights Tab
with tabs[1]:
    st.header('Insights')
    
    # Pricing Data
    st.subheader('Pricing Data')
    df2 = df.copy()
    df2['%Change'] = df['Adj Close'].pct_change() * 100
    df2.dropna(inplace=True)
    st.write(df2)
    annual_return = df2['%Change'].mean() * 252
    st.write(f'Annual Return: {annual_return:.2f}%')
    stdev = np.std(df2['%Change']) * np.sqrt(252)
    st.write(f'Standard Deviation: {stdev:.2f}%')
    risk_return = annual_return / stdev
    st.write(f'Risk Return Ratio: {risk_return:.2f}')

    # Fundamental Data
    st.subheader('Fundamental Data')
    key = 'CSZSZ88PNYQYA0BK'
    fd = FundamentalData(key, output_format='pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(user_input)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    
    st.write('Income Statement')
    income_statement = fd.get_income_statement_annual(user_input)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    
    st.write('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(user_input)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

    # News Data
    st.subheader('News')
    sn = StockNews(user_input, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news.get('sentiment_title', [None])[i]
        st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news.get('sentiment_summary', [None])[i]
        st.write(f'News Sentiment: {news_sentiment}')

# Predictions Tab
# Predictions Tab
with tabs[2]:
    st.header('LSTM Model Predictions')

    # Data Preparation for Prediction
    data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_test = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_array = scaler.fit_transform(data_train)

    model = load_model('LSTMpredict.h5')

    # Prepare Test Data
    past_100_days = data_train.tail(100)
    final_test_df = pd.concat([past_100_days, data_test], ignore_index=True)
    input_test_data = scaler.transform(final_test_df)

    x_test = []
    y_test = []
    for i in range(100, input_test_data.shape[0]):
        x_test.append(input_test_data[i - 100:i])
        y_test.append(input_test_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_predicted = model.predict(x_test)

    # Rescaling Values
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]

    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plot Predictions
    st.subheader('Predictions vs Original Values')
    fig_pred = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('LSTM Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.gca().set_xticks([])
    st.pyplot(fig_pred)

    # Performance Metrics
    st.subheader('Model Performance')

    # Calculate RMSE and MAE
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    mae = mean_absolute_error(y_test, y_predicted)

    # Display metrics
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

    # Additional Insights
    st.subheader('Additional Insights')

    # Suggested Next Steps
    st.write('**Suggested Next Steps:**')
    st.write(
        '1. **Retrain Model:** Consider retraining the model with more recent data to capture the latest market trends.')
    st.write(
        '2. **Hyperparameter Tuning:** Experiment with different hyperparameters for the LSTM model to improve its performance.')
    st.write(
        '3. **Model Validation:** Validate the model on other stock tickers to ensure its robustness and generalization.')

    # Model Limitations
    st.write('**Model Limitations:**')
    st.write(
        '1. **Historical Data Dependency:** The model relies on historical data, which may not always accurately predict future trends.')
    st.write(
        '2. **Market Volatility:** Stock markets are inherently volatile, and predictions may be affected by sudden market changes.')
    st.write(
        '3. **Feature Engineering:** Additional features such as technical indicators or sentiment analysis could improve the model.')
