import streamlit as st
import datetime
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import pandas as pd
import numpy as np

import math


def main():
    START = "2015-01-01"
    TODAY = date.today().strftime('%Y-%m-%d')

    st.title("Stock Forecast App")
   
    st.markdown(html_temp, unsafe_allow_html=True)

    stocks = ('GOOG', 'AAPL', 'MSFT', 'FB','CSCO','QCOM','SBUX','TSLA')
    selected_stock = st.selectbox("Select dataset for prediction: ", stocks)
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    data_load_State = st.text("Loading data...")

    @st.cache(allow_output_mutation=True)
    def loading_data(ticker):
        df1 = yf.download(ticker, START, TODAY)
        df1.reset_index(inplace=True)
        return df1
    df1 = loading_data(selected_stock)
    df1["Date"]=pd.to_datetime(df1['Date']).dt.date
    data_load_State.text("Loading data.......Done")

    st.subheader("Raw data")
    st.write(df1.tail())
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1['Date'], y=df1["Open"], name="Stock_Open"))
        fig.add_trace(go.Scatter(x=df1['Date'], y=df1["Close"], name="Stock_Close"))
        fig.layout.update(title_text="Time Series data with Rangeslider", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()



    df_train = df1[["Date","Close"]]
    df_train =df_train.rename(columns={"Date":"ds","Close":"y"})
    model = Prophet()
    model.fit(df_train)
    pred = model.make_future_dataframe(periods=period)
    forecast = model.predict(pred)

    st.write("Forecast data")
    def plot_final_data():
        fig1 = plot_plotly(model,forecast)
        st.plotly_chart(fig1)
    plot_final_data()

    st.write("Forecast Componet")
    fig2 = model.plot_components(forecast)
    st.write(fig2)


# driver code
if __name__ == '__main__':
    main()

