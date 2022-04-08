from datetime import date
from pydoc import classname
from tracemalloc import start
import dash
from dash import html, dcc
import matplotlib
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import yfinance as yf
import matplotlib.pyplot as plt
from plotly import tools
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
import keras
from keras import backend as K
# import tensorflow



# -----------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv(
#     "https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro_bees.csv")
# df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[
#     ['Pct of Colonies Impacted']].mean()
# df.reset_index(inplace=True)
# print(df[:5])

ticker = yf.Ticker('AAPL')
df = ticker.history(period='max')
df['Date'] = df.index
df['Year'] = df.index.year
# df.to_csv('app.csv');
# df = pd.read_csv('app.csv')

# model = pickle.load(open('./Models/IBM', 'rb'))

# pred = []
# test_predict = df['Close']
# prediction_data = [i[0] for i in test_predict[::-1][:100][::-1]]
# i = 0
# newScaler = MinMaxScaler(feature_range=(0, 1))

# while i < 10:

#   new_prediction_data = newScaler.fit_transform(np.array(prediction_data).reshape(-1, 1))
#   predicted_value = newScaler.inverse_transform(model.predict(np.array(new_prediction_data).reshape((1, 100, 1))))
#   pred.append(predicted_value[0][0])
#   # print(predicted_value)
#   prediction_data = prediction_data[1:]
#   prediction_data.append(predicted_value[0][0])

#   i = i+1

# print(pred);

K.clear_session()
savedModel = keras.models.load_model('Save_Model', compile=False)


def create_first_dataset(dataset, time_step=1, input_size=1):
    a = []
    for j in range(time_step):
        a.append(dataset[j:(j+input_size), 0])
    return array(a)


# ------------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            [
                html.P("Welcome to the IE Project EXPO!", className="start"),
                html.Div([
                    # stock code input
                    html.P("Stock Code: ", className="start"),
                    dcc.Input(id='stock-input', value='AAPL', type='text')
                ]),
                html.Div([
                    # Date range picker input
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=df['Date'].min(),
                        max_date_allowed=df['Date'].max(),
                        start_date=df['Date'].min(),
                        end_date=df['Date'].max()
                    )
                ]),
                html.Div([
                    # Stock price button
                    # html.Button(id='submit-button',  children='Stock Price'),
                    # Indicators button
                    # html.Button(id='indicator-button',  children='Indicators'),
                    # Number of days of forecast input
                    # add a linebreak
                    html.Br(),
                    dcc.Input(id='forecast-input', value=10, type='number'),
                    # Forecast button
                    html.Button(id='forecast-button',  children='Forecast')
                ]),
            ],
            className="inputs"),


        html.Div(
            [
                html.Div(
                    [  # Logo
                        # Company Name
                    ],
                    className="header"),
                html.Div(  # Description
                    id="description", className="decription_ticker"),

                html.Div([
                    html.Div([
                        dcc.Graph(id='price-graph')
                    ], id="graphs-content"),
                    html.Div([
                        # Indicator plot
                        dcc.Graph(id='indicator-graph')
                    ], id="main-content")
                ], id='present-graphs'),

                html.Div([
                    # Forecast plot
                    dcc.Graph(id='forecast-graph')
                ], id="forecast-content")
            ],
            className="content")
    ], className="container")


@app.callback(
    [
        Output(component_id='price-graph', component_property='figure'),
        Output(component_id='indicator-graph', component_property='figure'),
        Output(component_id='forecast-graph', component_property='figure')
    ],

    [
        Input(component_id='stock-input', component_property='value'),
        Input(component_id='date-picker-range',
              component_property='start_date'),
        Input(component_id='date-picker-range', component_property='end_date'),
        Input(component_id='forecast-input', component_property='value')
    ]
)
def update_graphs(stock_code, start_date, end_date, forecast_days):
    ticker = yf.Ticker(stock_code)

    df = ticker.history(period='max')
    df['Date'] = df.index
    df['Year'] = df.index.year

    

    print(stock_code)
    print(start_date, end_date, end=' ')
    fig = px.line(df, x='Date', y='Close', range_x=[
        start_date, end_date], title='Price of ' + stock_code, width=570, height=400)

    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig2 = px.scatter(df,  x='Date', y='EWA_20', title="Exponential Moving Average vs Date", range_x=[
                      start_date, end_date], width=570, height=400)

    

    ticker = yf.Ticker(stock_code)
    df = ticker.history(period='max')
    df1 = df['Close']
    df2 = pd.DataFrame(df1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))


    NumberOfFutureDays = forecast_days

    x_input = df1[-149:]
    temp_list = create_first_dataset(x_input, 100, 50)
    temp_list = temp_list.reshape(1, 100, 50)

    lst_output = []
    n_steps = 100
    i = 0

    temp = temp_list

    while(i < NumberOfFutureDays):
        yhat = savedModel.predict(temp, verbose=0)
        lst_output.extend(yhat.tolist())
        new_input = temp[0, -1, 1:]
        new_input = np.append(new_input, yhat)
        temp = temp[0, 1:]
        temp = np.append(temp, [new_input], axis=0)
        temp = temp.reshape(1, temp.shape[0], temp.shape[1])
        i = i+1

    print(lst_output)
    print(len(lst_output))

    future_p = [i[0] for i in scaler.inverse_transform(lst_output)]
    print(future_p)
    print(len(future_p))

    future_pred = pd.DataFrame(df2)
    for i in range(NumberOfFutureDays):
        future_pred.loc[future_pred.shape[0]] = [None]
    print(future_pred)
    print((len(df2)))
    abcd = future_pred[(len(df2)):]
    
    abcd['predictions'] = future_p

    fig3 = px.line(abcd.reset_index(), y='predictions', title="Price prediction for the stock: " + stock_code)
    print(future_pred.tail())

    return fig, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True)
