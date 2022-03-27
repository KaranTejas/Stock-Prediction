from datetime import date
from pydoc import classname
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

 


# ------------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            [
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div([
                    # stock code input
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
                    html.Button(id='submit-button',  children='Stock Price'),
                    # Indicators button
                    html.Button(id='indicator-button',  children='Indicators'),
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
                ], id = 'present-graphs'),
                
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
    ]
)
def update_graphs(stock_code, start_date, end_date):
    ticker = yf.Ticker(stock_code)
    
    df = ticker.history(period='max')
    df['Date'] = df.index
    df['Year'] = df.index.year

    print(stock_code)
    print(start_date, end_date, end=' ')
    fig = px.line(df, x='Date', y='Close', range_x=[
                start_date, end_date], title='Price of ' + stock_code, width=570, height=400)

    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig2 = px.scatter(df,  x='Date', y='EWA_20', title="Exponential Moving Average vs Date", range_x=[start_date, end_date], width=570, height=400)
        
    fig3 = px.line()

    return fig, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True)
