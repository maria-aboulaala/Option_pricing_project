# Standart python imports
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
# Third party imports
import streamlit as st
import yfinance as yf
# Local package imports
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

# @st.cache
# def get_historical_data(ticker):
#     """Getting historical data for speified ticker and caching it with streamlit app."""

#     return Ticker.get_historical_data(ticker)

# Ignore the Streamlit warning for using st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main title
st.title('Option pricing')
st.sidebar.image('./R.jpg')
# User selected model from sidebar 
pricing_method = st.sidebar.radio('Choisir une methode de pricng', options=[model.value for model in OPTION_PRICING_MODEL])
# st.sidebar.image('\R.jpg')

# Displaying specified model
st.subheader(f'Pricing method: {pricing_method}')
start_date = datetime(2020, 1, 1)  # Date de début en 2020 (par exemple, 1er janvier 2020)
end_date = datetime.now()  # Date de fin (date actuelle)

if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    # Parameters for Black-Scholes model
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    
    if st.button(f'Calculate option price for {ticker}'):
        # Getting data for selected ticker
        data = yf.download(ticker, start = start_date, end= end_date)
        data = pd.DataFrame(data)
        st.write(data.tail())
        st.line_chart(data['Adj Close'])

        # Formating selected model parameters
        spot_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        # Calculating option price
        BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
        call_option_price = BSM.calculate_option_price('Call Option')
        put_option_price = BSM.calculate_option_price('Put Option')

        # Displaying call/put option price
        st.write('Call option price:')
        st.write(call_option_price)
        st.write('Put option price:')
        st.write(put_option_price)


elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    # Parameters for Monte Carlo simulation
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)

    if st.button(f'Calculate option price for {ticker}'):
        # Getting data for selected ticker
        data = get_historical_data(ticker)
        data = pd.DataFrame(data)
        st.write(data.tail())
        # Ticker.plot_data(data, ticker, 'Adj Close')
        st.line_chart(data['Adj Close'])

        # Formating simulation parameters
        spot_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        # ESimulating stock movements
        MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
        MC.simulate_prices()

        # Visualizing Monte Carlo Simulation
        MC.plot_simulation_results(num_of_movements)
        st.pyplot()

        # Calculating call/put option price
        call_option_price = MC.calculate_option_price('Call Option')
        put_option_price = MC.calculate_option_price('Put Option')

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')

elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    # Parameters for Binomial-Tree model
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_time_steps = st.slider('Number of time steps', 5000, 100000, 15000)

    if st.button(f'Calculate option price for {ticker}'):
         # Getting data for selected ticker
        data = get_historical_data(ticker)
        data = pd.DataFrame(data)
        st.write(data.tail())
        Ticker.plot_data(data, ticker, 'Adj Close')
        st.pyplot()

        # Formating simulation parameters
        spot_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        # Calculating option price
        BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
        call_option_price = BOPM.calculate_option_price('Call Option')
        put_option_price = BOPM.calculate_option_price('Put Option')

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')