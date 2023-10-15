
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd

import streamlit as st
import yfinance as yf

from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker, TrinomialTreeModel 

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'
    TRINOMIAL = 'Trinomial Model'


# Ignore the Streamlit warning for using st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main title
st.title(':pound: Option pricing :dollar:')
# User selected model from sidebar 
pricing_method = st.sidebar.radio('Choisir une methode de pricng', options=[model.value for model in OPTION_PRICING_MODEL])


# Displaying specified model
st.subheader(f'	 La methode de Pricing : {pricing_method}')
start_date = datetime(2020, 1, 1)  # Date de début en 2020 (par exemple, 1er janvier 2020)
end_date = datetime.now()  # Date de fin (date actuelle)

if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    st.subheader('Introduction:')
   
    
    # Parameters for Black-Scholes model
    ticker = st.selectbox("Choisir un stock", ["AAPL", "GOOG"])
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    
    if st.button(f'Calculate option price for {ticker}'):

        # Getting data for selected ticker
        data = yf.download(ticker, start = start_date, end= end_date)
        data = pd.DataFrame(data)

        st.subheader(f':1234: Tableau de données de {ticker}')
        st.write(data.tail())

        st.subheader(f':chart: Graphique de {ticker}')
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

        st.subheader(f' :straight_ruler: Greeks {ticker}')


   




elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    # Parameters for Monte Carlo simulation
    ticker = st.selectbox("Pick one", ["AAPL", "GOOG"])
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)

    if st.button(f'Calculate option price for {ticker}'):
        # Getting data for selected ticker
        data = yf.download(ticker, start = start_date, end= end_date)
        data = pd.DataFrame(data)
        st.subheader(f':1234: Tableau de données de {ticker}')
        st.write(data.tail())
        st.subheader(f':chart: Graphique de {ticker}')
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
        data = yf.download(ticker, start = start_date, end= end_date)
        data = pd.DataFrame(data)
        st.write(data.tail())
        Ticker.plot_data(data, ticker, 'Adj Close')
        st.pyplot()

        # Formating simulation parameters
        spot_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        st.latex(r'''
    a +
    ''')

        # Calculating option price
        BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
        call_option_price = BOPM.calculate_option_price('Call Option')
        put_option_price = BOPM.calculate_option_price('Put Option')

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')

elif pricing_method == OPTION_PRICING_MODEL.TRINOMIAL.value:
    # Parameters for Binomial-Tree model
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_time_steps = st.slider('Number of time steps', 5000, 100000, 15000)

    if st.button(f'Calculate option price for {ticker}'):
         # Getting data for selected ticker
        data = yf.download(ticker, start = start_date, end= end_date)
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
        BOPM = TrinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
        call_option_price = BOPM.calculate_option_price('Call Option')
        put_option_price = BOPM.calculate_option_price('Put Option')

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')