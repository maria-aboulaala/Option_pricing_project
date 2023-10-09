import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
st.title('My title')










with st.form(key="my_form"):
    stock_name = st.selectbox(
    'Le symbole de stock',
    ('AAPL', 'MSFT', 'META', 'GOOG', 'AMZN'))
    nSim = st.number_input('Le nombre de simulation', step=1, min_value=1)
    K = st.number_input('le prix d exercice de l option')
    r = st.number_input('le taux d intérêt sans risque', min_value=0, max_value=1)
    
    st.form_submit_button("Simuler")

    # Initialise parameters
S0 = 100      # initial stock price
K = 100       # strike price
T = 1         # time to maturity in years
r = 0.06      # annual risk-free rate
N = 3         # number of time steps
u = 1.1       # up-factor in binomial models
d = 1/u       # ensure recombining tree
opttype = 'C' # Option Type 'C' or 'P'

def binomial_tree_fast(K,T,S0,r,N,u,d,opttype='C'):
    #precompute constants
    dt = T/N
    q = (np.exp(r*dt) - d) / (u-d)
    disc = np.exp(-r*dt)
    
    # initialise asset prices at maturity - Time step N
    C = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1)) 
    
    # initialise option values at maturity
    C = np.maximum( C - K , np.zeros(N+1) )
        
    # step backwards through tree
    for i in np.arange(N,0,-1):
        C = disc * ( q * C[1:i+1] + (1-q) * C[0:i] )
    st.write(C[0])
    return C[0]

binomial_tree_fast(K,T,S0,r,N,u,d,opttype='C')






