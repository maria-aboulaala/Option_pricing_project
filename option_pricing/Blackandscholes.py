# Third party imports
import numpy as np
from scipy.stats import norm 
import matplotlib as plt
from datetime import datetime, timedelta
import yfinance as yf
# Local package imports
from .base import OptionPricingModel


class BlackScholesModel(OptionPricingModel):

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):

        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma

    def _calculate_call_option_price(self): 
        """
        Calculates price for call option according to the formula.        
        Formula: S*N(d1) - PresentValue(K)*N(d2)
        """
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)     
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        return (self.S * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0))
    

    def _calculate_put_option_price(self): 
        """
        Calculates price for put option according to the formula.        
        Formula: PresentValue(K)*N(-d2) - S*N(-d1)
        """  
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)    
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0))
    
    def _calculate_greeks(self):
        
    # Calcul des grecques
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

    # Delta (Δ) pour l'option d'achat
        delta_call = norm.cdf(d1)

    # Delta (Δ) pour l'option de vente
        delta_put = delta_call - 1.0

    # Gamma (Γ) commun aux options d'achat et de vente
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    # Theta (θ) pour l'option d'achat
        theta_call = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    # Theta (θ) pour l'option de vente
        theta_put = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

    # Vega (ν) commun aux options d'achat et de vente
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)

    # Rho (ρ) pour l'option d'achat
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)

    # Rho (ρ) pour l'option de vente
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)


        return {
            "Delta Call": delta_call,
            "Delta Put": delta_put,
            "Gamma": gamma,
            "Theta Call": theta_call,
            "Theta Put": theta_put,
            "Vega": vega,
            "Rho Call": rho_call,
            "Rho Put": rho_put
         }
    



 
    
    # # ticker = st.selectbox("Choisir un stock", ["AAPL", "GOOG"])
    # # strike_price = st.number_input('Strike price', 300)
   
   

    #     # Calculate delta for a call option at different times
    #     t = np.linspace(0, exercise_date, len(S))
    #     d1 = (np.log(S / strike_price) + (risk_free_rate + 0.5 * sigma ** 2) * (exercise_date - t)) / (sigma * np.sqrt(exercise_date - t))
    #     call_delta = norm.cdf(d1)

    #     # Plot the delta vs. time
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(t, call_delta, label='Call Option Delta', color='blue')
    #     plt.xlabel('Time to Expiration (Years)')
    #     plt.ylabel('Delta (Δ)')
    #     plt.title('Call Option Delta vs. Time to Expiration')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

