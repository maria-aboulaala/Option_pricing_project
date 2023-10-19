# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
from .base import OptionPricingModel


class BinomialTreeModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using BOPM (Binomial Option Pricing Model).
    It caclulates option prices in discrete time (lattice based), in specified number of time points between date of valuation and exercise date.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):

        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps

    def _calculate_call_option_price(self): 
        """Calculates price for call option according to the Binomial formula."""
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(S_T - self.K, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]

    def _calculate_put_option_price(self): 
        """Calculates price for put option according to the Binomial formula."""  
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(self.K - S_T, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]


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
