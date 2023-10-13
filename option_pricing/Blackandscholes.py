# Third party imports
import numpy as np
from scipy.stats import norm 

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