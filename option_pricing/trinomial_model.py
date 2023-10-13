# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
from .base import OptionPricingModel

import numpy as np
from .base import OptionPricingModel

class TrinomialTreeModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Trinomial Option Pricing Model.
    It calculates option prices in discrete time (lattice based) with a trinomial tree.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        """
        Initializes variables used in the Trinomial Option Pricing Model.

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option contract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_time_steps: number of time periods between the valuation date and exercise date
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps

    def _calculate_call_option_price(self):
        """Calculates the price for a call option using the Trinomial Option Pricing Model."""
        return V[0]

    def _calculate_put_option_price(self):
        """Calculates the price for a put option using the Trinomial Option Pricing Model."""
        return V[0]

