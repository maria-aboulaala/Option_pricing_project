# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
from .base import OptionPricingModel

import numpy as np
from scipy.stats import norm 

from .base import OptionPricingModel

class TrinomialTreeModel(OptionPricingModel):

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps

    def _calculate_call_option_price(self): 
        # Calcul du prix de l'option d'achat selon la formule trinomiale
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps
        u = np.exp(self.sigma * np.sqrt(3 * dT))
        d = 1 / u

        # Price vector initialization
        V = np.zeros(2 * self.number_of_time_steps + 1)
        S_T = np.zeros(2 * self.number_of_time_steps + 1)

        for j in range(2 * self.number_of_time_steps + 1):
            S_T[j] = self.S * (u ** (self.number_of_time_steps - j))

        a = np.exp(self.r * dT)
        p = (a - d) / (u - d)
        q = 1 - p

        V[:] = np.maximum(S_T - self.K, 0)

        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-2] = np.exp(-self.r * dT) * (p * V[1:-1] + q * V[:-2])
            V[-1] = 0

        return V[self.number_of_time_steps]

    def _calculate_put_option_price(self): 
        # Calcul du prix de l'option de vente selon la formule trinomiale
        dT = self.T / self.number_of_time_steps
        u = np.exp(self.sigma * np.sqrt(3 * dT))
        d = 1 / u

        V = np.zeros(2 * self.number_of_time_steps + 1)
        S_T = np.zeros(2 * self.number_of_time_steps + 1)

        for j in range(2 * self.number_of_time_steps + 1):
            S_T[j] = self.S * (u ** (self.number_of_time_steps - j))

        a = np.exp(self.r * dT)
        p = (a - d) / (u - d)
        q = 1 - p

        V[:] = np.maximum(self.K - S_T, 0)

        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-2] = np.exp(-self.r * dT) * (p * V[1:-1] + q * V[:-2])
            V[-1] = 0

        return V[self.number_of_time_steps]

    def _calculate_greeks(self):
        # Calcul des grecques
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1.0

        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

        theta_call = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        theta_put = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)

        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
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
