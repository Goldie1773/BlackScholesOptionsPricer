"""
Black-Scholes option pricer to calculate the price of a European call or put option using the Black-Scholes formula.

Inputs:
    S: stock price
    K: strike price
    r: risk-free rate
    q: dividend yield
    sigma: volatility
    T: time to expiration
    option_type: call or put

Outputs:
    price: option price
    
"""

import math
from scipy.stats import norm

def calc_call_price(S, K, r, q, sigma, T):
    """
    Calculate the price of a European call option using the Black-Scholes formula.
    
    p1 = S * e^(-q*T) * N(d1), where N(d1) is the cumulative distribution function of the standard normal distribution.
    
    p2 = K * e^(-r*T) * N(d2), where N(d2) is the cumulative distribution function of the standard normal distribution.
    
    price = p1 - p2
    
        args:
            S: stock price
            K: strike price
            r: risk-free rate
            q: dividend yield
            sigma: volatility
            T: time to expiration
    
        Output:
            price: call option price
    """
    d1 = calc_d1(S, K, r, q, sigma, T)
    d2 = calc_d2(d1, sigma, T)
    
    p1 = S * math.exp(-q*T) * norm.cdf(d1)
    p2 = K * math.exp(-r*T) * norm.cdf(d2)
    
    price = p1 - p2
    
    return price
    
def calc_put_price(S, K, r, q, sigma, T):
    """
    Calculate the price of a European put option using the Black-Scholes formula.
    
    p1 = K * e^(-r*T) * N(-d2), where N(-d2) is the cumulative distribution function of the standard normal distribution.
    
    p2 = S * e^(-q*T) * N(-d1), where N(-d1) is the cumulative distribution function of the standard normal distribution.
    
    price = p1 - p2
    
        args:
            S: stock price
            K: strike price
            r: risk-free rate
            q: dividend yield
            sigma: volatility
            T: time to expiration
    
        Output:
            price: put option price
    """
    d1 = calc_d1(S, K, r, q, sigma, T)
    d2 = calc_d2(d1, sigma, T)
    
    p1 = K * math.exp(-r*T) * norm.cdf(-d2)
    p2 = S * math.exp(-q*T) * norm.cdf(-d1)
    
    price = p1 - p2
    
    return price
    
def calc_d1(S, K, r, q, sigma, T):
    """
    Calculate d1 in the Black-Scholes formula.
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        d1: d1 value
    """
    
    nominator = math.log(S/K) + (r - q + sigma**2/2)*T
    denominator = sigma * math.sqrt(T)
    d1 = nominator / denominator
    return d1
    
    
def calc_d2(d1, sigma, T):
    """
    Calculate d2 in the Black-Scholes formula.
    
    args:
        d1: d1 value
        sigma: volatility
        T: time to expiration
    
    Output:
        d2: d2 value
    """
    d2 = d1 - sigma * math.sqrt(T)
    return d2


underlying_price = 100
call_strike_price = 90
put_strike_price = 110
volatility = 0.15
risk_free_rate = 0.05
dividend_yield = 0.03
time_to_expiration = 1/24

call_price = calc_call_price(underlying_price, call_strike_price, risk_free_rate, dividend_yield, volatility, time_to_expiration)
put_price = calc_put_price(underlying_price, put_strike_price, risk_free_rate, dividend_yield, volatility, time_to_expiration)

print(f"Call price: {call_price:.2f}")
print(f"Put price: {put_price:.2f}")