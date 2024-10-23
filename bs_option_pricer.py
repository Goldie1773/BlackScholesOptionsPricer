import math
from scipy.stats import norm

def bsm_option_pricer(S, K, r, q, sigma, T):
    """
    Calculate the price of a European call or put option using the Black-Scholes-Merton formula.
    
        args:
            S: stock price
            K: strike price
            r: risk-free rate
            q: dividend yield
            sigma: volatility
            T: time to expiration
            
        Output:
            call_price: call option price
            put_price: put option price
    """

    d1 = calc_d1(S, K, r, q, sigma, T)
    d2 = calc_d2(d1, sigma, T)

    call_price = calc_call_price(S, K, r, q, T, d1, d2)
    call_delta = calc_delta_call(q, T, d1)
    call_theta = calc_theta_call(S, K, r, q, sigma, T, d1, d2)
    call_rho = calc_rho_call(K, r, T, d2)
    
    put_price = calc_put_price(S, K, r, q, T, d1, d2)
    put_delta = calc_delta_put(q, T, d1)
    put_theta = calc_theta_put(S, K, r, q, sigma, T, d1, d2)
    put_rho = calc_rho_put(K, r, T, d2)
    
    gamma = calc_gamma(S, q, sigma, T, d1)
    vega = calc_vega(S, q, T, d1)
    
    prices = {
        "call_price": call_price,
        "call_delta": call_delta,
        "call_theta": call_theta,
        "call_rho": call_rho,
        "put_price": put_price,
        "put_delta": put_delta,
        "put_theta": put_theta,
        "put_rho": put_rho,
        "gamma": gamma,
        "vega": vega
    }
    
    return prices


def calc_call_price(S, K, r, q, T, d1, d2):
    """
    Calculate the price of a European call option using the Black-Scholes-Merton formula.
    
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
    price = S * math.exp(-q*T) * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    return price
    
def calc_put_price(S, K, r, q, T, d1, d2):
    """
    Calculate the price of a European put option using the Black-Scholes-Merton formula.
    
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
    price = K * math.exp(-r*T) * norm.cdf(-d2) - S * math.exp(-q*T) * norm.cdf(-d1)
    return price
    
def calc_d1(S, K, r, q, sigma, T):
    """
    Calculate d1 in the Black-Scholes-Merton formula.
    
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
    Calculate d2 in the Black-Scholes-Merton formula.
    
    args:
        d1: d1 value
        sigma: volatility
        T: time to expiration
    
    Output:
        d2: d2 value
    """
    d2 = d1 - sigma * math.sqrt(T)
    return d2


def calc_delta_call(q, T, d1):
    """
    Calculate the delta of a European call option.
    
    delta = e^(-q*T) * N(d1)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        delta: delta value
    """
    delta = math.exp(-q*T) * norm.cdf(d1)
    return delta


def calc_delta_put(q, T, d1):
    """
    Calculate the delta of a European put option.
    
    delta = e^(-q*T) * (N(d1) - 1)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        delta: delta value
    """
    delta = math.exp(-q*T) * (norm.cdf(d1) - 1)
    return delta


def calc_theta_call(S, K, r, q, sigma, T, d1, d2):
    """
    Calculate the theta of a European call option.
    A negative theta means the option price decreases as time passes.
    
    theta = -S * e^(-q*T) * n(d1) * sigma / (2 * sqrt(T)) - r * K * e^(-r*T) * N(d2) + q * S * e^(-q*T) * N(d1)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        theta: theta value
    """
    theta = -S * math.exp(-q*T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * norm.cdf(d2) + q * S * math.exp(-q*T) * norm.cdf(d1)
    return theta

def calc_theta_put(S, K, r, q, sigma, T, d1, d2):
    """
    Calculate the theta of a European put option.
    A negative theta means the option price decreases as time passes.
    
    theta = -S * e^(-q*T) * n(d1) * sigma / (2 * sqrt(T)) + r * K * e^(-r*T) * N(-d2) - q * S * e^(-q*T) * N(-d1)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        theta: theta value
    """
    theta = -S * math.exp(-q*T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * norm.cdf(-d2) - q * S * math.exp(-q*T) * norm.cdf(-d1)
    return theta

def calc_rho_call(K, r, T, d2):
    """
    Calculate the rho of a European call option.
    
    rho = K * T * e^(-r*T) * N(d2)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        rho: rho value
    """
    rho = K * T * math.exp(-r*T) * norm.cdf(d2)
    return rho

def calc_rho_put(K, r, T, d2):
    """
    Calculate the rho of a European put option.
    
    rho = -K * T * e^(-r*T) * N(-d2)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        rho: rho value
    """
    rho = -K * T * math.exp(-r*T) * norm.cdf(-d2)
    return rho

def calc_gamma(S, q, sigma, T, d1):
    """
    Calculate the gamma of a European call or put option.
    
    gamma = e^(-q*T) * n(d1) / (S * sigma * sqrt(T))
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        gamma: gamma value
    """
    gamma = math.exp(-q*T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma
    
    
def calc_vega(S, q, T, d1):
    """
    Calculate the vega of a European call or put option.
    
    vega = S * e^(-q*T) * n(d1) * sqrt(T)
    
    args:
        S: stock price
        K: strike price
        r: risk-free rate
        q: dividend yield
        sigma: volatility
        T: time to expiration
    
    Output:
        vega: vega value
    """
    vega = S * math.exp(-q*T) * norm.pdf(d1) * math.sqrt(T)
    return vega   
    

underlying_price = 100
strike_price = 102.8
volatility = 0.15
risk_free_rate = 0.05
dividend_yield = 0.03
time_to_expiration = 1 # in years

option_prices = bsm_option_pricer(underlying_price, strike_price, risk_free_rate, dividend_yield, volatility, time_to_expiration)

print(f"Call option price: {option_prices['call_price']:.2f}, delta: {option_prices['call_delta']:.3f}, theta: {option_prices['call_theta']:.3f}, rho: {option_prices['call_rho']:.3f}")
print(f"Put option price: {option_prices['put_price']:.2f}, delta: {option_prices['put_delta']:.3f}, theta: {option_prices['put_theta']:.3f}, rho: {option_prices['put_rho']:.3f}")
print(f"Gamma: {option_prices['gamma']:.3f}, Vega: {option_prices['vega']:.3f}")



"""
To Do:
1. Add greeks calculation to the option pricer - completed
2. Add user interface to the option pricer using Streamlit
3. Add heatmap on the effect of volaility and the underlying price on the option price
4. Add purchase price to show PnL in the heatmap
"""