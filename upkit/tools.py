# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:06:01 2024

@author: mleon
"""
import numpy as np
import matplotlib.pyplot as plt

c = 30  # cm/ns

### Useful functions ###


def crystal_ball(x, A, mu, sigma, alpha, n):
    """ Vectorized Crystal Ball function for signal peaks with a radiative tail. """
    t = (x - mu) / sigma
    # Gaussian core
    gaussian_part = A * np.exp(-0.5 * t**2)
    # Power-law tail
    tail_part = A * ((n / abs(alpha)) ** n) * np.exp(-0.5 * alpha**2) * \
        ((n / abs(alpha) - abs(t)) ** -n)

    # Apply conditions using NumPy masking
    result = np.where(t > -alpha, gaussian_part, tail_part)
    return result


def novosibirsk(x, A, mu, sigma, Lambda):
    arg = 1 + Lambda * (x - mu)/sigma
    # valid_mask = arg > 0
    # log_term = np.zeros_like(x)
    log_term = np.log(arg)
    return A * np.exp(-0.5 * (log_term**2 / (2*Lambda**2)))


def plot_dt_P(x, a, b, c, d, e):
    return a*x + b*np.exp(-c*x)


def plot_M_beta(x, a, b, c, d, e):
    return a + b*np.exp(-c*x)+d*x+e*x**2


def beta_calculator(p, mass):
    return p / np.sqrt(p**2 + mass**2)


def mass_calculator(p, b):
    return p**2 * ((1 - b**2) / (b**2))


def beta_calc(x, m, n):
    return 700 / (700 / (x * c / np.sqrt(x**2 + m**2)) + 4 * n) / c


def dt_calc(x, m, n, t_st, t_tof):
    return t_tof - 720 / (beta_calc(x, m, n) * c) - t_st


def gauss_fit(x, Amp, mean, sigma):
    gauss = Amp * np.exp(-np.power((x - mean) / (2 * sigma), 2))
    return gauss


def gauss_fit2(x, Amp, mean, sigma, a, b):
    gauss = Amp * np.exp(-np.power((x - mean) / (2 * sigma), 2)) + b/(x-mean)**a
    return gauss


def poly2_fit(x, a, b, c):
    poly = a + b * x + c * np.power(x, 2)
    return poly


def poly3_fit(x, a, b, c, d):
    poly = a + b * x + c * np.power(x, 2) + d * np.power(x, 3)
    return poly


def poly4_fit(x, a, b, c, d, e):
    poly = a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power(x, 4)
    return poly


def poly6_fit(x, a, b, c, d, e, f, g):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6


def poly7_fit(x, a, b, c, d, e, f, g, h):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7


def poly10_fit(x, a, b, c, d, e, f, g, h, i, j, k):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9 + k*x**10


def poly_fit(x, a, b):
    poly = a + b * x
    return poly


def gauss_poly6_fit(x, Amp, mean, sigma, a, b, c, d, e, f, g):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly6_fit(x, a, b, c, d, e, f, g)
    return gauss + poly


def gauss_poly7_fit(x, Amp, mean, sigma, a, b, c, d, e, f, g, h):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly7_fit(x, a, b, c, d, e, f, g, h)
    return gauss + poly


def gauss_poly4_fit(x, Amp, mean, sigma, a, b, c, d, e):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly4_fit(x, a, b, c, d, e)
    return gauss + poly


def gauss_poly_fit(x, Amp, mean, sigma, a, b):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly_fit(x, a, b)
    return gauss + poly


def gauss_poly3_fit(x, Amp, mean, sigma, a, b, c, d):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly3_fit(x, a, b, c, d)
    return gauss + poly


def gauss_poly2_fit(x, Amp, mean, sigma, a, b, c):
    gauss = gauss_fit(x, Amp, mean, sigma)
    poly = poly2_fit(x, a, b, c)
    return gauss + poly


def get_view(x):
    x1, x2 = plt.xlim()
    sl = (x1 <= x) & (x <= x2)
    return sl


def circular_cut(x, y, radius, x_center, y_center):
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    inside_circle = np.where(distance <= radius)[0]
    outside_circle = np.where(distance > radius)[0]
    return inside_circle, outside_circle


def lorentz_fit(x, amp, mean, sigma):
    breit = amp * (sigma / (2 * np.pi * ((x - mean) ** 2 + (sigma / 2) ** 2)))
    return breit

def voigt_fit(x, amp, mean, sigma, gamma):
    """ Voigt profile function. """
    from scipy.special import voigt_profile
    return amp * voigt_profile(x - mean, sigma, gamma)


def lorentz_poly4_fit(x, amp, mean, sigma, a, b, c, d, e):
    breit = lorentz_fit(x, amp, mean, sigma)
    poly = poly4_fit(x, a, b, c, d, e)
    return breit + poly


def lorentz_poly2_fit(x, amp, mean, sigma, a, b, c):
    breit = lorentz_fit(x, amp, mean, sigma)
    poly = poly2_fit(x, a, b, c)
    return breit + poly


def gaussian_exp_tail(x, A, mu, sigma, B, C):
    gaussian = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    tail = B * np.exp(-C * (x - mu)) * (x > mu)  # Only applies for x > mu
    return gaussian + tail


def voigt_poly4_fit(x, amp, mean, sigma, gamma, a, b, c, d, e):
    """ Voigt profile with polynomial background. """
    from scipy.special import voigt_profile
    voigt = amp * voigt_profile(x - mean, sigma, gamma)
    poly = poly4_fit(x, a, b, c, d, e)
    return voigt + poly