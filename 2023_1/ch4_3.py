import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# x0 = 3, x1 = 4일 때, x0에 대한 편미분을 구하라.
def function_tmp1(x0):
    return x0**2 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))  # 5.999999999998451