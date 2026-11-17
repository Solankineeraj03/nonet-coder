"""Concept library: generic helpers available to synthesized programs.

This file is prepended verbatim to every generated program, so everything here
is part of the (non-neural) model. Keep it generic: no problem-specific logic.
"""
LIB_SRC = r'''
import math
from itertools import accumulate as _acc

def is_prime(n):
    if not isinstance(n, int) or n < 2: return False
    if n > 10**8: raise ValueError
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True

def digits(n):
    return [int(c) for c in str(abs(int(n)))]

def fib(n):
    if n > 2000: raise ValueError
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

def factorial(n):
    if n < 0 or n > 300: raise ValueError
    return math.factorial(n)

def divisors(n):
    if n < 1 or n > 10**6: raise ValueError
    return [d for d in range(1, n + 1) if n % d == 0]

def prime_factors(n):
    if n > 10**9: raise ValueError
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d); n //= d
        d += 1
    if n > 1: out.append(n)
    return out

def is_palindrome(s):
    return s == s[::-1]

def running(xs):
    return list(_acc(xs))

def safe_range(a, b=None):
    if b is None: a, b = 0, a
    if b - a > 10000: raise ValueError
    return list(range(a, b))
'''
