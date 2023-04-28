#---------------------------Import Packages------------------------------------

import numpy as np
from .helper import gaussian, target

#---------------------------Growth------------------------------------

class Growth_fn:
    """Class for the growth function which is used to update the board based on the neighbourhood sum.
    This replaces the traditional conditional update used in Conway's game of life and can be generalised to any
    continous function. 
    
    f(x,y,t+1) = g(k*f(x,y,t))
    
    where g is the growth function
    k is the update kernel 
    f(x,y,t) is the board state at time t
    N.b. The operator * is the convolution operator 
    
    It consists of growth and shrink parts, which act on the neighbourhood sum to update the board at each timestep.
    """
    def __init__(self, mu=0.15, sigma=0.015, ms=np.array([1]), ss=np.array([1]), hs=np.array([1]), rs=np.array([1]), c0s=np.array([1]), c1s=np.array([1])):
        
        # Values for Gaussian update rule (default orbium)
        self.mu = mu
        self.sigma = sigma
        self.ms = ms
        self.ss = ss
        self.hs = hs
        self.rs = rs
        self.c0s = c0s
        self.c1s = c1s
    
    def growth_conway(self, A:np.array, U:np.array, increment:False) -> np.array:
        """Conditinal update rule for Conway's game of life
        b1..b2 is birth range, s1..s2 is stable range (outside s1..s2 is the shrink range) 

        with increment the update condition is put in a growth function which is more like
        continuous update

        Args:
            U (np.array): The neighbourhood sum 

        Returns:
            np.array: The updated board at time t = t+1
        """
        if increment:
            return 0 + (U==3) - ((U<2)|(U>3))
        else:
            return (A & (U==2)) | (U==3)

    def growth_gaussian(self, U:np.array) -> np.array:
        """Use a smooth Gaussian growth function to update the lenia world, based on the neighbourhood sum.
        This is the function used by Lenia to achive smooth, fluid-like patterns.

        Args:
            U (np.array): The neighbourhood sum 

        Returns:
            np.array: The updated board at time t = t+1
        """
        # gaussian = lambda x, m, s: np.exp(-( ((x-m)/s)**2 / 2 ))
        return gaussian(U, self.mu, self.sigma)
    
    def multi_growth(self, Us:list, As:list) -> list:
        """
        same function as aboev with gaussian updates but suitable for multi-channels creatures
        """
        funcs = [ gaussian, gaussian, target ]
        return [ gaussian(U, m, s) for U,m,s in zip(Us, self.ms, self.ss) ]
        # return  [ funcs[c1](U, m, s, As[c1]) for U,m,s,c1 in zip(Us, self.ms, self.ss, self.c1s) ]
    
    def coex_growth(self, U:np.array):
        """
        for demonstrating two different lenia types in one world
        this growth function uses approximation to make the growth smoother
        """
        return np.maximum(0, 1 - (U - self.mu)**2 / (self.sigma**2 * 9) )**4 * 2 - 1