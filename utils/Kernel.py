#---------------------------Import Packages------------------------------------

import numpy as np
from .helper import sigmoid

#---------------------------Kernels------------------------------------

class Kernel:
    """
    Class for the kernel used during convolution update of each timestep of Lenia
    Create a variety of kernels:
     - Square kernel
     - (Interpolated) circle kernel
     - (Interpolated) ring kernel
     - Gaussian smoothed ring kernel
     - Multiple ring kernel
    """
    def __init__(self):
        self.kernel = self.square_kernel(3,1)
        
    def square_kernel(self, 
                      outer_diameter:int, 
                      inner_diameter:int,
                      ) -> np.array:
        """
        Create a square kernel for Moore neighbourhood, or extended Moore neighbourhood calculation
        
        e.g. 3,1 ->
            111
            101
            111
            
        e.g. 5,3 ->
            11111
            10001
            10001
            10001
            11111

        Args:
            outer_diameter (int): The outer diameter of the kernel ones (equal to the kernel size)
            inner_diameter (int): The inner diameter of the kernel zeros

        Returns:
            np.array: The resulting kernel
        """
        # Check that both diameters are either odd or even, else kernel is asymmetric
        if not ((outer_diameter % 2 == 0) and (inner_diameter % 2 == 0) or (outer_diameter % 2 == 1) and (inner_diameter % 2 == 1)): # both even
            print('ERROR: Use both odd or both even dimensions to ensure kernel symmetry')
            return None
        if outer_diameter <= inner_diameter:
            print('ERROR: Outer diameter (= {}) must be greater than inner (= {})'.format(outer_diameter,inner_diameter))
            return None

        inner = np.pad(np.ones([inner_diameter,inner_diameter]),(outer_diameter-inner_diameter) // 2)
        outer = np.ones([outer_diameter,outer_diameter])

        return outer - inner
    
    def circular_kernel(self, 
                        diameter:int, 
                        invert:bool=False,
                        ) -> np.array:
        """
        Create an interpolated circle kernel. 
        Used by self.ring_kernel
        
        e.g. 5 ->
        
        01110
        11111
        11111
        11111
        01110
        
        e.g. 7 ->
        
        0011100
        0111110
        1111111
        1111111
        0111110
        0011100

        Args:
            diameter (int): The outer diameter of the kernel (equal to the kernel size)
            invert (bool, optional): Whether to inver the values. Defaults to False.

        Returns:
            np.array: The resulting kernel
        """
        

        mid = (diameter - 1) / 2
        distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
        kernel = ((np.linalg.norm(distances, axis=0) - diameter/2) <= 0).astype(int)
        if invert:
            return np.logical_not(kernel).astype(int)
        
        return kernel

    def ring_kernel(self, 
                    outer_diameter:int, 
                    inner_diameter:int
                    ) -> np.array:
        """
        Create a binary, interpolated ring-like kernel. 
        Removes orthogonal bias, allowing isotropic patterns to form.

        Args:
            outer_diameter (int): The outer diameter of the kernel ones (equal to the kernel size).
            inner_diameter (int): The inner diameter of the kernel zeros.

        Returns:
            np.array: The resulting kernel
        """
        if not ((outer_diameter % 2 == 0) and (inner_diameter % 2 == 0) or (outer_diameter % 2 == 1) and (inner_diameter % 2 == 1)):
            print('ERROR: Use both odd or both even dimensions to ensure kernel symmetry')
            return None

        inner = np.pad(self.circular_kernel(inner_diameter),(outer_diameter-inner_diameter) // 2)
        outer = self.circular_kernel(outer_diameter)

        return outer - inner
    
    def smooth_ring_kernel(self,
                           mid, 
                           R,
                           mu:float=0.5, 
                           sigma:float=0.15
                           ) -> np.array:
        """
        Generate a smooth ring kernel by applying a bell-shaped (Gaussian) function to the kernel.
        Used by kernel_shell

        Args:
            mid (int): The outer radius of the kernel ones (equal to the kernel size).
            mu (float, optional): The mean value for Gaussian smoothing. Defaults to 0.5.
            sigma (float, optional): The stdev value for Gaussian smoothing. Defaults to 0.15.

        Returns:
            np.array: The resulting kernel
        """
        gaussian = lambda x, m, s: np.exp(-( ((x-m)/s)**2 / 2 ))
        if mid!=R:
            D = np.linalg.norm(np.asarray(np.ogrid[-mid:mid, -mid:mid])) / R
        else:
            D = np.linalg.norm(np.asarray(np.ogrid[-mid:mid, -mid:mid])+1) / R
        
        return (D<1) * gaussian(D, mu, sigma)
    
    def kernel_shell(self, 
                     mid:int,
                     R:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     mu:float=0.5, 
                     sigma:float=0.15, 
                     ) -> np.array:
        """
        Extend the kernal to multiple smooth rings ('shells').
        The number of shells can be changed by changing the number of items in 'peaks'.
        Shells are created equidistantly from the centre to the diameter.
        This allows the evolution of more interesting and diverse creatures.

        Args:
            mid (int): The outer radius of the kernel ones (equal to the kernel size).
            peaks (np.array, optional): The amplitude of the peaks for the shells, from inner to outer. 
                Defaults to np.array([1/2, 2/3, 1]).
            mu (float, optional): The mean value for Gaussian smoothing. Defaults to 0.5.
            sigma (float, optional): The stdev value for Gaussian smoothing. Defaults to 0.15.
            a (float, optional): The pre-factor for gaussian smoothing. Defaults to 4.0.

        Returns:
            np.array: The resulting kernel
        """
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / R
        k = len(peaks)
        kr = k * D

        peak = peaks[np.minimum(np.floor(kr).astype(int), k-1)]
        gaussian = lambda x, m, s: np.exp(-( ((x-m)/s)**2 / 2 ))

        return (D<1) * gaussian(kr % 1, mu, sigma) * peak
    
    def multi_kernels(self,
                      mid:int,
                      R:int,
                      kernel,
                      channels:bool=False
                      ) -> list:
        """
        Extend the kernal to multiple channels.
        The shell radius can be changed by each b values of the default creature parameter

        Args:
            mid (int): The outer radius of the kernel ones (equal to the kernel size).
            peaks (np.array, optional): The amplitude of the peaks for the shells, from inner to outer. 
                Defaults to np.array([1/2, 2/3, 1]).
            mu (float, optional): The mean value for Gaussian smoothing. Defaults to 0.5.
            sigma (float, optional): The stdev value for Gaussian smoothing. Defaults to 0.15.
            a (float, optional): The pre-factor for gaussian smoothing. Defaults to 4.0.

        Returns:
            np.array: The resulting kernel
        """
        if channels:
            Ds = [ np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / R * len(k['b']) / k['r'] for k in kernel ]
        else:
            Ds = [ np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / R * len(k['b']) for k in kernel ]
        gaussian = lambda x, m, s: np.exp(-( ((x-m)/s)**2 / 2 ))
        Ks = [ (D<len(k['b'])) * np.asarray(k['b'])[np.minimum(D.astype(int),len(k['b'])-1)] * gaussian(D%1, 0.5, 0.15) for D,k in zip(Ds,kernel) ]

        return Ks
    

    def flow_kernels(self, 
                    mid:int, 
                    nb_k:int, 
                    params):
        """
        modified kernel function for flow generation
        
        Args:
        mid (int): The outer radius of the kernel ones (equal to the kernel size).
        nb_k: number of kernels
        params: contains all the parameters for lenia, such as mu and sigma

        Returns:
            np.array: The resulting kernel after fourier transformation
        """
        ker_f = lambda x, a, w, b : (b * np.exp( - (x[..., None] - a)**2 / w)).sum(-1)

        """Compute kernels and return a dic containing fft kernels, T and R"""
        kernels = params['kernels']

        Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
            ((params['R']+15) * kernels['r'][k]) for k in range(nb_k) ]  # (x,y,k)
        K = np.dstack([sigmoid(-(D-1)*10) * ker_f(D, kernels["a"][k], kernels["w"][k], kernels["b"][k]) 
                        for k, D in zip(range(nb_k), Ds)])
        nK = K / np.sum(K, axis=(0,1), keepdims=True)
        fK = np.fft.fft2(np.fft.fftshift(nK, axes=(0,1)), axes=(0,1))

        return fK
        