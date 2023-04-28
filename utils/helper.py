#---------------------------Import Packages------------------------------------

import json
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pylab as pl
import scipy.signal as sl
import scipy

#---------------------------helper funcs------------------------------------

def load_from_json(filename:str) -> dict:
    """Load parameters from a .json file. Allows restoration of previous simulation states.

    Args:
        filename (str): The filename to load.

    Returns:
        dict: The state dictionary
    """
    # Deserialization
    print("Reading JSON file {} ...".format(filename))

    with open(filename, "r") as read_file:
        d_load = json.load(read_file)

        d_load["cells"] = np.asarray(d_load["cells"]) # Deserialise arrays

    print("Completed !")

    return d_load

def grab_plot():
    """Return the current Matplotlib figure as an image"""
    fig = pl.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    a = np.float32(img[..., 3:]/255.0)
    img = np.uint8(255*(1.0-a) + img[...,:3] * a)  # alpha
    pl.close()
    return img

def center(x, SIZE):
    """find the center of the creature in the cell"""
    CoM = ((np.mgrid[0:SIZE:1, 0:SIZE:1] * x).sum(axis=(1,2))/np.sum(x)).astype(np.int32)
    shift = np.array([SIZE//2, SIZE//2]) - CoM
    return np.roll(x, shift, axis=(0, 1))

def soft_clip(x, vmin, vmax):
  """helper function to replace np.clip function to make the growth of lenia smoother"""
  return 1 / (1 + np.exp(-4 * (x - 0.5))) 

def bell(x,m,s):
  """orginal growth function for lenia

    Args:
      x: The location
      m: mean value
      s: standard deviation

    Returns:
        growth value for the cell x
  """
  return np.exp(-((x-m)/s)**2 / 2)

def gaussian(U, m, s, A=None):
  """modify the growth function to make it centered between -1 and 1"""
  return bell(U, m, s)*2-1

def sigmoid(x):
  """helper func for flow lenia"""
  return 0.5 * (np.tanh(x / 2) + 1)

def target(U, m, s, A=None):
  """target function for asymptotic growth implementation of lenia """
  return bell(U, m, s) - A

def fft_convolve2d(A,K):
    """
    2D convolution, using FFT
    """
    fK = fft2(fftshift(K))
    fA = fft2(A)
    return np.real(ifft2(fK * fA))

def fft_convolve2d_scipy(A,K):
    """
    2D convolution, using FFT
    """
    fK = scipy.fft.fft2(scipy.fft.fftshift(K))
    fA = scipy.fft.fft2(A)
    return np.real(scipy.fft.ifft2(fK * fA))


def fft_multi_convolve2d(A,nKs):
    """
    2D FFT with multiple kernels
    """
    fKs = [fft2(fftshift(K)) for K in nKs]
    A = fft2(A)
    return [ np.real(ifft2(fK * A)) for fK in fKs ]

def fft_multi_convolve2d_scipy(A,nKs):
    """
    2D convolution, using FFT
    """
    fKs = [scipy.fft.fft2(scipy.fft.fftshift(K)) for K in nKs]
    fA = scipy.fft.fft2(A)
    return [ np.real(scipy.fft.ifft2(fK * fA)) for fK in fKs ] 

def fft_channel_convolve2d(As,nKs):
    """
    2D FFT with multiple channels
    """
    fAs = [ fft2(A) for A in As]
    fKs = [fft2(fftshift(K)) for K in nKs]
    A = fft2(A)
    return [ np.real(ifft2(fK * A)) for fK in fKs ]

def fft_channel_convolve2d_scipy(As,nKs):
    """
    2D convolution, using FFT
    """
    fKs = [ scipy.fft.fft2(scipy.fft.fftshift(K)) for K in nKs]
    fA = [scipy.fft.fft2(A) for A in As]
    return [ np.real(scipy.fft.ifft2(fK * fA)) for fK in fKs ]

kx = np.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
])
ky = np.transpose(kx)

def sobel_x(A):
  """
  A : (x, y, c)
  ret : (x, y, c)
  """
  return np.dstack([sl.convolve2d(A[:, :, c], kx, mode = 'same') 
                    for c in range(A.shape[-1])])
def sobel_y(A):
  return np.dstack([sl.convolve2d(A[:, :, c], ky, mode = 'same') 
                    for c in range(A.shape[-1])])
  
def sobel(A):
  """sobel filter function refer to https://en.wikipedia.org/wiki/Sobel_operator"""
  return np.concatenate((sobel_y(A)[:, :, None, :], sobel_x(A)[:, :, None, :]), axis = 2)