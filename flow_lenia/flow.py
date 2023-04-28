#---------------------------Import Packages------------------------------------
import numpy as np
import matplotlib.pylab as pl
import jax
import jax.numpy as jnp
from numpy.fft import fft2, ifft2
from functools import partial
import os

from utils.Growth_fn import Growth_fn
from utils.Kernel import Kernel
from utils.video import VideoWriter
from utils.helper import grab_plot, sobel
from utils.entropy import *

import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------ReintegrationTracking--------------------------------------------

"""advection of particles flow in btween cells for more information https://michaelmoroz.github.io/Reintegration-Tracking/"""
class ReintegrationTracking:

    def __init__(self, SX=256, SY=256, dt=.3, dd=5, sigma=.65):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        
        self.apply = self._build_apply()

    def __call__(self, *args):
        return self.apply(*args)

    def _build_apply(self):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes = (0, 0, None, None))
        def step_flow(X, mu, dx, dy):
            """Summary
            """
            Xr = jnp.roll(X, (dx, dy), axis = (0, 1))
            # Hr = np.roll(H, (dx, dy), axis = (0, 1)) #(x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis = (0, 1))

            dpmu = jnp.absolute(pos[..., None] - mur)

            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nX = Xr * area
            return nX

        def apply(X, F):

            mu = pos[..., None] + self.dt * F 
            nX= step_flow(X, mu, dxs, dys)


            # expnX = np.exp(nX.sum(axis = -1, keepdims = True)) - 1
            nX = jnp.sum(nX, axis = 0)
            # nH = np.sum(nH * expnX, axis = 0) / (expnX.sum(axis = 0)+1e-10) #avg rule

            return nX

        return apply

"""randomly generate lenia parameters for momentum&energy conservation demos"""
class Rule_space :
  #-----------------------------------------------------------------------------
  def __init__(self, nb_k, init_shape = (40, 40)):
    self.nb_k = nb_k
    self.init_shape = init_shape
    self.kernel_keys = 'r b w a m s h'.split()
    # rule grid 
    self.spaces = {
        "r" : {'low' : .2, 'high' : 1., 'mut_std' : .2, 'shape' : None},
        "b" : {'low' : .001, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
        "w" : {'low' : .01, 'high' : .5, 'mut_std' : .2, 'shape' : (3,)},
        "a" : {'low' : .0, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
        "m" : {'low' : .05, 'high' : .5, 'mut_std' : .2, 'shape' : None},
        "s" : {'low' : .001, 'high' : .18, 'mut_std' : .01, 'shape' : None},
        "h" : {'low' : .01, 'high' : 1., 'mut_std' : .2, 'shape' : None},
        'T' : {'low' : 10., 'high' : 50., 'mut_std' : .1, 'shape' : None},
        'R' : {'low' : 2., 'high' : 25., 'mut_std' : .2, 'shape' : None},
        'init' : {'low' : 0., 'high' : 1., 'mut_std' : .2, 'shape' : self.init_shape}
    }
  #-----------------------------------------------------------------------------
  def sample(self):
    np.random.seed(1337)
    kernels = {}
    for k in 'rmsh':
      kernels[k] = np.random.uniform(
          self.spaces[k]['low'], self.spaces[k]['high'], self.nb_k
      )
    for k in "awb":
      kernels[k] = np.random.uniform(
          self.spaces[k]['low'], self.spaces[k]['high'], (self.nb_k, 3)
      )
    return {
        'kernels' : kernels, 
        'T' : np.random.uniform(self.spaces['T']['low'], self.spaces['T']['high']),
        'R' : np.random.uniform(self.spaces['R']['low'], self.spaces['R']['high']),
        'init' : np.random.rand(*self.init_shape) 
    }

#---------------------------Flow Lenia------------------------------------

class Lenia_flow:

  def __init__(self, nb_k, size, C=1, dd=5, n=2, dt=0.3, sigma=0.65):
    self.sigma = sigma
    self.dt = dt
    # power for alpha weights
    self.n = n
    self.dd = dd
    # number of kernels
    self.nb_k = nb_k
    # number of channels
    self.C = C 
    self.c0 = [0] * nb_k
    self.c1 = [[i for i in range(nb_k)]]
    # sample random parameters
    self.params = Rule_space(nb_k).sample()
    self.creature_params = {k : self.params['kernels'][k] for k in self.params['kernels'].keys()}
    self.size= size
    x, y = np.arange(self.size), np.arange(self.size)
    X, Y = np.meshgrid(x, y)
    self.pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)
    # arrays for reintegration tracking
    rollxs = []
    rollys = []
    for dx in range(-self.dd, self.dd+1):
      for dy in range(-self.dd, self.dd+1):
        rollxs.append(dx)
        rollys.append(dy)
    self.rollxs = np.array(rollxs)
    self.rollys = np.array(rollys)

  # normal update without cosidering mass flow
  def step(self, x):
    kernel_FFT = Kernel().flow_kernels(x.shape[0]//2, self.nb_k, self.params)
    x_FFT = fft2(x, axes=[0,1])
    x_FFTk = x_FFT[:,:,self.c0]
    U = np.real(ifft2(kernel_FFT * x_FFTk, axes=(0,1)))
    G = Growth_fn(mu=self.creature_params['m'], sigma=self.creature_params['s']).growth_gaussian(U) * self.creature_params['h']  # (x,y,k)
    H = np.dstack([ G[:, :, self.c1[c]].sum(axis=-1) for c in range(self.C) ])  # (x,y,c)
    return np.clip(x + 1/self.params['T']*H, 0, 1) 
  
  # step update with mass conservation
  def step_flow(self, x):

    @partial(jax.vmap, in_axes = (0, 0, None, None))
    def flow(rollx, rolly, A, mus):
      rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
      dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (rollx, rolly), axis = (0, 1))) # (x, y, 2, c)
      sz = .5 - dpmu + self.sigma #(x, y, 2, c)
      area = jnp.prod(np.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
      nA = rollA * area
      return nA
    
    kernel_FFT = Kernel().flow_kernels(x.shape[0]//2, self.nb_k, self.params)
    x_FFT = fft2(x, axes=[0,1])
    x_FFTk = x_FFT[:,:,self.c0]
    U = jnp.real(ifft2(kernel_FFT * x_FFTk, axes=(0,1)))
    G = Growth_fn(mu=self.creature_params['m'], sigma=self.creature_params['s']).growth_gaussian(U) * self.creature_params['h']  # (x,y,k)
    H = jnp.dstack([ G[:, :, self.c1[c]].sum(axis=-1) for c in range(self.C) ])  # (x,y,c)

    #-------------------------------FLOW------------------------------------------
    # apply sobel filtre
    F = sobel(H) #(x, y, 2, c)
    # find the gradient
    C_grad = sobel(x.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)
    # Alpha weights the importance of term such that the concentration gradient dominates in high concentrations regions
    alpha = jnp.clip((x[:, :, None, :])**self.n, .0, 1.)
    
    F = F * (1 - alpha) - C_grad * alpha
    
    mus = self.pos[..., None] + self.dt * F #(x, y, 2, c) : target positions (distribution centers)

    nA = flow(self.rollxs, self.rollys, x, mus).sum(axis = 0)
    # nA = self.RT.apply(x,F)
    
    return nA

  # generate the world with mass conservation applied
  @staticmethod
  def flow_world(x, vmin=0, vmax=1, title_1='', title_2='', sep_x=None, alpha_1 = 1.0, alpha_2 = 1.0):
      SIZE = x.shape[0]
      if sep_x is None:
        sep_x = SIZE
      fig = pl.figure(figsize=(np.shape(x)[1]/80, np.shape(x)[0]/80), dpi=80)
      ax = fig.add_axes([0, 0, 1, 1])
      ax.grid(False)
      ax.text(sep_x//2, SIZE - 40, title_1, fontsize='xx-large', color='white', ha='center', va='center', alpha = alpha_1)
      ax.text(SIZE + sep_x//2, SIZE - 40, title_2, fontsize='xx-large', color='white', ha='center', va='center', alpha = alpha_2)
      ax.axvline(x=sep_x, linestyle='--', linewidth=4)
      img = ax.imshow(x, cmap='jet', interpolation='none', aspect=1, vmin=vmin, vmax=vmax)
      return grab_plot()

#---------------------------Main Simulations------------------------------------

def lenia_mass_flow(filename, entropy_file, conserved:False, record_entropy:False, SIZE=256):
  SIZE = SIZE
  # world = np.zeros((SIZE, SIZE, 1)) 
  lenia = Lenia_flow(20, SIZE)
  world = jnp.zeros((SIZE, SIZE, 1)) # consider one channel
  world = world.at[SIZE//2-20:SIZE//2+20, SIZE//2-20:SIZE//2+20, 0].set(lenia.params['init'])

  spatial_h0 = dict()
  spatial_h1 = dict()
  #world[128-20:128+20, 128-20:128+20, 0] = lenia.params['init']
  calculate_every = 30
  mass_log = []

  if conserved:
    with VideoWriter(filename,fps=15) as vid:
      for i in range(300):
        out = Lenia_flow.flow_world(world, title_1=None, title_2=None)
        # collect the creature's world entropy 
        if record_entropy and (i % calculate_every == 0 or i==300-1): # include the last iteration
          spatial_h0['generation_'+str(i)] = get_spatial_entropy(world.reshape((SIZE,SIZE)),window_size=24)
        for j in range(2):
          world = lenia.step_flow(world)
        mass_log.append(np.sum(world)/100)
        pl.figure(figsize=(SIZE/100, SIZE/100), dpi=100)
        pl.gcf().subplots_adjust(bottom=0.2, left=0.25, top=0.95, right=0.95)
        pl.plot(mass_log)
        pl.grid(False)
        pl.xlabel("step")
        pl.ylabel("total mass")
        pl.ylim(0, 15)
        pl_img = grab_plot()
        vid(np.concatenate((out, pl_img), axis=1))
    if record_entropy:
      np.savez(os.path.join('./generation_data', entropy_file), **spatial_h0)
  else:
    with VideoWriter(filename=filename,fps=15) as vid:
      for i in range(300):
        out = Lenia_flow.flow_world(world, title_1=None, title_2=None)
        if record_entropy and (i % calculate_every == 0 or i==300-1):
          spatial_h1['generation_'+str(i)] = get_spatial_entropy(world.reshape((SIZE,SIZE)),window_size=24)
        for j in range(2):
          world = lenia.step(world)
        mass_log.append(np.sum(world)/100)
        pl.figure(figsize=(SIZE/100, SIZE/100), dpi=100)
        pl.gcf().subplots_adjust(bottom=0.2, left=0.25, top=0.95, right=0.95)
        pl.plot(mass_log)
        pl.grid(False)
        pl.xlabel("step")
        pl.ylabel("total mass")
        pl.ylim(0, 100)
        pl_img = grab_plot()
        vid(np.concatenate((out, pl_img), axis=1))
    if record_entropy:
      np.savez(os.path.join('./generation_data', entropy_file), **spatial_h1)

