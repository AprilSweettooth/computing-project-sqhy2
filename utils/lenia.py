#---------------------------Import Packages------------------------------------

from collections import namedtuple
import scipy.signal
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pylab as pl
from numpy.fft import fft2, ifft2

from .simulation import Simulator, show_videofile
from .Growth_fn import Growth_fn
from .Kernel import Kernel
from .Cells import Cells
from .video import VideoWriter
from .helper import grab_plot, gaussian, center
from QCA.Semi_CA import *

import warnings
warnings.filterwarnings("ignore")


#---------------------------Original Lenia------------------------------------

class Lenia_origin(namedtuple('Lenia', 'R, mu, sigma, size, dt, cells, b')):

  #rescale the creature to a lager/smaller size
  def rescale(p, x, n=4):
    x = np.repeat(np.repeat(np.array(x), n, axis=0), n, axis=1)
    p = p._replace(R = p.R * n)
    return p, x

  #update function on each iteration
  def step(p, x):
    kernel = Kernel().kernel_shell(128, p.R, np.asarray(p.b))
    kernel_FFT = fft2(kernel / np.sum(kernel))
    x_FFT = fft2(x)
    U = np.roll(np.real(ifft2(kernel_FFT * x_FFT)), 128, (0, 1))
    G = gaussian(U, p.mu, p.sigma)
    return np.clip(x + p.dt*G, 0, 1) 

  """
  here we put two creature in one world and experimenting
  for example, we put a barrier inbetween orbium and rotator,
  after remove the barrier to the orbium's side, the rotator now
  live in the orbium's world, with different mu and sigma
  """
  @staticmethod
  def coexist_world(x, vmin=0, vmax=1, title_1='', title_2='', sep_x=None, alpha_1 = 1.0, alpha_2 = 1.0):
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

  @staticmethod
  def get_creatures(p):
    if isinstance(p, list):
      p1=p[0]; p2=p[1]
      p1 = Lenia_origin(R=p1.R, mu=p1.mu, sigma=p1.sigma, size=p1.size, dt=p1.dt, cells=p1.cells, b=p1.b)
      p2 = Lenia_origin(R=p2.R, mu=p2.mu, sigma=p2.sigma, size=p2.size, dt=p2.dt, cells=p2.cells, b=p2.b)
      c1 = p1.cells
      c2 = p2.cells
      p1, c1 = p1.rescale(c1, n=6)
      p2, c2 = p2.rescale(c2, n=6)
      return (p1, c1), (p2, c2)
    else:
      p = Lenia_origin(R=p.R, mu=p.mu, sigma=p.sigma, size=p.size, dt=p.dt, cells=p.cells, b=p.b)
      c = p.cells
      return (p, c)

#---------------------------Main Simulations------------------------------------

def lenia_origin_diagram(params, creature, kernel, plot_kernel, demo=False, cx=32, cy=32, scale=1, clip='np',show_on_finish=True,lenia='classical'):
    SIZE=params.size
    # get lenia's parameters
    (p, c)= Lenia_origin.get_creatures(params)
    title = "{}'s World\n$\mu={}, \sigma={}$".format(creature.split("_")[0], p.mu, p.sigma)
    # create an empty world
    x = Cells(initialisation_type='zeros', grid_size=SIZE).cell
  
    # paste creature
    c = scipy.ndimage.zoom(c, scale, order=0)  
    cs = c.shape
    x[cx:cx + cs[0], cy:cy + cs[1]] = c
  
    growth_fn = Growth_fn(mu=p.mu, sigma=p.sigma)

    # calling main simulator
    sim = Simulator(cells=x,kernel=kernel,growth_fn=growth_fn,type=creature,title=title, demo=demo, dT=p.dt, cmap='viridis', lenia=lenia, clip=clip)
    if not demo: # demo only plot kernel
      # extract the execution time only
      sim.animate(frames=300)
      path = os.path.join('.' + '/demo/' + creature + '_size_' + str(SIZE) +'.mp4')
      sim.save_animation(os.path.join(creature+'_size_'+str(SIZE)+'.mp4'))
      if show_on_finish:
        show_videofile(path)
    # plot kernel info
    if plot_kernel:
       sim.plot_kernel_info(p.R)


def lenia_extended_diagram(creature, json, SIZE, plot_kernel, demo=False, cx=32, cy=32, scale=1, clip='np',show_on_finish=True):
    title = "{}'s World\nMulti-Parameters".format(creature.split("_")[0])
    c = json['cells']
    x = Cells(initialisation_type='zeros', grid_size=SIZE).cell
    
    # paste creature
    cs = c.shape
    c = scipy.ndimage.zoom(c, scale, order=0)  
    R = scale * json['R']
    x[cx:cx + cs[0], cy:cs[1] + cy] = c
    x = center(x, SIZE)
    # create parameter list
    ms = np.array([ k['m'] for k in json['kernels'] ])
    ss = np.array([ k['s'] for k in json['kernels'] ])
    hs = np.array([ k['h'] for k in json['kernels'] ])
    c0s = np.array([ k['c0'] for k in json['kernels'] ])
    c1s = np.array([ k['c1'] for k in json['kernels'] ])

    growth_fn = Growth_fn(ms=ms, ss=ss, hs=hs, c0s=c0s, c1s=c1s)
    kernels = Kernel().multi_kernels(SIZE//2, R, json['kernels'])

    sim = Simulator(cells=x,kernel=kernels,growth_fn=growth_fn,type=creature,title=title, dT=json['dt'], cmap='viridis', lenia='extended', clip=clip)
    if not demo:
      sim.animate(frames=300)
      path = os.path.join('.' + '/demo/' + creature + '_size_' +str(SIZE) +'.mp4')
      sim.save_animation(os.path.join(creature+'_size_'+str(SIZE)+'.mp4'))
      if show_on_finish:
        show_videofile(path)
    if plot_kernel:
       sim.plot_kernel_info_list(R, json['kernels'])


def lenia_multi_channel_diagram(creature, json, SIZE, plot_kernel, demo=False, cx=32, cy=32, scale=1, clip='np',show_on_finish=True):
    title = "{}'s World\nMulti-Parameters".format(creature.split("_")[0])
    xs = [ Cells(initialisation_type='zeros', grid_size=SIZE).cell for i in range(3) ]
    # paste creature
    Cs = json['cells']
    Cs = [ scipy.ndimage.zoom(np.asarray(c), scale, order=0) for c in Cs ]  
    R = scale * json['R']
    for x,c in zip(xs,Cs):  x[cx:cx+c.shape[0], cy:cy+c.shape[1]] = c

    ms = np.array([ k['m'] for k in json['kernels'] ])
    ss = np.array([ k['s'] for k in json['kernels'] ])
    hs = np.array([ k['h'] for k in json['kernels'] ])
    rs = np.array([ k['r'] for k in json['kernels'] ])
    c0s = np.array([ k['c0'] for k in json['kernels'] ])
    c1s = np.array([ k['c1'] for k in json['kernels'] ])

    growth_fn = Growth_fn(ms=ms, ss=ss, hs=hs, rs=rs, c0s=c0s, c1s=c1s)
    kernels = Kernel().multi_kernels(SIZE//2, R, json['kernels'], True)
    sim = Simulator(cells=xs,kernel=kernels,growth_fn=growth_fn,type=creature,title=title, dT=json['dt'], cmap='viridis', lenia='3_channel', clip=clip)
    if not demo:
      sim.animate(frames=300)
      path = os.path.join('.' + '/demo/' + creature + '_size_' + str(SIZE) + '.mp4')
      sim.save_animation(os.path.join(creature+'_size_'+str(SIZE)+'.mp4'))
      if show_on_finish:
        show_videofile(path)
    if plot_kernel:
       sim.plot_kernel_info_list(R=R, kernels=json['kernels'])

def lenia_coex_diagram(params, creature):
  SIZE=params[0].size
  (p1, c1), (p2, c2) = Lenia_origin.get_creatures(params)
  MID = int(SIZE / 2)
  titles = dict(title_1="{}'s World\n$\mu={}, \sigma={}$".format(creature[0].split("_")[0], params[0].mu, params[0].sigma),
                title_2="{}'s World\n$\mu={}, \sigma={}$".format(creature[1].split("_")[0], params[1].mu, params[1].sigma))

  x_1 = np.zeros((SIZE, SIZE), dtype=float)
  x_2 = np.zeros((SIZE, SIZE), dtype=float)

  def center(x):
    CoM = ((np.mgrid[0:SIZE:1, 0:SIZE:1] * x).sum(axis=(1,2))/np.sum(x)).astype(np.int32)
    shift = np.array([SIZE//2, SIZE//2]) - CoM
    return np.roll(x, shift, axis=(0, 1))

  # paste creature 1
  cs = c1.shape
  x_1[SIZE//2:SIZE//2 + cs[0], SIZE//2 - cs[1]:SIZE//2] = c1
  # x_1 = x_1.at[SIZE//2:SIZE//2 + cs[0], SIZE//2 - cs[1]:SIZE//2].set(c1)
  x_1 = center(x_1)

  # paste creature 2
  cs = c2.shape
  #x_2 = x_2.at[(SIZE - cs[0])//2:(SIZE + cs[0])//2, (SIZE - cs[1])//2:(SIZE + cs[1])//2].set(c2)
  x_2[(SIZE - cs[0])//2:(SIZE + cs[0])//2, (SIZE - cs[1])//2:(SIZE + cs[1])//2] = c2
  with VideoWriter(fps=15) as vid:
    for i in tqdm(range(100)):
      x = np.concatenate((x_1, x_2), axis=1)
      vid(Lenia_origin.coexist_world(x, **titles))
      for _ in range(2):
        x_1 = center(p1.step(x_1))
        x_2 = p2.step(x_2)
    print('creating lenia world completed')
    for i in tqdm(range(60)):
      # move seperator
      scale = 1 - i/60
      vid(Lenia_origin.coexist_world(x, sep_x=SIZE*scale, alpha_1=2.5*max(scale-0.6, 0), **titles))
    print('move separator completed')
    for i in tqdm(range(100)):
      x = np.concatenate((x_1, x_2), axis=1)
      vid(Lenia_origin.coexist_world(x, sep_x=0, alpha_1=0.0, **titles))
      for _ in range(2):
        x_1 = center(p2.step(x_1))
        x_2 = p2.step(x_2)
    print('coexistence stage completed')
    for i in tqdm(range(60)):
      # move seperator
      scale = i/60
      vid(Lenia_origin.coexist_world(x, sep_x=SIZE*2*scale, alpha_1=2*max(scale-0.5, 0), alpha_2=2*max(0.5-scale, 0)))
    print('move separator completed')
    for i in tqdm(range(100)):
      x = np.concatenate((x_1, x_2), axis=1)
      vid(Lenia_origin.coexist_world(x, sep_x=2*SIZE, alpha_2=0.0, **titles))
      for _ in range(2):
        x_1 = center(p1.step(x_1))
        x_2 = p1.step(x_2)
    print('simulation ompleted')


def lenia_mass_diagram(params):
  SIZE = params[0].size
  c = Cells(grid_size=SIZE//16, seed=0).cell.astype(np.float32)*1.8
  (p1, _), (p2, _) = Lenia_origin.get_creatures(params)
  _, c = p1.rescale(c, n=4)
  x = np.zeros((SIZE, SIZE), dtype=float)
  coords = (0, c.shape[0])
  x[coords[0]:coords[1], coords[0]:coords[1]] = c

  mass_log = []
  with VideoWriter('mass_simulation.mp4',fps=15) as vid:
    for i in range(120):
      out = Lenia_origin.coexist_world(x, title_1=None, title_2=None)
      for i in range(2):
        x = p2.step(x)
      mass_log.append(np.sum(x)/1000)
      pl.figure(figsize=(SIZE/100, SIZE/100), dpi=100)
      pl.gcf().subplots_adjust(bottom=0.2, left=0.25, top=0.95, right=0.95)
      pl.plot(mass_log)
      pl.grid(False)
      pl.xlabel("step")
      pl.ylabel("total mass")
      pl.ylim(0, 15)
      pl_img = grab_plot()
      vid(np.concatenate((out, pl_img), axis=1))