#---------------------------Import Packages------------------------------------
from functools import partial
from collections import namedtuple
import PIL
import PIL.ImageFont, PIL.ImageDraw
import numpy as np
import matplotlib.pylab as pl
import jax
import jax.numpy as jp

from utils.video import VideoWriter, np2pil

#---------------------------JAX Utils------------------------------------

# Vectorizing map. Creates a function which maps fun over argument axes.
def vmap2(f):
  return jax.vmap(jax.vmap(f))

#---------------------------Coexistence Fields------------------------------------

Fields = namedtuple('Fields', 'U G R E')

def peak_f(x, mu, sigma):
  return jp.exp(-((x-mu)/sigma)**2)

def fields_f(p, points, x):
  r = jp.sqrt(jp.square(x-points).sum(-1).clip(1e-10))
  U = peak_f(r, p.mu_k, p.sigma_k).sum()*p.w_k
  G = peak_f(U, p.mu_g, p.sigma_g)
  R = p.c_rep/2 * ((1.0-r).clip(0.0)**2).sum()
  return Fields(U, G, R, E=R-G)

# visualization utils functions for lenia simulation
def lerp(x, a, b):
  return jp.float32(a)*(1.0-x) + jp.float32(b)*x
def cmap_e(e):
  return 1.0-jp.stack([e, -e], -1).clip(0) @ jp.float32([[0.3,1,1], [1,0.3,1]])
def cmap_ug(u, g):
  vis = lerp(u[...,None], [0.1,0.1,0.3], [0.2,0.7,1.0])
  return lerp(g[...,None], vis, [1.17,0.91,0.13])

@partial(jax.jit, static_argnames=['w', 'show_UG', 'show_cmap'])
def show_lenia(params, points, extent, w=400, show_UG=False, show_cmap=True):
  xy = jp.mgrid[-1:1:w*1j, -1:1:w*1j].T*extent
  e0 = -peak_f(0.0, params.mu_g, params.sigma_g)
  f = partial(fields_f, params, points)
  fields = vmap2(f)(xy)
  r2 = jp.square(xy[...,None,:]-points).sum(-1).min(-1)
  points_mask = (r2/0.02).clip(0, 1.0)[...,None]
  vis = cmap_e(fields.E-e0) * points_mask
  if show_cmap:
    e_mean = jax.vmap(f)(points).E.mean()
    bar = np.r_[0.5:-0.5:w*1j]
    bar = cmap_e(bar) * (1.0-peak_f(bar, e_mean-e0, 0.005)[:,None])
    vis = jp.hstack([vis, bar[:,None].repeat(16, 1)])
  if show_UG:
    vis_u = cmap_ug(fields.U, fields.G)*points_mask
    if show_cmap:
      u = np.r_[1:0:w*1j]
      bar = cmap_ug(u, peak_f(u, params.mu_g, params.sigma_g))
      bar = bar[:,None].repeat(16, 1)
      vis_u = jp.hstack([bar, vis_u])
    vis = jp.hstack([vis_u, vis])
  return vis

fontpath = pl.matplotlib.get_data_path()+'/fonts/ttf/DejaVuSansMono.ttf'
pil_font = PIL.ImageFont.truetype(fontpath, size=16)

# get text on top of image for writing titles
def text_overlay(img, text, pos=(20,10), color=(255,255,255)):
  img = np2pil(img)
  draw = PIL.ImageDraw.Draw(img)
  draw.text(pos, text, fill=color, font=pil_font)
  return img

def animate_lenia(params, tracks, rate=10, slow_start=0, w=400, show_UG=True,
                  name='lenia_field.mp4', text=None, vid=None, bar_len=None,
                  bar_ofs=0, extent=None):
  if vid is None:
    vid = VideoWriter(fps=60, filename=name)
  if extent is None:
    extent = jp.abs(tracks).max()*1.2
  if bar_len is None:
    bar_len = len(tracks)
  for i, points in enumerate(tracks):
    if not (i<slow_start or i%rate==0):
      continue
    img = show_lenia(params, points, extent, w=w, show_UG=show_UG)
    bar = np.linspace(0, bar_len, img.shape[1])
    bar = (0.5+(bar>=i+bar_ofs)[:,None]*jp.ones(3)*0.5)[None].repeat(2, 0)
    frame = jp.vstack([img, bar])
    if text is not None:
      frame = text_overlay(frame, text)
    vid(frame)
  return vid