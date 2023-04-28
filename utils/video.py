#---------------------------Import Packages------------------------------------

import io
import time
import subprocess
import PIL
import PIL.ImageFont, PIL.ImageDraw
import numpy as np
from IPython.display import display, Image, clear_output
import ipywidgets as widgets
import os

from .simulation import show_videofile

#---------------------------Utils Video------------------------------------

"""helper functions for video generation"""
def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def imshow(a, fmt='jpeg', display=display):
  return display(Image(data=imencode(a, fmt)))

"""main class for video writing for better visualisation"""
class VideoWriter:
  def __init__(self, filename='two_creatures.mp4', fps=30.0, show_on_finish=True):
    self.ffmpeg = None
    self.filename = filename
    self.path = os.path.join('.' + '/demo/' + self.filename)
    self.fps = fps
    self.view = widgets.Output()
    self.last_preview_time = 0.0
    self.frame_count = 0
    self.show_on_finish = show_on_finish
    display(self.view)

  def add(self, img):
    img = np.asarray(img)
    h, w = img.shape[:2]
    if self.ffmpeg is None:
      self.ffmpeg = self._open(w, h)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.ffmpeg.stdin.write(img.tobytes())
    t = time.time()
    self.frame_count += 1
    if self.view and t-self.last_preview_time > 1.0:
       self.last_preview_time = t
       with self.view:
         clear_output(wait=True)

  def __call__(self, img):
    return self.add(img)

  def _open(self, w, h):
    cmd = f'''ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
      -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p
      -c:v libx264 -crf 20 {self.path}'''.split()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

  def close(self):
    if self.ffmpeg:
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.ffmpeg = None
    if self.view:
      with self.view:
        clear_output()
      self.view.close()
      self.view = None

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.show_on_finish:
        self.show()

  def _ipython_display_(self):
    self.show()

  def show(self):
      self.close()
      show_videofile(self.path)