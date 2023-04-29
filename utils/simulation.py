#---------------------------Import Packages------------------------------------

import base64
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import scipy.signal
import os.path
import os
import multiprocessing
from numpy.fft import fft2, ifft2, fftshift
import timeit
import reikna.fft, reikna.cluda  
import pyfftw

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# import from internal functions
from .helper import *
from .Growth_fn import Growth_fn
from QCA.Semi_CA import *

# define storage path
MAX_FRAMES = 3000
OUTPUT_PATH = './demo'
NPZ_PATH = './generation_data'
EXEC_T_PATH = './exec_time'

#---------------------------Simulation------------------------------------

class Simulator(Growth_fn):

    def __init__(self, 
                 cells, 
                 kernel, 
                 growth_fn, 
                 type,
                 title,
                 demo:bool=False,
                 dT:float=0.1, 
                 cmap:str='viridis',
                 lenia:str='classical',
                 clip:str='np',
                 Adaptiveness:list=[0.001,0.001,0.2]
                 ):
        self.type = type.split("_")[0] # creature name
        # simulation method for example, fft means convolution with fast fourier transformation
        self.method = type.split("_")[1] 
        # title for simulation
        self.title = title
        # use demo to extract pre-stored data
        self.demo = demo
        # time interval for each update, for conway's GoL it is 1
        self.dT = dT
        self.cmap = cmap
        # original or lenia variant
        self.lenia = lenia
        # clip function, choice between numpy's clip and softclip
        self.clip = clip
        # Kernel paramaters
        self.kernel = kernel
        # Growth function parameters
        if self.type=='conway':
            self.growth = growth_fn
        else:
            self.normalise_kernel()
            if lenia=='classical':
                self.mu = growth_fn.mu
                self.sigma = growth_fn.sigma
                self.growth = growth_fn.growth_gaussian
            elif lenia=='extended':
                self.growth = growth_fn.multi_growth
            elif lenia=='3_channel':
                # ms means list of std for different channels
                self.ms = growth_fn.ms
                self.ss = growth_fn.ss
                self.hs = growth_fn.hs
                self.rs = growth_fn.rs
                self.c0s = growth_fn.c0s
                self.c1s = growth_fn.c1s
                self.growth = growth_fn.multi_growth 
            # elif lenia=='Quantum':
            #     self.mu = growth_fn.mu
            #     self.sigma = growth_fn.sigma

        # The cell state
        self.cell = cells
        # times for simulating adaptive step growth
        self.min_dt = Adaptiveness[0]
        self.max_dt = Adaptiveness[2]
        self.error_dt = Adaptiveness[1]

        # different world generation with dt/stepsize changing 
        if self.method=='StepSize':
            self.show_comparison() 
        elif self.method=='AdaptiveStep':
            self.show_adaptive_world()
        else:
            if isinstance(self.cell, list):
                self.world_shape = (self.cell)[0].shape
                self.fig, self.img = self.show_world(s=2) # Frames of animation
            else:
                self.world_shape = self.cell.shape 
                self.fig, self.img = self.show_world(s=1) # Frames of animation

        # record execution time
        self.start_time = timeit.default_timer()
        self.timings = []
        if isinstance(self.cell, list):
            self.type_exec_t = type + '_size_' + str(self.cell[0].shape[-1])
        else: 
            self.type_exec_t = type + '_size_' + str(self.cell.shape[-1])
        self.npy = os.path.join(EXEC_T_PATH, self.type_exec_t) 

        # count times for adaptive step
        self.t_count = 0.0
        # parameters for store generations as npz file
        self.count = 0
        if isinstance(self.cell, list):
            self.generation = type + '_size_' + str(self.cell[0].shape[-1])
        else:
            self.generation = type + '_size_' + str(self.cell.shape[-1])
        self.npz = os.path.join(NPZ_PATH, self.generation)
        self.npz_load = self.npz +'.npz'
        self.savez_dict = dict()
        # invert cmap
        self.savez_dict['generation_0'] = self.cell
        self.has_gpu = True
        self.is_gpu = True
        self.compile_gpu(self.cell)
    """
    introduce five ways to compute the convolution
    First one is the basic convolution function by scipy
    second is numpy's fft
    Third is gpu based reikna fft integrated with numpy
    Fourth one is the scipy's fft
    Last one is pyFFTW package interface with scipy's fft convolve
    """

#---------------------------Reikna------------------------------------

    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create()
            self.gpu_fft = reikna.fft.FFT(A).compile(self.gpu_thr)
            self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
        except Exception as exc:
            self.has_gpu = False
            self.is_gpu = False


    def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
        if self.is_gpu and self.gpu_thr and gpu_func:
            op_dev = self.gpu_thr.to_device(A.astype(dtype))
            gpu_func(op_dev, op_dev, **kwargs)
            return op_dev.get()
        else:
            return cpu_func(A)

    def fft(self, A): return self.run_gpu(A, np.fft.fft2, self.gpu_fft, np.complex64)
    def ifft(self, A): return self.run_gpu(A, np.fft.ifft2, self.gpu_fft, np.complex64, inverse=True)
    def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)
    def fft_conv(self, A, K): return np.real(self.ifft(self.fft(self.fftshift(K))*self.fft(A)))

#---------------------------pyFFTW------------------------------------
    
    def fftw(self, A, K):
        # Use the backend pyfftw.interfaces.scipy_fft
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
            # Turn on the cache for optimum performance
            pyfftw.interfaces.cache.enable()
            return np.real(fft_convolve2d_scipy(A, K))


    def normalise_kernel(self):
        """Normalise the kernel such the values sum to 1. 
        This makes generalisations much easier and ensures that the range of the neighbourhood sums is independent 
        of the kernel used. 
        Ensures the values of the growth function are robust to rescaling of the board/kernel. 
        Returns:
            np.array: The resulting normlised kernel
        """
        if isinstance(self.kernel, list):
            kernel_norm = [ K / np.sum(K) for K in self.kernel ]
            self.kernel_norm = kernel_norm
        else:
            self.k = self.kernel
            kernel_norm = self.kernel / (1*np.sum(self.kernel))
            self.norm_factor = 1/ (1*np.sum(self.kernel))
            self.kernel = kernel_norm 
        return kernel_norm 
    
    """ 
    funtions to generate the fig and set the first img array to the default creature shape    
    """
    def show_comparison(self):
        
        titles =[f"{self.type} pattern, step size {self.dT[1]}, every 1 frames"]

        self.fig, self.ax = plt.subplots(1,3, figsize=(9,3), facecolor="white")

        grid_display = self.cell[1].squeeze()
        slow_grid_display = self.cell[0].squeeze()
        fast_grid_display = self.cell[2].squeeze()
        # grid_display = self.cell[1]
        # slow_grid_display = self.cell[0]
        # fast_grid_display = self.cell[2]
        
        self.subplot_0 = self.ax[0].imshow(slow_grid_display, interpolation="nearest", vmin=0.0, vmax=1)
        self.subplot_1 = self.ax[1].imshow(grid_display, interpolation="nearest", vmin=0.0, vmax=1)
        self.subplot_2 = self.ax[2].imshow(fast_grid_display, interpolation="nearest", vmin=0.0, vmax=1)
        
        self.fig.suptitle(titles[0], fontsize=8)
        for hh in range(3):
            self.ax[hh].set_yticklabels('')
            self.ax[hh].set_xticklabels('')
        
        plt.tight_layout()

    def show_adaptive_world(self):

        titles = ["Adaptive step size demo"]
        self.fig, self.ax = plt.subplots(1,1, figsize=(9,9), facecolor="white")
        
        grid_display = self.cell.squeeze()
        
        self.subplot_0 = self.ax.imshow(grid_display, interpolation="nearest", vmin=0.0, vmax=1)
        
        posx = self.cell.shape[-1]//2
        posy = self.cell.shape[-1]//10
        self.ax.text(posx, posy, titles[0], fontsize=15, color='white', ha='center', va='center', alpha = 1)
        tposx = self.cell.shape[-1]//2
        tposy = self.cell.shape[-1] - self.cell.shape[-1]//10
        self.text = self.ax.text(tposx, tposy, f"StepSize={round(self.dT,2)}", fontsize=15, color='white', ha='center', va='center', alpha = 1)
        self.ax.axis('off')

        plt.tight_layout()

    def show_world(self, s, display:bool=False):
        dpi = 50 # Using a higher dpi will result in higher quality graphics but will significantly affect computation
        self.fig = plt.figure(figsize=(10*np.shape(self.cell)[s]/dpi, 10*np.shape(self.cell)[s-1]/dpi), dpi=dpi)
        ax = self.fig.add_axes([0, 0, 1, 1])
        if isinstance(self.cell, list):
            posx = self.cell[0].shape[-1]//2
            posy = self.cell[0].shape[-1]//10
        else:
            posx = self.cell.shape[-1]//2
            posy = self.cell.shape[-1]//10
            ax.text(posx, posy, self.title, fontsize=40, color='white', ha='center', va='center', alpha = 1)
        ax.axis('off')
        # if self.method=='Quantum':
        #     c = self.cell
        #     self.img = ax.imshow(convert_prob(c), cmap=self.cmap, interpolation='none', aspect=1, vmin=0) #  vmax=vmax 

        if s==1:
            self.img = ax.imshow(self.cell, cmap=self.cmap, interpolation='none', aspect=1, vmin=0) #  vmax=vmax
        else:
            self.img = ax.imshow(np.dstack(self.cell), cmap=self.cmap, interpolation='none', aspect=1, vmin=0) #  vmax=vmax
        if display:
            plt.show()
        else: # Do not show intermediate figures when creating animations (very slow)
            plt.close()
        return self.fig, self.img

    """
    Animation steps for normal lenia and
    Qanimate_step: Semi-Quantum version of lenia
    Sanimate_step: lenia with step size changes
    Aanimate_steo: lenia with adaptive step size changes
    animate_step_extended: multiple kernel
    animate_step_channel: multiple channels

    with S and A animation, the plot directly return a plot without generate img arrays
    """
    def animate_step(self, i:int) -> plt.imshow:
        self.count += 1
        if not self.demo:
            if self.type=='conway':
                if self.method=='original':
                    U = sum(np.roll(self.cell, (i,j), axis=(0,1)) for i in (-1,0,+1) for j in (-1,0,+1) if (i!=0 or j!=0)) 
                    self.cell = self.growth(self, self.cell, U, False) # conway rule
                elif self.method=='convolve':
                    U = scipy.signal.convolve2d(self.cell, self.kernel, mode='same', boundary='wrap')
                    self.cell = np.clip(self.cell + self.dT*self.growth(self, self.cell, U, True), 0, 1)
                self.savez_dict['generation_'+str(self.count)] = 1-self.cell

            elif self.lenia=='Quantum':
                U = fft_convolve2d(self.cell, self.kernel)
                G = np.zeros((self.cell.shape[0], self.cell.shape[1]))
                for i in range(len(G)):
                    for j in range(len(G)):
                        # G[i,j] = self.growth(U[i,j], self.cell[i,j])
                        G[i,j] = 2*(cont_SQGOL(self.cell[i,j], U[i,j])**2)-1
                self.cell = np.clip(self.cell + self.dT*G, 0, 1) 
                self.savez_dict['generation_'+str(self.count)] = self.cell 

            else:
                # record the execution time for algorithm comparison
                self.timings.append(timeit.default_timer()-self.start_time)
                if self.method=='convolve':
                    U = scipy.signal.convolve2d(self.cell, self.kernel, mode='same', boundary='wrap')
                    if self.clip=='soft_clip':
                        self.cell = soft_clip(self.cell + self.dT*self.growth(U), 0, 1)
                    else:
                        self.cell = np.clip(self.cell + self.dT*self.growth(U), 0, 1)

                elif self.method=='fft':
                    U = fft_convolve2d(self.cell, self.kernel)
                    if self.clip=='soft_clip': 
                        self.cell = soft_clip(self.cell + self.dT*self.growth(U), 0, 1) 
                    else:
                        self.cell = np.clip(self.cell + self.dT*self.growth(U), 0, 1) 

                elif self.method=='gpu':
                    U = self.fft_conv(self.cell, self.kernel)
                    self.cell = np.clip(self.cell + self.dT*self.growth(U), 0, 1)  

                elif self.method=='scipy':
                    U = fft_convolve2d_scipy(self.cell, self.kernel)
                    self.cell = np.clip(self.cell + self.dT*self.growth(U), 0, 1)  
               
                elif self.method=='pyfftw':
                    U = self.fftw(self.cell, self.kernel)
                    self.cell = np.clip(self.cell + self.dT*self.growth(U), 0, 1)  

                self.savez_dict['generation_'+str(self.count)] = self.cell
        else:
            tmp= np.load(self.npz_load)
            self.cell = tmp['generation_'+str(self.count)]
        self.img.set_array(self.cell) # render the updated state 
        return self.img,

    # directly showing in notebook
    # def Qanimate_step(self, i:int) -> plt.imshow:
    #     U = fft_convolve2d(convert_prob(self.cell), self.kernel) 
    #     self.cell = Qupdate(self.cell, U/100)
    #     c = convert_prob(self.cell)
    #     self.img.set_array(c)
    #     return self.img,

    def Sanimate_step(self, i:int) -> plt.imshow:

        for i in range(len(self.cell)):
            # U = fft_convolve2d(self.cell[i].squeeze(), self.kernel)
            # self.cell[i] = np.clip(self.cell[i].squeeze() + self.dT[i]*self.growth(U), 0, 1)   
            U = fft_convolve2d(self.cell[i], self.kernel)
            self.cell[i] = np.clip(self.cell[i] + self.dT[i]*self.growth(U), 0, 1)   
        grid_display = self.cell[1].squeeze()
        slow_grid_display = self.cell[0].squeeze()
        fast_grid_display = self.cell[2].squeeze()
        # grid_display = self.cell[1]
        # slow_grid_display = self.cell[0]
        # fast_grid_display = self.cell[2]
        self.subplot_0.set_array(slow_grid_display)
        self.subplot_1.set_array(grid_display)
        self.subplot_2.set_array(fast_grid_display)

        self.ax[0].set_title(self.title[0], fontsize=7)
        self.ax[1].set_title(self.title[1], fontsize=7)
        self.ax[2].set_title(self.title[2], fontsize=7)
        
        plt.tight_layout()

    def Aanimate_step(self, i:int) -> plt.imshow:
        # count the total time elapsed
        # self.t_count += self.error_dt

        self.count += 1

        if self.count==20:
            self.count = 0
            self.dT += self.error_dt 
        
        if self.dT > self.max_dt or self.dT < self.min_dt:
            pass

        U = fft_convolve2d(self.cell.squeeze(), self.kernel)
        self.cell = np.clip(self.cell.squeeze() + self.dT*self.growth(U), 0, 1)   

        grid_0_display = self.cell.cpu().squeeze()
            
        self.subplot_0.set_array(grid_0_display)
        self.text.set_text(round(self.dT,2))
        self.ax.axis('off') 
        # self.ax.set_title(f"StepSize={self.dT:.3f} &  accumulated time: {self.t_count:.3f}", fontsize=15)

    def animate_step_extended(self, i:int) -> plt.imshow:
        self.count += 1
        if not self.demo:
            self.timings.append(timeit.default_timer()-self.start_time)
            if self.method=='fft':
                Us = fft_multi_convolve2d(self.cell, self.kernel_norm)
                Gs = self.growth(Us, self.cell)
                if self.clip=='soft_clip':
                    self.cell = soft_clip(self.cell + self.dT * np.mean(np.asarray(Gs),axis=0), 0, 1)
                else:
                    self.cell = np.clip(self.cell + self.dT * np.mean(np.asarray(Gs),axis=0), 0, 1)

            elif self.method=='scipy':
                Us = fft_multi_convolve2d_scipy(self.cell, self.kernel_norm)
                Gs = self.growth(Us, self.cell)
                self.cell = np.clip(self.cell + self.dT * np.mean(np.asarray(Gs),axis=0), 0, 1)
            self.savez_dict['generation_'+str(self.count)] = self.cell
        else:
            tmp = np.load(self.npz_load)
            self.cell = tmp['generation_'+str(self.count)]
        self.img.set_array(self.cell)
        return self.img,

    def animate_step_channel(self, i:int) -> plt.imshow:
        self.count += 1
        if not self.demo:
            self.timings.append(timeit.default_timer()-self.start_time)
            if self.method=='fft':
                fKs = [ fft2(fftshift(K)) for K in self.kernel_norm]
                fAs = [ fft2(A) for A in self.cell ]
                Us = [ np.real(ifft2(fK * fAs[c0])) for fK,c0 in zip(fKs,self.c0s) ]
                ''' calculate growth values for destination channels c1 '''
                Gs = self.growth(Us, self.cell)
                Hs = [ sum(h*G for G,h,c1 in zip(Gs,self.hs,self.c1s) if c1==c) for c in range(3) ]
                if self.clip=='soft_clip':
                    self.cell = [ soft_clip(A + self.dT * H, 0,1) for A,H in zip(self.cell, Hs) ]
                else:
                    self.cell = [ np.clip(A + self.dT * H, 0,1) for A,H in zip(self.cell, Hs) ]

            elif self.method=='scipy':
                fKs = [ scipy.fft.fft2(scipy.fft.fftshift(K)) for K in self.kernel_norm]
                fAs = [ scipy.fft.fft2(A) for A in self.cell ]
                Us = [ np.real(scipy.fft.ifft2(fK * fAs[c0])) for fK,c0 in zip(fKs,self.c0s) ]
                ''' calculate growth values for destination channels c1 '''
                Gs = self.growth(Us, self.cell)
                Hs = [ sum(h*G for G,h,c1 in zip(Gs,self.hs,self.c1s) if c1==c) for c in range(3) ]
                if self.clip=='soft_clip':
                    self.cell = [ soft_clip(A + self.dT * H, 0,1) for A,H in zip(self.cell, Hs) ]
                else: 
                    self.cell = [ np.clip(A + self.dT * H, 0,1) for A,H in zip(self.cell, Hs) ]
            self.savez_dict['generation_'+str(self.count)] = self.cell
        else:
            tmp = np.load(self.npz_load)
            self.cell = tmp['generation_'+str(self.count)]

        if self.type=='pacman' or self.type=='pacmanSoft':
            self.img.set_array(np.dstack([self.cell[1], self.cell[2], self.cell[0]])) # rearrange cells for better visual effect
        else:
            self.img.set_array(np.dstack(self.cell)) 
        return self.img,

    """
    function call for animation generation
    """
    def animate(self, 
                frames:int, 
                interval:float=50, 
                blit=True,
                ):
        if self.lenia=='classical' or self.lenia=='Quantum':
            self.anim = matplotlib.animation.FuncAnimation(self.fig, self.animate_step, 
                                                frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit) 
        elif self.lenia=='extended':
           self.anim = matplotlib.animation.FuncAnimation(self.fig, self.animate_step_extended, 
                                                frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit)
        elif self.lenia=='3_channel':
           self.anim = matplotlib.animation.FuncAnimation(self.fig, self.animate_step_channel, 
                                                frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit)    

    # integrated with the classical animation
    # def Qanimate(self, 
    #             frames:int, 
    #             interval:float=50, 
    #             blit=True,
    #             ):
    #     self.anim = matplotlib.animation.FuncAnimation(self.fig, self.Qanimate_step, 
    #                                         frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit) 

    def Sanimate(self, 
                frames:int,
                interval:float=50, 
                ):
        # self.anim = matplotlib.animation.FuncAnimation(self.fig, self.Sanimate_step, 
                                            # frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit)
        path = os.path.join(OUTPUT_PATH, ''.join([self.type, '.mp4']))  
        matplotlib.animation.FuncAnimation(self.fig, self.Sanimate_step, frames=frames, interval=interval).save(path)

    def Aanimate(self, 
                frames:int, 
                interval:float=50, 
                ):
        # self.anim = matplotlib.animation.FuncAnimation(self.fig, self.Sanimate_step, 
                                            # frames=frames, interval=interval, save_count=MAX_FRAMES, blit=blit)
        path = os.path.join(OUTPUT_PATH, ''.join([self.type, '.mp4']))  
        matplotlib.animation.FuncAnimation(self.fig, self.Aanimate_step, frames=frames, interval=interval).save(path)
        # IPython.display.HTML(matplotlib.animation.FuncAnimation(self.fig, self.Sanimate_step, frames=frames, interval=interval).to_jshtml())

    """
    save animation in either gif or mp4 file
    """
    def save_animation(self, 
                       filename:str,
                        ):
        if not self.anim:
            raise Exception('ERROR: Run animation before attempting to save')
            return 
        
        fmt = os.path.splitext(filename)[1] # isolate the file extension
        
        try: # make outputs folder if not already exists
            os.makedirs(OUTPUT_PATH)
        except FileExistsError:
            # directory already exists
            pass

        if fmt == '.gif':
            f = os.path.join(OUTPUT_PATH, filename) 
            writer = matplotlib.animation.PillowWriter(fps=30) 
            self.anim.save(f, writer=writer)
        elif fmt == '.mp4':
            f = os.path.join(OUTPUT_PATH, filename) 
            writer = matplotlib.animation.FFMpegWriter(fps=30) 
            self.anim.save(f, writer=writer)
        else:
            raise Exception('ERROR: Unknown save format. Must be .gif or .mp4')
        np.savez(self.npz, **self.savez_dict)
        np.save(self.npy, self.timings)

    """
    illustrate kernel and growth information
    from left to right:

    kernel image   |    kernel cross section   |   Growth function with changing neighbor sum
    """
    def plot_kernel_info(self,
                         R:int=0,
                         cmap:str='viridis', 
                         bar:bool=False,
                         save:str=None,
                         ) -> None:
        if self.type=='conway':
            k_xsection = self.kernel[self.kernel.shape[0] // 2, :]
        else: 
            k_xsection = self.k[self.kernel.shape[0] // 2, :]
        k_sum = np.sum(self.kernel)
        
        fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,1,2]})
        
        # Show kernel as heatmap
        ax[0].imshow(self.kernel, cmap=cmap, vmin=0)
        ax[0].title.set_text('Kernel')
        
        # Show kernel cross-section
        ax[1].title.set_text('Kernel Cross-section')
        ax[1].set_xlim([self.kernel.shape[0] // 2 - 3 - R, self.kernel.shape[0] // 2 + 3 + R])
        if bar==True:
            ax[1].bar(range(0,len(k_xsection)), k_xsection, width=1)
        else:
            ax[1].plot(k_xsection)
        
        # Growth function
        A = np.zeros((1,1)) # dummy var for calling growth func
        ax[2].title.set_text('Growth Function')
        if self.type=='conway':
            x = np.arange(k_sum + 1)
            ax[2].step(x, self.growth(self,A,x,True))
        else:
           x = np.linspace(0, k_sum, 1000)
           ax[2].step(x, self.growth(x))
        
        if save:
            print('Saving kernel and growth function info to', os.path.join(OUTPUT_PATH, 'kernel_info'))
            plt.savefig(os.path.join(OUTPUT_PATH, 'kernel_info.png') )

    """for multiple kernels"""
    def plot_kernel_info_list(self, R, kernels, use_c0=False, cmap='viridis'):
        K_size = self.kernel[0].shape[0];  K_mid = K_size // 2
        fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,2,2]})
        if use_c0:
            K_stack = [ np.clip(np.zeros(self.kernel[0].shape) + sum(K/3 for k,K in zip(kernels,self.kernel) if k['c0']==l), 0, 1) for l in range(3) ]
        else:
            K_stack = self.kernel[:3]
        ax[0].imshow(np.dstack(K_stack), cmap=cmap, interpolation="nearest", vmin=0)
        ax[0].title.set_text('kernels Ks')
        X_stack = [ K[K_mid,:] for K in self.kernel_norm ]
        ax[1].plot(range(K_size), np.asarray(X_stack).T)
        ax[1].title.set_text('Ks cross-sections')
        ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
        x = np.linspace(0, 1, 1000)
        G_stack = [ Growth_fn(mu=k['m'], sigma=k['s']).growth_gaussian(x) * k['h'] for k in kernels ]
        ax[2].plot(x, np.asarray(G_stack).T)
        ax[2].axhline(y=0, color='grey', linestyle='dotted')
        ax[2].title.set_text('growths Gs')
        return fig

"""show video at the end of simulation"""
def show_videofile(fn):
  b64 = base64.b64encode(open(fn, 'rb').read()).decode('utf8')
  s = f'''<video controls loop>
<source src="data:video/mp4;base64,{b64}" type="video/mp4">
Your browser does not support the video tag.</video>'''
  display(HTML(s))