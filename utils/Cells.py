#---------------------------Import Packages------------------------------------

import numpy as np
import scipy.sparse
from .Kernel import *

#---------------------------Cells------------------------------------

class Cells:
    """Class for the cellular automata board. Holds the state values at each timestep.
    Initialise boards with different initial conditions:
        - Zeros
        - uniform between 0 and 1
        - Random values
        - Sparse random values 
        - Gaussian distributed conditions
        - Radially-symmetric conditions
    """
    def __init__(self,
                 initialisation_type:str='random',
                 grid_size:int=64, 
                 seed:int=None):
        """_summary_

        Args:
            grid_size (int): The size of the array used to store the values for the cellular automata
            seed (int, optional): The random seed used during board creation. Set the seed to obtain reproducible
            reults with random boards. Defaults to None.
            
        """
        self.grid_size = grid_size
        self.density = 0.5 # Sparsity
        self.initialisation_type=initialisation_type
        self.type = type
        self.seed = seed
        self.cell = self.initialise_cell()
        
    def initialise_cell(self) -> np.array:
        """Create an array used to store the values for the cellular automata.

        Returns:
            np.array: The intitialised board at t=0
        """
        np.random.seed(self.seed)
        if self.initialisation_type == 'zeros': 
            self.cell = np.zeros([self.grid_size, self.grid_size])

        elif self.initialisation_type == 'uniform': 
            self.cell = np.random.uniform(0,1,(self.grid_size, self.grid_size))
            
        elif self.initialisation_type == 'random': 
            self.cell = np.random.randint(2, size=(self.grid_size, self.grid_size))
            
        elif self.initialisation_type == 'sparse': 
            self.cell = scipy.sparse.random(self.grid_size, self.grid_size, density=self.density).A
            
        elif self.initialisation_type == 'gaussian':
            R = self.grid_size/2
            self.cell = np.linalg.norm(np.asarray(np.ogrid[-R:R, -R:R], dtype=object) + 1) / R
            
        elif self.initialisation_type == 'ring':
            self.cell = Kernel().smooth_ring_kernel(32)

        return self.cell 
