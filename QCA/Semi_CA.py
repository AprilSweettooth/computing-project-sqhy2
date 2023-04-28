#---------------------------Import Packages------------------------------------

import numpy as np
import json
import matplotlib.pyplot as plt

MAX_FRAMES = 3000
OUTPUT_PATH = './demo'

#---------------------------Semi Quantum helpers------------------------------------

# convert real part and imaginary part of the matrix into a single NxNx2 matrix
def load_Qcell_from_json(filename:str) -> dict:

    print("Reading JSON file {} ...".format(filename))

    with open(filename, "r") as read_file:
        d_load = json.load(read_file)

        # for interference pattern
        # d_load["cells"] = np.asarray(d_load["cell"])
        # d_load["Icells"] = np.asarray(d_load["Icell"])

    print('Completed !')
    return np.asarray(d_load)

def norm_cell(a):
    new_a = np.zeros([a.shape[0], 2])
    for i in range(a.shape[0]):
        new_a[i,:] = np.array([a[i], np.sqrt(1 - a[i]**2)])
    return new_a

# paste the quantum creature inside the quantum world
def Qcell(Q_pattern, grid_size):

    def norm_cell(a):
        new_a = np.zeros([a.shape[0], a.shape[1], 2], dtype=complex)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                new_a[i,j] = np.array([a[i,j], np.sqrt(1 - abs(a[i,j])**2)])
        return new_a
    
    sx = (grid_size-Q_pattern.shape[0]) // 2
    sy = (grid_size-Q_pattern.shape[1]) // 2

    cell = np.zeros([grid_size, grid_size, 2], dtype=complex)
    cell[:,:,:] = np.array([0+0j,1], dtype=complex)
    
    cell[sx:sx+Q_pattern.shape[0], sy:sy+Q_pattern.shape[1], :] = norm_cell(Q_pattern)

    return cell

def neighbouring_sites(i,j,width):
    '''Return the coordinates of the 4 sites adjacent to [i,j] on an width*width lattice. Takes into account periodic boundary conditions.'''
    neighbour = []
    neighbour.append([(i+1)%width, j]) 
    neighbour.append([(i-1)%width, j]) 
    neighbour.append([i, (j+1)%width]) 
    neighbour.append([i, (j-1)%width])
    neighbour.append([(i+1)%width, (j+1)%width]) 
    neighbour.append([(i+1)%width, (j-1)%width])
    neighbour.append([(i-1)%width, (j+1)%width])  
    neighbour.append([(i-1)%width, (j-1)%width])     
    return neighbour

def liveliness(i,j,cells,width, interference=False):
    '''Sums the cell value of all neighbours of the cell at [i,j].'''
    if not interference:
        return np.sum([cells[site[0], site[1]][0] for site in neighbouring_sites(i,j,width)])
    else:
        return np.sum([cells[site[0], site[1]] for site in neighbouring_sites(i,j,width)])

def convert_prob(cells):
    new_cells = np.zeros([cells.shape[0], cells.shape[0]])
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            new_cells[i,j] = abs(cells[i,j][0])**2
    return new_cells

# Semi-quantum Game of Life
def SQGOL(cell, a):
    # for GoL we need discrete value of liveliness
    B = np.array([[1, 1], [0, 0]])
    D = np.array([[0, 0], [1, 1]])
    S = np.array([[1, 0], [0, 1]])
    # rules
    if a <= 1 or a >= 4:
        op = D

    elif (a > 1 and a <= 2):
        op = ((np.sqrt(2) + 1) * 2 - (np.sqrt(2) + 1) * a) * D + (
            a - 1) * S  #(((np.sqrt(2)+1)*(2-a))**2+(a-1)**2)

    elif (a > 2 and a <= 3):
        op = (((np.sqrt(2) + 1) * 3) - (np.sqrt(2) + 1) * a) * S + (
            a - 2) * B  #(((np.sqrt(2)+1)*(3-a))**2+(a-2)**2)

    elif (a > 3 and a < 4):
        op = ((np.sqrt(2) + 1) * 4 - (np.sqrt(2) + 1) * a) * B + (
            a - 3) * D  #(((np.sqrt(2)+1)*(4-a))**2+(a-3)**2)

    # Normalize
    cell = np.matmul(op, cell)
    cell = cell / np.linalg.norm(cell)
    return cell

def cont_SQGOL(cell, a):
    # continuous update rule for lenia
    cell = np.array([cell, np.sqrt(1-cell**2)])
    B = np.array([[1, 1], [0, 0]])
    D = np.array([[0, 0], [1, 1]])
    S = np.array([[1, 0], [0, 1]])
    if a <= 0.1 or a >= 0.4:
        op = D

    elif (a > 0.1 and a <= 0.2):
        op = ((np.sqrt(2) + 1) * 0.2 - (np.sqrt(2) + 1) * a) * D + (
            a - 0.1) * S  #(((np.sqrt(2)+1)*(2-a))**2+(a-1)**2)

    elif (a > 0.2 and a <= 0.3):
        op = (((np.sqrt(2) + 1) * 0.3) - (np.sqrt(2) + 1) * a) * S + (
            a - 0.2) * B  #(((np.sqrt(2)+1)*(3-a))**2+(a-2)**2)

    elif (a > 0.3 and a < 0.4):
        op = ((np.sqrt(2) + 1) * 0.4 - (np.sqrt(2) + 1) * a) * B + (
            a - 0.3) * D  #(((np.sqrt(2)+1)*(4-a))**2+(a-3)**2)

    # Normalize
    cell = np.matmul(op, cell)
    cell = cell / np.linalg.norm(cell)
    return cell[0]

# continuous rule for interference parttern
def cont_SQGOL_with_phase(cell, a):
    a = abs(a)
    p = np.angle(a)
    B = np.array([cell+np.sqrt(1-abs(cell)**2)*np.exp(1j*p), 0], dtype=complex) # directly get resulting state
    D = np.array([0, np.sqrt(1-abs(cell)**2)+abs(cell)*np.exp(1j*p)], dtype=complex) # to simplify calculation we apply the normalisation condition
    S = np.array([cell, np.sqrt(1-abs(cell)**2)], dtype=complex)

    if a <= 0.1 or a >= 0.4:
        c = D

    elif (a > 0.1 and a <= 0.2):
        c = ((np.sqrt(2) + 1) * 0.2 - (np.sqrt(2) + 1) * a) * D + (
            a - 0.1) * S  #(((np.sqrt(2)+1)*(2-a))**2+(a-1)**2)

    elif (a > 0.2 and a <= 0.3):
        c = (((np.sqrt(2) + 1) * 0.3) - (np.sqrt(2) + 1) * a) * S + (
            a - 0.2) * B  #(((np.sqrt(2)+1)*(3-a))**2+(a-2)**2)

    elif (a > 0.3 and a < 0.4):
        c = ((np.sqrt(2) + 1) * 0.4 - (np.sqrt(2) + 1) * a) * B + (
            a - 0.3) * D  #(((np.sqrt(2)+1)*(4-a))**2+(a-3)**2)
        
    # def normalise_complex_arr(a):
    #     a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    #     return a_oo/np.abs(a_oo).max()
    # normalise cell

    c = c[0] / np.sqrt(abs(c[0])**2 + abs(c[1])**2)
    return c

# conway update
def CQupdate(cells):
    size = cells.shape[0]
    new_cell = np.zeros((size, size, 2))
    for i in range(size):
        for j in range(size):
            a = liveliness(i,j,cells,size)
            new_cell[i,j] = SQGOL(cells[i,j], a)
    return new_cell

# lenia update 
def Qupdate(cells, U):
    size = cells.shape[0]
    new_cell = np.zeros((size, size, 2))
    for i in range(size):
        for j in range(size):
            new_cell[i,j] = cont_SQGOL(cells[i,j], U[i,j])
    return new_cell

# lenia update with interference
def Iupdate(cells):
    size = cells.shape[0]
    new_cell = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            U = liveliness(i,j,cells,size, interference=True)
            new_cell[i,j] = cont_SQGOL_with_phase(cells[i,j], U/10)
    return new_cell

# test the shape of growth function
def plot_update_Qrule(psi,m,s):
    bell = lambda x: np.exp(-((x-m)/s)**2 / 2)
    A = np.arange(0,1,0.01)
    G = np.zeros([len(A), len(psi)])
    for i in range(len(A)):
        for j in range(len(psi)):
            G[i, j] = cont_SQGOL(psi[j], A[i])**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Computation time Comparison of Orbium', size=20)
    ax1.title.set_text('Growth function Comparison')
    ax2.title.set_text('Shifted Continuous Growth function')

    for i in range(len(psi)):
        ax1.plot(A, G[:,i])
        ax1.plot(A, bell(A))
        ax2.plot(A, G[:,i])
        # ax2.plot(A, bell(A-(0.35-m)) + 0.95*psi[i][0]**4*bell(A-(0.25-m)))
        ax2.plot(A, bell(A-(0.3-m)))
    ax1.set_xlim((0,0.5))
    ax2.set_xlim((0,0.5))
    plt.show()