import numpy as np

def get_grid_entropy(grid):
    """
    $
    H = -\sum{ p log_2(p)}
    $
    """
    
    eps = 1e-9
    # convert grid to uint8
    
    stretched_grid = (grid - np.min(grid)) / np.max(grid - np.min(grid))
    
    uint8_grid = np.uint8(255*stretched_grid)
    
    p = np.zeros(256)
    
    for ii in range(p.shape[0]):
        p[ii] = np.sum(uint8_grid == ii)
        
    # normalize p
    p = p / p.sum()
    
    h = - np.sum( p * np.log2( eps+p))
    
    return h

def get_spatial_entropy(grid, window_size=63):
    
    half_window = (window_size - 1) // 2
    dim_grid = grid.shape
    
    # padded_grid = np.pad(grid, pad_width=(0), mode="wrap")
    
    spatial_h = np.zeros_like(grid)

    for xx in range(dim_grid[0]): #half_window, half_window + dim_grid[0]):
        for yy in range(dim_grid[1]): #half_window, half_window + dim_grid[1]):
            
            #spatial_h[xx, yy] = get_grid_entropy(\
            #        padded_grid[xx+half_window:xx + 2*half_window+1, \
            #        yy+half_window:yy +2*half_window+1])
            x_start = int(max([xx-half_window, 0]))
            x_end = int(min([xx+half_window,dim_grid[0]]))
            y_start = int(max([yy-half_window, 0]))
            y_end = int(min([yy+half_window, dim_grid[1]]))
            spatial_h[xx,yy] = get_grid_entropy(grid[x_start:x_end, y_start:y_end])#.detach().cpu().numpy()) 
            
    return spatial_h


