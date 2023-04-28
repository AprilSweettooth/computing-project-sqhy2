This is the npy file for execution time comparison.
First part of the file name is the creature and method to generate it
Second part is the overall cell size
For example, orbium_fft_size_64 means there is an overall 64x64 cell for orbium with FFT applied during iteration
The times are stored as a single array in the npy file. To see it, you can use code:
np.load(filename)