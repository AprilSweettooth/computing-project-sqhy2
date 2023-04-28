This is the npz file storing each iteration of the array for lenia during simulation
The intent to store the array is to make the computation faster
Each file is composed of iterations of cell array
For example: generation_1 = array[[...], [...], ...]
To see the exact data, you can use np.load(filename, allow_pickle=True).files

This folder also contains data for entropy collections and each generation is stored similarly as above