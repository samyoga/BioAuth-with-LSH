import numpy as np

class RandomVectorGenerator:
    def __init__(self, num_matrices=256, matrix_size=50):
        self.num_matrices = num_matrices
        self.matrix_size = matrix_size

    def generate(self):
        #Initialize a dictionary to store matrices
        binary_matrices_dict = {}

        for i in range(1, self.num_matrices+1):
            # Generate a random matrix with matrix_size
            # Each element of the matrix is a list with 50 random elements
            # random_matrix = np.random.randint(0, 2, size=(matrix_size, matrix_size))
            random_matrix = np.random.choice([-1, 0, 1], size=(self.matrix_size, self.matrix_size))

            random_vector = random_matrix.flatten()

            #convert matrix to list and store in dictionary
            binary_matrices_dict[i] = random_vector.tolist()
            
        # print(binary_matrices_dict)
        return binary_matrices_dict
        