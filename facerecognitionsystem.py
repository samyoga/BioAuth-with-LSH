from zipfile import ZipFile
import cv2
import json
import numpy as np

# function to extract zip files
def extract_images(zip_file):
    with ZipFile(zip_file, 'r') as zip_extract:
        zip_extract.extractall()

# function to enroll users
def enroll_user():
    features_dict = {}
    for i in range(1, 101):
        #read images from the gallery set
        images_gallery = cv2.imread(f"GallerySet/subject{i}_img1.pgm")
        if i not in features_dict:
            features_dict[i] = []
        features_dict[i].append(images_gallery.tolist())
    return features_dict

# function to hash the templates
def hash_templates(input_array, random_matrices):
    dot_products = []
    
    for random_matrix in random_matrices:
        product = np.dot(random_matrix, input_array)
        dot_products.append(product)
    
    return dot_products

# function to generate random matrices
def generate_random_matrices(num_matrices, matrix_size):
    #Initialize a list to store matrices
    random_matrices = []
    random_keys = []
    my_dict = {}

    for i in range(1, num_matrices+1):
        random_keys.append(i)

    for _ in range(num_matrices):
        # Generate a random matrix with matrix_size
        # Each element of the matrix is a list with 3 random elements
        random_matrix = np.random.randint(-100, 101, size=(matrix_size, matrix_size, 3))  
        random_matrices.append(random_matrix.tolist())

    
    
    # print(random_keys)
    key_value_pairs = zip(random_keys, random_matrices)
    my_dict = dict(key_value_pairs)
    print(my_dict)
    return my_dict

# function to verify users
def verify_user():
    return 0

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    features_dict = enroll_user()

    num_matrices = 255
    matrix_size = 50
    random_matrices = generate_random_matrices(num_matrices, matrix_size)
    # hash_templates(features_dict, random_matrices)
    
    # dump dictionary to JSON
    with open("gallery.json", "w") as outfile:
        json.dump(features_dict, outfile)

    with open("random_matrices.json", "w") as outfile:
        json.dump(random_matrices, outfile)
    probe_data = "ProbeSet"
