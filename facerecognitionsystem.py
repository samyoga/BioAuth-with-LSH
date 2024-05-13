from zipfile import ZipFile
import cv2
import json
import numpy as np

# function to extract zip files
def extract_images(zip_file):
    with ZipFile(zip_file, 'r') as zip_extract:
        zip_extract.extractall()

# function to enroll users
def enroll_user(image_data):
    features_dict = {}
    for i in range(1, 101):
        #read images from the gallery set
        images_gallery = cv2.imread(f"GallerySet/subject{i}_img1.pgm")
        if i not in features_dict:
            features_dict[i] = []
        features_dict[i].append(images_gallery.tolist())
    
    return features_dict

# function to hash the templates
def hash_templates():
    
    return 0

# function to generate random matrices
def generate_random_matrices(num_matrices, matrix_size):
    #Initialize a list to store matrices
    random_matrices = []

    for _ in range(num_matrices):
        # Generate a random matrix with matrix_size
        # Each element of the matrix is a list with 3 random elements
        random_matrix = np.random.rand(matrix_size, matrix_size, 3)
        random_matrices.append(random_matrix)

    return random_matrices

# function to verify users
def verify_user():
    return 0

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    gallery_data = "GallerySet"
    features_dict = enroll_user(gallery_data)
    
    # dump dictionary to JSON
    with open("gallery.json", "w") as outfile:
        json.dump(features_dict, outfile)
        
    probe_data = "ProbeSet"
