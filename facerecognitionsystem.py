from zipfile import ZipFile
import cv2
import json
import numpy as np
import re

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
        grayscale_image = cv2.cvtColor(images_gallery, cv2.COLOR_BGR2GRAY)

        if i not in features_dict:
            features_dict[i] = []
       
        features_dict[i].append(grayscale_image.tolist())
    
    for key, value_list in features_dict.items():
        # Update the dictionary with the modified value
        features_dict[key] = value_list[0]

    # print(features_dict)
    return features_dict

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
        # Each element of the matrix is a list with 50 random elements
        random_matrix = np.random.randint(-100, 101, size=(matrix_size, matrix_size))  
          # print(random_matrices)
        random_matrices.append(random_matrix.tolist())

    # Converting to dictionary
    key_value_pairs = zip(random_keys, random_matrices)
    my_dict = dict(key_value_pairs)
    return my_dict

# function to hash the templates
def hash_templates(input_array, random_matrices):
    dot_products = {}

    for input_key, input_value in input_array.items():
        for random_key, random_value in random_matrices.items():
            input_data = np.array(input_value)
            random_data = np.array(random_value)

            input_vectors = input_data.reshape(-1)
            random_vectors = random_data.reshape(-1)
           
            # multiply input vectors with random vectors
            # apply sign function to the product
            mult_res = np.sign(input_vectors * random_vectors).tolist()
            dot_products[random_key] = mult_res

    # print(dot_products)
    
    return dot_products

# function to verify users
def verify_user():
    features_dict = {}
    counter = 1
    for i in range(1, 101):
        #read images from the probe set
        images_probe_1 = cv2.imread(f"ProbeSet/subject{i}_img2.pgm")
        images_probe_2 = cv2.imread(f"ProbeSet/subject{i}_img3.pgm")
        grayscale_probeimage1 = cv2.cvtColor(images_probe_1, cv2.COLOR_BGR2GRAY)
        grayscale_probeimage2 = cv2.cvtColor(images_probe_2, cv2.COLOR_BGR2GRAY)

        if i not in features_dict:
            features_dict[i] = []
        
        features_dict[counter] = grayscale_probeimage1.tolist()
        counter +=1
        features_dict[counter] = grayscale_probeimage2.tolist()
        counter +=1
    
    # for key, value_list in features_dict.items():
    #     # Update the dictionary with the modified value
    #     features_dict[key] = value_list[0]

    # print(features_dict)
    return features_dict

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    features_dict_enroll = enroll_user()
    features_dict_verify = verify_user()

    num_matrices = 255
    matrix_size = 50

    # only run once at the beginnig
    # generate_random_matrices(num_matrices, matrix_size)


    # Read the JSON file and load it into a dictionary
    with open('random_matrices.json', 'r') as file:
        random_matrices = json.load(file)
    hash_value = hash_templates(features_dict_enroll, random_matrices)
    
    # dump dictionary to JSON
    # with open("probe.json", "w") as outfile:
    #     json.dump(features_dict_verify, outfile)

    # dump dictionary to JSON
    # with open("gallery.json", "w") as outfile:
    #     json.dump(features_dict_enroll, outfile)

    # with open("random_matrices.json", "w") as outfile:
    #     json.dump(random_matrices, outfile)

    with open("hashed_output_gallery.json", "w") as outfile:
        json.dump(hash_value, outfile)
    
