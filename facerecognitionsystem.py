from zipfile import ZipFile
import cv2
import json
import numpy as np
import re

# function to extract zip files
def extract_images(zip_file):
    with ZipFile(zip_file, 'r') as zip_extract:
        zip_extract.extractall()

# function to extract features of gallery data
def extract_features_gallery():
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

# function to extract features of probe data
def extract_features_probe():
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
    
    return dot_products

def verify_users(hashed_gallery, hashed_probe, threshold = 0.8):
    matches = {} # To store the best match for each probe

    #Loop through each probe hash
    for probe_key, probe_value in hashed_probe.items():
        best_match_key = None
        best_match_score = -1

        # Convert the probe value to a numpy array 
        probe_array = np.array(probe_value)

        # Loop through each gallery hash
        for gallery_key, gallery_value in hashed_gallery.items():
            gallery_array = np.array(gallery_value)
 
            # Ensure the gallery and probe arrays have the same shape before comparison
            if probe_array.shape != gallery_array.shape:
                continue

            # Calculate similarity
            similarity = np.sum(probe_array == gallery_array)/probe_array.size

            #keep track of the best match
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_key = gallery_key

        # Store the best match if it exceeds the similarity threshold
        if best_match_score >= threshold:
            matches[probe_key] = best_match_key
        else:
            matches[probe_key] = "No match found"


    return matches


# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    features_dict_gallery = extract_features_gallery()
    features_dict_probe = extract_features_probe()

    num_matrices = 255
    matrix_size = 50

    # only run once at the beginnig
    # generate_random_matrices(num_matrices, matrix_size)


    # Read the JSON file and load it into a dictionary
    with open('random_matrices.json', 'r') as file:
        random_matrices = json.load(file)
    hash_value_gallery = hash_templates(features_dict_gallery, random_matrices)
    hash_value_probe = hash_templates(features_dict_probe, random_matrices)
    
    
    # dump dictionary to JSON
    # with open("probe.json", "w") as outfile:
    #     json.dump(features_dict_probe, outfile)

    # dump dictionary to JSON
    # with open("gallery.json", "w") as outfile:
    #     json.dump(features_dict_gallery, outfile)

    # with open("random_matrices.json", "w") as outfile:
    #     json.dump(random_matrices, outfile)

    # with open("hashed_output_gallery.json", "w") as outfile:
    #     json.dump(hash_value_gallery, outfile, indent=4)
    
    # with open("hashed_output_probe.json", "w") as outfile:
    #     json.dump(hash_value_probe, outfile, indent=4)
    
    #Load hashed gallery and hashed probe from JSON
    with open('hashed_output_gallery.json', 'r') as gallery_file:
        hashed_gallery = json.load(gallery_file)

    with open('hashed_output_probe.json', 'r') as probe_file:
        hashed_probe = json.load(probe_file)

    #perform user verification
    matches = verify_users(hashed_gallery, hashed_probe, threshold=0.8)

    #output results
    for probe, gallery in matches.items():
        print(f"Probe {probe} matches with Gallery {gallery}")
