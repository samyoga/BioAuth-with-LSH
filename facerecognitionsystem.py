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
        #read images from the gallery set and convert to grayscale image
        images_gallery = cv2.imread(f"GallerySet/subject{i}_img1.pgm")
        grayscale_image = cv2.cvtColor(images_gallery, cv2.COLOR_BGR2GRAY)

        #converting grayscale to binary images using adaptive thresholding
        # apply adapting thresholding using initial values
        block_size = 11
        constant = 2

        binary_image = cv2.adaptiveThreshold(
            grayscale_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant
        )

        if i not in features_dict:
            features_dict[i] = []
       
        features_dict[i].append(binary_image.tolist())
    
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

        #converting grayscale to binary images using adaptive thresholding
        # apply adapting thresholding using initial values
        block_size = 11
        constant = 2

        binary_probeimage1 = cv2.adaptiveThreshold(
            grayscale_probeimage1,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant
        )

        binary_probeimage2 = cv2.adaptiveThreshold(
            grayscale_probeimage2,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant
        )

        if i not in features_dict:
            features_dict[i] = []
        
        features_dict[counter] = binary_probeimage1.tolist()
        counter +=1
        features_dict[counter] = binary_probeimage2.tolist()
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

    print(my_dict)
    return my_dict

def multiply_structures(structure1, structure2):
    # Intitalize dictionary to store the results
    result = {}

    # Iterate through each key in first structure
    for key1, matrix1 in structure1.items():
        matrix1 = np.array(matrix1) # convert list to numpy array

        # create a sub-dictionary for each key in structure1
        result[key1] = {}

        #Iterate through each key in the second structure
        for key2, matrix2 in structure2.items():
            matrix2 = np.array(matrix2) # convert list to numpy array

            # Ensure both matrices are 50*50 before multiplication 
            if matrix1.shape == matrix2.shape:
                #Perform element-wise multiplication (use np.dot for matrix multiplication)
                multiplied_matrix = np.multiply(matrix1, matrix2)

                #Apply the sign function to the multiplied matrix
                signed_matrix = np.sign(multiplied_matrix)

                # Convert the result back to list and store in the result dictionary
                result[key1][key2] = signed_matrix.tolist()

            else:
                print(f"Matrix dimensions don't match for {key1} and {key2}.")

    return result

def compute_similarity(matrix1, matrix2):
    """
    Compute the similarity between two matrices.
    For signed matrices, you can use Hamming distance or another suitable metric.
    """
    # Flatten the matrices to compare element-wise
    flat_matrix1 = np.array(matrix1).flatten()
    flat_matrix2 = np.array(matrix2).flatten()

    # print(flat_matrix1)

    # Compute the dot product
    dot_product = np.dot(flat_matrix1, flat_matrix2)

    # Compute the magnitudes (Euclidean norms)
    magnitude1 = np.linalg.norm(flat_matrix1)
    magnitude2 = np.linalg.norm(flat_matrix2)

    # Handle division by zero case
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # If either vector is all zeros, return similarity of 0

    # Compute the cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim
    
    # # Compute similarity (e.g., Hamming distance for binary data)
    # similarity = np.sum(flat_matrix1 == flat_matrix2) / len(flat_matrix1)
    # return similarity

def find_best_matches(hashed_gallery, hashed_probe, threshold=0.8):
    """
    Find the best matches between gallery and probe data based on similarity.
    """
    matches = {}

    for gallery_key, gallery_data in hashed_gallery.items():
    
        best_match_key = None
        best_match_score = -1
        
        for probe_key, probe_data in hashed_probe.items():
            # Compute similarity for each matrix
            
            for gallery_subkey, gallery_matrix in gallery_data.items():
                for probe_subkey, probe_matrix in probe_data.items():
                    similarity = compute_similarity(gallery_matrix, probe_matrix)
                    
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_key = probe_key
            
        if best_match_score >= threshold:
            matches[gallery_key] = best_match_key
        else:
            matches[gallery_key] = "No match found"

    return matches


# def verify_users(hashed_gallery, hashed_probe, threshold = 0.5):
#     """
#     Compare hashed_gallery and hashed_probe to find the best match for each probe.
    
#     Parameters:
#     - hashed_gallery: A dictionary of hashed gallery templates
#     - hashed_probe: A dictionary of hashed probe templates
#     - threshold: A similarity threshold for determining a match
    
#     Returns:
#     - matches: A dictionary mapping each probe to its best gallery match or 'No match found'
#     """

#     matches = {} # To store the best match for each probe

#     #Loop through each probe hash
#     for probe_key, probe_value in hashed_probe.items():
#         best_match_key = None
#         best_match_score = -1

#         # Convert the probe value to a numpy array 
#         probe_array = np.array(probe_value)

#         # Loop through each gallery hash
#         for gallery_key, gallery_value in hashed_gallery.items():
#             gallery_array = np.array(gallery_value)
 
#             # Ensure the gallery and probe arrays have the same shape before comparison
#             if probe_array.shape != gallery_array.shape:
#                 continue

#             # Calculate similarity using hamming distance
#             similarity = np.sum(probe_array == gallery_array)/probe_array.size

#             #keep track of the best match
#             if similarity > best_match_score:
#                 best_match_score = similarity
#                 best_match_key = gallery_key

#         # Store the best match if it exceeds the similarity threshold
#         if best_match_score >= threshold:
#             matches[probe_key] = best_match_key
#         else:
#             matches[probe_key] = "No match found"

#     return matches

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    features_dict_gallery = extract_features_gallery()
    features_dict_probe = extract_features_probe()

    num_matrices = 5
    matrix_size = 50

    # only run once at the beginnig
    # random_matrix = generate_random_matrices(num_matrices, matrix_size)

    # with open("random_matrix.json", "w") as outfile:
    #     json.dump(random_matrix, outfile)

    # Read the JSON file and load it into a dictionary
    with open('random_matrix.json', 'r') as file:
        random_matrix = json.load(file)

    multiply1 = multiply_structures(features_dict_gallery, random_matrix)
    multiply2 = multiply_structures(features_dict_probe, random_matrix)
    # hash_value_gallery = hash_templates(features_dict_gallery, random_matrices)
    # hash_value_probe = hash_templates(features_dict_probe, random_matrices)
    
    with open("multiply1.json", "w") as outfile:
        json.dump(multiply1, outfile)

    with open("multiply2.json", "w") as outfile:
        json.dump(multiply2, outfile)

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
    # with open('hashed_output_gallery.json', 'r') as gallery_file:
    #     hashed_gallery = json.load(gallery_file)

    # with open('hashed_output_probe.json', 'r') as probe_file:
    #     hashed_probe = json.load(probe_file)

    #perform user verification
    matches = find_best_matches(multiply1, multiply2, threshold=0.8)
    # print(matches)
    #output results
    # for probe, gallery in matches.items():
    #     print(f"Probe {probe} matches with Gallery {gallery}")
    with open("matches.json", "w") as outfile:
        json.dump(matches, outfile)
