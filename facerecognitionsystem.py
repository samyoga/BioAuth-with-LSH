from zipfile import ZipFile
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# function to extract zip files
def extract_images(zip_file):

    with ZipFile(zip_file, 'r') as zip_extract:
        zip_extract.extractall("Dataset")

#Function to extract images from a dataset
def extract_features(directory, num_subjects, image_pattern):

    features_dict = {}
    counter = 1

    for i in range(1, num_subjects + 1):
        for pattern in image_pattern:
            #Construct image file path
            image_file = f"{directory}/subject{i}{pattern}"
           
            #read images
            images = cv2.imread(image_file)
            if images is None: 
                print(f"Warning: Could not load image at {image_file}. Skipping.")
                continue # skip to the next image if loading fails

            #convert to grayscale image
            grayscale_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

            #Resize to 50*50 if necessary
            if grayscale_image.shape !=(50,50):
                grayscale_image = cv2.resize(grayscale_image, (50,50))

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


            # Convert binary images to 0 and 1
            binary_image = np.where(binary_image == 255,1,0)

            # Flatten the 2D image (50x50) to a 1D vector (2500 elements)
            input_vector = binary_image.flatten()

            #Store the image in a dictionary
            features_dict[counter]=input_vector.tolist()
            counter += 1

    return features_dict

# function to generate random matrices
def generate_random_matrices(num_matrices, matrix_size):
    #Initialize a dictionary to store matrices
    binary_matrices_dict = {}

    for i in range(1, num_matrices+1):
        # Generate a random matrix with matrix_size
        # Each element of the matrix is a list with 50 random elements
        # random_matrix = np.random.randint(0, 2, size=(matrix_size, matrix_size))
        random_matrix = np.random.choice([-1, 0, 1], size=(matrix_size, matrix_size))

        random_vector = random_matrix.flatten()

        #convert matrix to list and store in dictionary
        binary_matrices_dict[i] = random_vector.tolist()
        
    # print(binary_matrices_dict)
    return binary_matrices_dict

def generate_matrix_vector_multiplication(feature_template, random_matrix):
    # Intitalize dictionary to store the results
    hashed_result = {}
    # #Intialize flattened data
    # flattened_data = {}

    for random_vector_id, random_vector in random_matrix.items():
        assert len(random_vector) == 2500


    # random_matrix_data = np.array(random_matrix['1'])
    # assert random_matrix_data.shape == (2500, 2500) # check the size of random projection matrix

    # For each feature vector, multiply it with the random projection matrix
    for subject_id, feature_vector in feature_template.items():
        feature_vector = np.array(feature_vector)
        assert len(feature_vector) == 2500

        #Initialize list to store hash code for this subject
        hash_code = []

        # Compute inner product with each random vector and apply sign function
        for random_vector_id, random_vector in random_matrix.items():
            random_vector = np.array(random_vector)

            #compute inner product between two vectors. (1 * 2500) with (2500*1)
            projected_feature_vector = np.dot(feature_vector, random_vector)

            # apply sign function
            signed_result = int(np.sign(projected_feature_vector))

            # append sign result to hash code
            hash_code.append(signed_result)

        hashed_result[subject_id] = hash_code

    return hashed_result

# Main function
if __name__ == "__main__":
    
    extract_images("Dataset/UTKGallery.zip")

    feature_dict_utkgallery_file = "utkgallery.json"

    if not os.path.isfile(feature_dict_utkgallery_file):
        features_dict_utkgallery = extract_features("Dataset/UTKFace1", 1611, ['.jpg'])
        with open(feature_dict_utkgallery_file, "w") as outfile:
            json.dump(features_dict_utkgallery, outfile)
    else:
        with open(feature_dict_utkgallery_file, "r") as file:
            features_dict_utkgallery = json.load(file)


    num_matrices = 1000
    matrix_size = 50
    random_matrix_file = "random_matrix.json"

    if not os.path.isfile(random_matrix_file):
        random_matrix = generate_random_matrices(num_matrices, matrix_size)
        with open(random_matrix_file, "w") as outfile:
            json.dump(random_matrix, outfile)

    else:
        with open(random_matrix_file, "r") as file:
            random_matrix = json.load(file)

    hashed_enrolled_users_file = "hashed_enrolled.json"

    if not os.path.isfile(hashed_enrolled_users_file):
        hashed_enrolled = generate_matrix_vector_multiplication(features_dict_utkface, random_matrix)
        with open(hashed_enrolled_users_file, "w") as outfile:
            json.dump(hashed_enrolled, outfile)
    else:
        with open(hashed_enrolled_users_file, "r") as file:
            hashed_enrolled = json.load(file)