from zipfile import ZipFile
import cv2
import json
import numpy as np
import re
import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# function to extract zip files
def extract_images(zip_file):
    with ZipFile(zip_file, 'r') as zip_extract:
        zip_extract.extractall()

#Function to extract images from a dataset
def extract_features(directory, num_subjects, image_pattern):
    """
    Extracts binary features from images of subjects, either for gallery or probe data.

    Args:
        directory (str): The director containing the images (GallerySet and ProbeSet).   
        num_subjects (int): The number of subjects in the dataset.
        image_pattern (list): A list of image file patterns for each subject. 
                              ['_img1.pgm'] for gallery, ['_img2.pgm', '_img3.pgm'] for probe

    Returns:
        dict: A dictionary with key is the image index and the value is the binary image feature.
    """

    features_dict = {}
    counter = 1

    for i in range(1, num_subjects + 1):
        for pattern in image_pattern:
            #Construct image file path
            image_file = f"{directory}/subject{i}{pattern}"

            #read images and convert to grayscale image
            images = cv2.imread(f"GallerySet/subject{i}_img1.pgm", cv2.IMREAD_GRAYSCALE)
            # grayscale_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

            # Ensure the image is 50x50 pixels
            if images.shape != (50, 50):
                raise ValueError("Image must be 50x50 pixels")

            # Flatten the 2D image (50x50) to a 1D vector (2500 elements)
            input_vector = images.flatten()
            # #converting grayscale to binary images using adaptive thresholding
            # # apply adapting thresholding using initial values
            # block_size = 11
            # constant = 2

            # binary_image = cv2.adaptiveThreshold(
            #     grayscale_image,
            #     255,
            #     cv2.ADAPTIVE_THRESH_MEAN_C,
            #     cv2.THRESH_BINARY,
            #     block_size,
            #     constant
            # )

            # # Convert binary images to 0 and 1
            # binary_image = np.where(binary_image == 255,1,0)

            #Store the binary image in a dictionary
            features_dict[counter]=input_vector.tolist()
            counter += 1

    return features_dict


# # function to extract features of gallery data
# def extract_features_gallery():
#     features_dict = {}
#     for i in range(1, 101):
#         #read images from the gallery set and convert to grayscale image
#         images_gallery = cv2.imread(f"GallerySet/subject{i}_img1.pgm")
#         grayscale_image = cv2.cvtColor(images_gallery, cv2.COLOR_BGR2GRAY)

#         #converting grayscale to binary images using adaptive thresholding
#         # apply adapting thresholding using initial values
#         block_size = 11
#         constant = 2

#         binary_image = cv2.adaptiveThreshold(
#             grayscale_image,
#             255,
#             cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,
#             block_size,
#             constant
#         )

#         # Convert binary image to 0s and 1s
#         binary_image = np.where(binary_image == 255, 1, 0)

#         if i not in features_dict:
#             features_dict[i] = []
       
#         features_dict[i].append(binary_image.tolist())
    
#     for key, value_list in features_dict.items():
#         # Update the dictionary with the modified value
#         features_dict[key] = value_list[0]

#     # print(features_dict)
#     return features_dict

# # function to extract features of probe data
# def extract_features_probe():
#     features_dict = {}
#     counter = 1
#     for i in range(1, 101):
#         #read images from the probe set
#         images_probe_1 = cv2.imread(f"ProbeSet/subject{i}_img2.pgm")
#         images_probe_2 = cv2.imread(f"ProbeSet/subject{i}_img3.pgm")
#         grayscale_probeimage1 = cv2.cvtColor(images_probe_1, cv2.COLOR_BGR2GRAY)
#         grayscale_probeimage2 = cv2.cvtColor(images_probe_2, cv2.COLOR_BGR2GRAY)

#         #converting grayscale to binary images using adaptive thresholding
#         # apply adapting thresholding using initial values
#         block_size = 11
#         constant = 2

#         binary_probeimage1 = cv2.adaptiveThreshold(
#             grayscale_probeimage1,
#             255,
#             cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,
#             block_size,
#             constant
#         )

#         binary_probeimage2 = cv2.adaptiveThreshold(
#             grayscale_probeimage2,
#             255,
#             cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,
#             block_size,
#             constant
#         )

#         # Convert binary image to 0s and 1s
#         binary_probeimage1 = np.where(binary_probeimage1 == 255, 1, 0)
#         binary_probeimage2 = np.where(binary_probeimage2 == 255, 1, 0)

#         if i not in features_dict:
#             features_dict[i] = []
        
#         features_dict[counter] = binary_probeimage1.tolist()
#         counter +=1
#         features_dict[counter] = binary_probeimage2.tolist()
#         counter +=1

#     # print(features_dict)
    # return features_dict

# function to generate random matrices
def generate_random_matrices(num_matrices, matrix_size):
    #Initialize a dictionary to store matrices
    binary_matrices_dict = {}

    for i in range(1, num_matrices+1):
        # Generate a random matrix with matrix_size
        # Each element of the matrix is a list with 50 random elements
        random_matrix = np.random.randint(-100, 101, size=(matrix_size, matrix_size))

        #convert matrix to list and store in dictionary
        binary_matrices_dict[i] = random_matrix.tolist()
        
    # print(binary_matrices_dict)
    return binary_matrices_dict

# Function to flatten lists within the nested dictionary
def flatten_lists(data):
    flattened_data = {}
    
    # Iterate through the first level of keys
    for key1 in data:
        sub_dict = data[key1]
        flattened_data[key1] = {}
        
        # Iterate through the second level of keys
        for key2 in sub_dict:
            list_of_lists = sub_dict[key2]
            
            # Flatten the list of lists
            flattened_list = [item for sublist in list_of_lists for item in sublist]
            
            # Store the flattened list in the new structure
            flattened_data[key1][key2] = flattened_list
    
    return flattened_data

def generate_matrix_vector_multiplication(feature_template, random_matrix):
    # Intitalize dictionary to store the results
    hashed_result = {}
    # #Intialize flattened data
    # flattened_data = {}

    random_matrix_data = np.array(random_matrix['1'])
    assert random_matrix_data.shape == (2500, 2500) # check the size of random projection matrix

    # For each feature vector, multiply it with the random projection matrix
    for subject_id, feature_vector in feature_template.items():
        feature_vector = np.array(feature_vector)
        assert len(feature_vector) == 2500

        #multiply matrix with input vector. (2500 * 2500) with (2500*1)
        projected_feature_vector = np.dot(random_matrix_data, feature_vector)
        # apply sign function
        signed_result = np.sign(projected_feature_vector)

        hashed_result[subject_id] = signed_result.tolist()

    # # Iterate through each key in first structure
    # for key1, matrix1 in structure1.items():
    #     matrix1 = np.array(matrix1) # convert list to numpy array

    #     # create a sub-dictionary for each key in structure1
    #     result[key1] = {}

    #     #Iterate through each key in the second structure
    #     for key2, matrix2 in structure2.items():
    #         matrix2 = np.array(matrix2) # convert list to numpy array

    #         # # Ensure both matrices are 50*50 before multiplication 
    #         # if matrix1.shape == matrix2.shape:
    #         #     #Perform element-wise multiplication (use np.dot for matrix multiplication)
    #         #     multiplied_matrix = np.multiply(matrix1, matrix2)

    #         #     #Apply the sign function to the multiplied matrix
    #         #     signed_matrix = np.sign(multiplied_matrix)

    #         #     # Convert the result back to list and store in the result dictionary
    #         #     result[key1][key2] = signed_matrix.tolist()

    #         # else:
    #         #     print(f"Matrix dimensions don't match for {key1} and {key2}.")

    # # flattened_data = flatten_lists(result)

    return hashed_result

def compute_similarity(matrix1, matrix2):
    """
    Compute the similarity between two matrices.
    For signed matrices, you can use Hamming distance or another suitable metric.
    """
    # # Flatten the matrices to compare element-wise
    # flat_matrix1 = np.array(matrix1).flatten()
    # flat_matrix2 = np.array(matrix2).flatten()

    # print(flat_matrix1)

    # Compute the dot product
    dot_product = np.dot(matrix1, matrix2)

    # Compute the magnitudes (Euclidean norms)
    magnitude1 = np.linalg.norm(matrix1)
    magnitude2 = np.linalg.norm(matrix2)

    # Handle division by zero case
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # If either vector is all zeros, return similarity of 0

    # Compute the cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim

def hamming_similarity(probe_vector, gallery_vector):
    """
    Compute similarity between two vectors using hamming distance
    """
    probe_array = np.array(probe_vector)
    gallery_array = np.array(gallery_vector)

    distance = np.sum(probe_array != gallery_array)

    # Compute similarity as 1-(distance/total_bits)
    total_bits = len(probe_vector)
    similarity = 1 - (distance/total_bits)

    return similarity

def generate_score_matrix(hashed_gallery, hashed_probe):
    num_probes = len(hashed_probe)  # Number of probe images (200)
    num_gallery = len(hashed_gallery)  # Number of gallery images (100)
    
    # Initialize the score matrix with zeros
    score_matrix = np.zeros((num_probes, num_gallery))
    
    # Loop through each probe and gallery image
    for probe_key in hashed_probe:
        for gallery_key in hashed_gallery:
            # Get the feature vectors for the current probe and gallery images
            probe_vector = np.array(hashed_probe[probe_key])
            gallery_vector = np.array(hashed_gallery[gallery_key])
            
            # Compute the similarity score
            similarity = hamming_similarity(probe_vector, gallery_vector)
            
            # Update the score matrix
            probe_index = int(probe_key) - 1  # Convert key to index (0-based)
            gallery_index = int(gallery_key) - 1  # Convert key to index (0-based)
            score_matrix[probe_index, gallery_index] = similarity

    return score_matrix

def extract_genuine_impostor_scores(score_matrix):
    genuine_scores = []
    impostor_scores = []

    num_probes, num_gallery = score_matrix.shape

    for gallery_image in range(100):
        for probe_image in range(200):
            # if the image from the gallery dataset matches with image from probe dataset, consider the score from the matrix as genuine score else imposter
            if probe_image == 2*gallery_image or probe_image == 2*gallery_image +1:
                genuine_scores.append(score_matrix[probe_image][gallery_image])
            else:
                impostor_scores.append(score_matrix[probe_image][gallery_image])

    # # Genuine scores (Diagonal for corresponding matches)
    # for i in range(min(num_probes, num_gallery)):  # Only as many genuine matches as there are subjects
    #     genuine_scores.append(score_matrix[i, i])  # e.g., probe1 with gallery1, probe2 with gallery2

    # # Impostor scores (Off-diagonal for non-matches)
    # for i in range(num_probes):
    #     for j in range(num_gallery):
    #         if i != j:  # Skip the genuine matches
    #             impostor_scores.append(score_matrix[i, j])

    return genuine_scores, impostor_scores

def plot_histogram(genuine_scores, impostor_scores):
    plt.figure(figsize=(10, 6))

    # Plot histogram for genuine scores
    plt.hist(genuine_scores, density=True, alpha=0.6, label="Genuine Scores", color="green", edgecolor='black')

    # Plot histogram for impostor scores
    plt.hist(impostor_scores, density=True, alpha=0.6, label="Impostor Scores", color="red", edgecolor='black')

    # Add labels and legend
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Genuine and Impostor Scores')
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()

def decidability_index(impostor_scr, genuine_scr):
    mu_0 = np.mean(impostor_scr)
    sigma_0 = np.std(impostor_scr)
    mu_1 = np.mean(genuine_scr)
    sigma_1 = np.std(genuine_scr)
    d_prime = np.sqrt(2) * abs(mu_1 - mu_0) / np.sqrt(sigma_1**2 + sigma_0**2)

    return d_prime

def compute_cmc(genuine_scores, similarity_matrix):
    cmc_scr = []

    for m in range(1, len(genuine_scores) +1):
        count = 0
        for s in range(len(genuine_scores)):
            sorted_cs = sorted(similarity_matrix[s], reverse=True)
            if genuine_scores[s] in sorted_cs[:m]:
                count +=1
        cmc_scr.append((count/len(genuine_scores))*100)

    return cmc_scr

def plot_cmc_curve(cmc_val):
    plt.figure(figsize=(12, 5))

    plt.plot(range(1, len(cmc_val) + 1), cmc_val, label='CMC Curve')
    plt.ylabel("Identification Rate (%)")
    plt.xlabel("Rank-t")
    plt.title("Cumulative Match Characteristic Curve")
    # plt.grid(True)  # Add grid lines for better readability
    plt.legend()
    plt.show()

def plot_roc_curve(genuine_scores, impostor_scores):
    threshold_values = np.linspace(0,1,1000)
    #Initialize empty list to store FAR and FRR values
    false_acceptance_rate = []
    false_rejection_rate = []

    #Calculating FAR and FRR for each threshold value
    for threshold in threshold_values:
        FA_sum = np.sum(impostor_scores >= threshold)
        far_val = FA_sum/len(impostor_scores)

        FR_sum = np.sum(genuine_scores < threshold)
        frr_val = FR_sum/len(genuine_scores)

        false_acceptance_rate.append(far_val)
        false_rejection_rate.append(frr_val)

    #plotting the ROC curve
    plt.plot(false_acceptance_rate, false_rejection_rate, color='red')
    plt.plot([0, 1], [0, 1], color='yellow', lw=2, linestyle='--')
    plt.title('Receiver Operating Curve')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.show()

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    features_dict_gallery = extract_features("GallerySet", 100, ['_img1.pgm'])
    features_dict_probe = extract_features("ProbeSet", 100, ['_img2.pgm', '_img3.pgm'])

    # # dump dictionary to JSON
    # with open("probe.json", "w") as outfile:
    #     json.dump(features_dict_probe, outfile)

    # # dump dictionary to JSON
    # with open("gallery.json", "w") as outfile:
    #     json.dump(features_dict_gallery, outfile)

    num_matrices = 1
    matrix_size = 2500

    # only run once at the beginnig
    # random_matrix = generate_random_matrices(num_matrices, matrix_size)

    # with open("random_matrix.json", "w") as outfile:
    #     json.dump(random_matrix, outfile)

    # Read the JSON file and load it into a dictionary
    with open('random_matrix.json', 'r') as file:
        random_matrix = json.load(file)

    hashed_gallery = generate_matrix_vector_multiplication(features_dict_gallery, random_matrix)
    hashed_probe = generate_matrix_vector_multiplication(features_dict_probe, random_matrix)
    
    # with open("hashed_gallery.json", "w") as outfile:
    #     json.dump(hashed_gallery, outfile)

    # with open("hashed_probe.json", "w") as outfile:
    #     json.dump(hashed_probe, outfile)

    # #perform user verification
    # matches = find_best_matches(multiply1, multiply2, threshold=0.8)
    
    # with open("matches.json", "w") as outfile:
    #     json.dump(matches, outfile)

    # #generate score matrix
    score_matrix = generate_score_matrix(hashed_gallery, hashed_probe)
    # dump score matrix to a json file
    # with open("score_matrix.json", "w") as outfile:
    #     json.dump(score_matrix.tolist(), outfile)

    genuine_scores, impostor_scores = extract_genuine_impostor_scores(score_matrix)
    # print ("genuine", genuine_scores)
    # print("impostor", impostor_scores)
    # d_prime = decidability_index(impostor_scores, genuine_scores)
    # print ("Decidability index score for this face recognition system is", d_prime)
    # plot_histogram(genuine_scores, impostor_scores)
    # cmc_val = compute_cmc(genuine_scores, score_matrix)
    # plot_cmc_curve(cmc_val)

    plot_roc_curve(genuine_scores, impostor_scores)