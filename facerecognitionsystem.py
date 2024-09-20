from zipfile import ZipFile
import cv2
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import os
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
        random_matrix = np.random.randint(-100, 101, size=(matrix_size, matrix_size))

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

def compute_cosine_similarity(matrix1, matrix2):
    """
    Compute the similarity between two matrices.
    For signed matrices, you can use Hamming distance or another suitable metric.
    """

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
    num_probes = len(hashed_probe)  # Number of probe images (e.g., 200)
    num_gallery = len(hashed_gallery)  # Number of gallery images (e.g., 100)
    
    # Initialize the score matrix with zeros
    score_matrix = np.zeros((num_probes, num_gallery))
    
    # Loop through each probe and gallery image
    for probe_key in hashed_probe:
        for gallery_key in hashed_gallery:
            # Get the feature vectors for the current probe and gallery images
            probe_vector = np.array(hashed_probe[probe_key])
            gallery_vector = np.array(hashed_gallery[gallery_key])

            # # Ensure vectors are of the correct length
            # assert len(probe_vector) == len(gallery_vector), "Probe and Gallery vectors must be of the same length."

            # Compute the similarity score using cosine similarity
            similarity = hamming_similarity(probe_vector, gallery_vector)
            
            # Convert string keys to zero-based indices
            probe_index = int(probe_key) - 1  # Convert key to index (0-based)
            gallery_index = int(gallery_key) - 1  # Convert key to index (0-based)

            # Update the score matrix
            score_matrix[probe_index, gallery_index] = similarity

    return score_matrix


def extract_genuine_impostor_scores(score_matrix):
    genuine_scores = []
    impostor_scores = []

    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if i==j:
                genuine_scores.append(score_matrix[i][j])
            else:
                impostor_scores.append(score_matrix[i][j])

    return genuine_scores, impostor_scores

def plot_histogram(genuine_scores, impostor_scores):
    plt.figure(figsize=(10, 6))

    # Check if both score lists are populated
    if len(genuine_scores) == 0:
        print("Warning: Genuine scores are empty.")
    if len(impostor_scores) == 0:
        print("Warning: Impostor scores are empty.")

    # Plot histogram for genuine scores
    plt.hist(genuine_scores, density=True, bins=30, alpha=0.6, label="Genuine Scores", color="green", edgecolor='black')

    # Plot histogram for impostor scores
    plt.hist(impostor_scores, density=True, bins=30, alpha=0.6, label="Impostor Scores", color="red", edgecolor='black')

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

    feature_dict_gallery_file = "gallery.json"
    feature_dict_probe_file = "probe.json"

    if not os.path.isfile(feature_dict_gallery_file):
        features_dict_gallery = extract_features("GallerySet", 100, ['_img1.pgm'])
        with open(feature_dict_gallery_file, "w") as outfile:
            json.dump(features_dict_gallery, outfile)
    else:
        with open(feature_dict_gallery_file, "r") as file:
            features_dict_gallery = json.load(file)

    if not os.path.isfile(feature_dict_probe_file):
        feature_dict_probe = extract_features("ProbeSet", 100, ['_img2.pgm', '_img3.pgm'])
        with open(feature_dict_probe_file, "w") as outfile:
            json.dump(feature_dict_probe, outfile)
    else:
        with open(feature_dict_probe_file, "r") as file:
            feature_dict_probe = json.load(file)

    num_matrices = 256
    matrix_size = 50
    random_matrix_file = "random_matrix.json"

    if not os.path.isfile(random_matrix_file):
        random_matrix = generate_random_matrices(num_matrices, matrix_size)
        with open(random_matrix_file, "w") as outfile:
            json.dump(random_matrix, outfile)

    else:
        with open(random_matrix_file, "r") as file:
            random_matrix = json.load(file)

    
    hashed_gallery_file = "hashed_gallery.json"
    hashed_probe_file = "hashed_probe.json"

    if not os.path.isfile(hashed_gallery_file):
        hashed_gallery = generate_matrix_vector_multiplication(features_dict_gallery, random_matrix)
        with open(hashed_gallery_file, "w") as outfile:
            json.dump(hashed_gallery, outfile)
    else:
        with open(hashed_gallery_file, "r") as file:
            hashed_gallery = json.load(file)

    if not os.path.isfile(hashed_probe_file):
        hashed_probe = generate_matrix_vector_multiplication(feature_dict_probe, random_matrix)
        with open(hashed_probe_file, "w") as outfile:
            json.dump(hashed_probe, outfile)
    else:
        with open(hashed_probe_file, "r") as file:
            hashed_probe = json.load(file)       


    # Generate score matrix

    score_matrix_file = "score_matrix.json"
    if not os.path.isfile(score_matrix_file):
        score_matrix = generate_score_matrix(hashed_gallery, hashed_probe)
        with open(score_matrix_file, "w") as outfile:
            json.dump(score_matrix.tolist(), outfile)
    else:
        with open(score_matrix_file, 'r') as file:
            score_matrix = json.load(file)

    # print(score_matrix)
    score_matrix = np.array(score_matrix)

    genuine_scores, impostor_scores = extract_genuine_impostor_scores(score_matrix)
    print ("genuine", genuine_scores)
    # # print("impostor", impostor_scores)

    d_prime = decidability_index(impostor_scores, genuine_scores)
    print ("Decidability index score for this face recognition system is", d_prime)

    # Save histogram plot if not already saved
    plot_histogram_file = "histogram.png"
    plot_histogram(genuine_scores, impostor_scores)
    plt.savefig(plot_histogram_file)

    # Save ROC curve plot if not already saved
    plot_roc_curve_file = "roc_curve.png"
    plot_roc_curve(genuine_scores, impostor_scores)
    plt.savefig(plot_roc_curve_file)

    plot_cmc_curve_file = "cmc.png"
    cmc_val = compute_cmc(genuine_scores, score_matrix)
    plot_cmc_curve(cmc_val)
    plt.savefig(plot_cmc_curve_file)

