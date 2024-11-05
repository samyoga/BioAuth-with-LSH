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

            # Ensure vector is 2500 elements long
            if len(input_vector) != 2500:
                input_vector = np.pad(input_vector, (0, 2500 - len(input_vector)), 'constant')

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

def hamming_similarity(probe_vector, gallery_vector):
    # """
    # Compute similarity between two vectors using hamming distance
    # """
    # probe_array = np.array(probe_vector)
    # gallery_array = np.array(gallery_vector)

    # distance = np.sum(probe_array != gallery_array)

    # # Compute similarity as 1-(distance/total_bits)
    # total_bits = len(probe_vector)
    # similarity = 1 - (distance/total_bits)

    # return similarity

    """
    Compute similarity between two vectors using Hamming distance, handling different vector lengths by truncation.
    """
    # Truncate vectors to the length of the shorter one
    min_length = min(len(probe_vector), len(gallery_vector))
    probe_array = np.array(probe_vector[:min_length])
    gallery_array = np.array(gallery_vector[:min_length])

    # Calculate Hamming distance
    distance = np.sum(probe_array != gallery_array)

    # Compute similarity as 1-(distance/total_bits)
    similarity = 1 - (distance / min_length)

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

    for gallery_image in range(1611):
        for probe_image in range(3222):
            # if the image from the gallery dataset matches with image from probe dataset, consider the score from the matrix as genuine score else imposter
            if probe_image == 2*gallery_image + 1: 
                genuine_scores.append(score_matrix[probe_image][gallery_image])
            else : 
                impostor_scores.append(score_matrix[probe_image][gallery_image])

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

def compute_cmc(score_matrix, num_probes, num_gallery):
    cmc_scores = []
    
    for m in range(1, num_gallery + 1):
        correct_matches = 0
        
        for probe_index in range(num_probes):
            # Sort gallery scores for the current probe in descending order
            sorted_gallery_indices = np.argsort(score_matrix[probe_index])[::-1]
            
            # The genuine match for this probe is at gallery_index = probe_index // 2 (since each subject has 2 probe images)
            genuine_match_index = probe_index // 2  # Adjust this based on your dataset structure
            
            # Check if the genuine match is within the top 'm' ranked gallery images
            if genuine_match_index in sorted_gallery_indices[:m]:
                correct_matches += 1
        
        # Calculate the identification rate for rank 'm'
        cmc_scores.append((correct_matches / num_probes) * 100)
    
    return cmc_scores

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

def calculate_min_max_val_rank(cmc_val):
    low_val = min(cmc_val)
    high_val = max(cmc_val)
    low_rank = cmc_val.index(low_val) + 1
    high_rank = cmc_val.index(high_val) + 1
    
    return low_val, high_val, low_rank, high_rank

# Main function
if __name__ == "__main__":
    
    extract_images("Dataset/UTKGallery.zip")
    extract_images("Dataset/UTKProbe.zip")

    feature_dict_utkgallery_file = "utkgallery.json"
    feature_dict_utkprobe_file = "utkprobe.json"

    if not os.path.isfile(feature_dict_utkgallery_file):
        features_dict_utkgallery = extract_features("Dataset/UTKGallery", 1611, ['_img1.jpg'])
        with open(feature_dict_utkgallery_file, "w") as outfile:
            json.dump(features_dict_utkgallery, outfile)
    else:
        with open(feature_dict_utkgallery_file, "r") as file:
            features_dict_utkgallery = json.load(file)

    if not os.path.isfile(feature_dict_utkprobe_file):
        features_dict_utkprobe = extract_features("Dataset/UTKProbe", 1611, ['_img2.jpg', '_img3.jpg'])
        with open(feature_dict_utkprobe_file, "w") as outfile:
            json.dump(features_dict_utkprobe, outfile)
    else:
        with open(feature_dict_utkprobe_file, "r") as file:
            features_dict_utkprobe = json.load(file)


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

    hashed_enrolled_users_file = "hashed_enrolled.json"
    hashed_probe_users_file = "hashed_probe_users.json"

    if not os.path.isfile(hashed_enrolled_users_file):
        hashed_enrolled = generate_matrix_vector_multiplication(features_dict_utkgallery, random_matrix)
        with open(hashed_enrolled_users_file, "w") as outfile:
            json.dump(hashed_enrolled, outfile)
    else:
        with open(hashed_enrolled_users_file, "r") as file:
            hashed_enrolled = json.load(file)

    if not os.path.isfile(hashed_probe_users_file):
        hashed_probe_users = generate_matrix_vector_multiplication(features_dict_utkprobe, random_matrix)
        with open(hashed_probe_users_file, "w") as outfile:
            json.dump(hashed_probe_users, outfile)
    else:
        with open(hashed_probe_users_file, "r") as file:
            hashed_probe_users = json.load(file)      

    #Generate score matrix
    score_matrix_file = "score_matrix.json"
    if not os.path.isfile(score_matrix_file):
        score_matrix = generate_score_matrix(hashed_enrolled, hashed_probe_users)
        with open(score_matrix_file, "w") as outfile:
            json.dump(score_matrix.tolist(), outfile)
    else:
        with open(score_matrix_file, 'r') as file:
            score_matrix = json.load(file)

   
    score_matrix = np.array(score_matrix)
    print(score_matrix)
    print(f"Score matrix shape: {score_matrix.shape}")

    genuine_scores, impostor_scores = extract_genuine_impostor_scores(score_matrix)
    print ("genuine", genuine_scores)
    # # print("impostor", impostor_scores)

    d_prime = decidability_index(impostor_scores, genuine_scores)
    print ("Decidability index score for this face recognition system is", d_prime)

    # Save histogram plot if not already saved
    plot_histogram_file = "histogram.png"
    plot_histogram(genuine_scores, impostor_scores)
    # save_plot(plot_histogram_file, plot_histogram, genuine_scores, impostor_scores)

    # Save ROC curve plot if not already saved
    plot_roc_curve_file = "roc_curve.png"
    plot_roc_curve(genuine_scores, impostor_scores)
    # save_plot(plot_roc_curve_file, plot_roc_curve, genuine_scores, impostor_scores)

    plot_cmc_curve_file = "cmc.png"
    cmc_val = compute_cmc(score_matrix, 200, 100)
    low_val, high_val, low_rank, high_rank = calculate_min_max_val_rank(cmc_val)
    print("The lowest value of the system is", low_val, "at Rank-", low_rank)
    print("The highest value of the system is", high_val, "at Rank-", high_rank)
    plot_cmc_curve(cmc_val)
    # save_plot(plot_cmc_curve_file, plot_cmc_curve, cmc_val)