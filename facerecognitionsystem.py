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


# Main function
if __name__ == "__main__":
    
    extract_images("Dataset/UTKFace1.zip")

    feature_dict_utkface_file = "utkface.json"

    if not os.path.isfile(feature_dict_utkface_file):
        features_dict_utkface = extract_features("Dataset/UTKFace1", 1611, ['.jpg'])
        with open(feature_dict_utkface_file, "w") as outfile:
            json.dump(features_dict_utkface, outfile)
    else:
        with open(feature_dict_utkface_file, "r") as file:
            features_dict_gallery = json.load(file)


