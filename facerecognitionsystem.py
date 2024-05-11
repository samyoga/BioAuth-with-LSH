from zipfile import ZipFile
import os
import tqdm
import cv2

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
        features_dict[i].append(images_gallery)
    return features_dict

# function to verify users
def verify_user():
    return 0

# Main function
if __name__ == "__main__":
    extract_images("GallerySet.zip")
    extract_images("ProbeSet.zip")

    gallery_data = "GallerySet"
    features_dict = enroll_user(gallery_data)
    print(features_dict)
    probe_data = "ProbeSet"
