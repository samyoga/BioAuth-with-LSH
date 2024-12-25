import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, image_size=(50,50), block_size=11, constant=2):
        self.image_size = image_size
        self.block_size = block_size
        self.constant = constant

    def preprocess(self, image_path):
        #Read the image in grayscale
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        #Ensure the image is of desired size
        if images.shape != self.image_size:
            raise ValueError(f"Image must be of size {self.image_size}")
        
        #Apply adaptive thresholding
        binary_image = cv2.adaptiveThreshold(
            images,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.block_size,
            C=self.constant
        )

        # Convert binary images to 0 and 1
        binary_image = np.where(binary_image == 255,1,0)

        # Flatten the 2D image (50x50) to a 1D vector (2500 elements)
        input_vector = binary_image.flatten()

        return input_vector
    
    def extract_features(self, directory, num_subjects, image_pattern):
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
        features = {}
        counter = 1

        for i in range(1, num_subjects + 1):
            for pattern in image_pattern:
                # Construct the file path
                image_path = f"{directory}/subject{i}{pattern}"

                #Preprocess and extract features
                try:
                    features[counter] = self.preprocess(image_path).tolist()
                except FileNotFoundError:
                    print(f"file not found:{image_path}")
                except ValueError as e: 
                    print(f"Error processing {image_path} as {e}")

                counter +=1
        
        return features

