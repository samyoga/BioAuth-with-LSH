import numpy as np
from Classes.similarity_calculator import SimilarityCalculator

class ScoreMatrixGenerator:
    def __init__(self):
        self.genuine_scores = []
        self.impostor_scores = []
    
    def generate(self, hashed_gallery, hashed_probe):
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

                # Compute the similarity score using cosine similarity
                similarity = SimilarityCalculator.compute_similarity(probe_vector, gallery_vector)
                
                # Convert string keys to zero-based indices
                probe_index = int(probe_key) - 1  # Convert key to index (0-based)
                gallery_index = int(gallery_key) - 1  # Convert key to index (0-based)

                # Update the score matrix
                score_matrix[probe_index, gallery_index] = similarity
                        
        for gallery_image in range(num_gallery):
            for probe_image in range(num_probes):
                # if the image from the gallery dataset matches with image from probe dataset, consider the score from the matrix as genuine score else imposter
                if probe_image == 2*gallery_image or probe_image == 2*gallery_image + 1: 
                    self.genuine_scores.append(score_matrix[probe_image][gallery_image])
                else : 
                    self.impostor_scores.append(score_matrix[probe_image][gallery_image])
        
        return score_matrix
    
    def decidability_index(self):
        mu_0 = np.mean(self.impostor_scores)
        sigma_0 = np.std(self.impostor_scores)
        mu_1 = np.mean(self.genuine_scores)
        sigma_1 = np.std(self.genuine_scores)
        d_prime = np.sqrt(2) * abs(mu_1 - mu_0) / np.sqrt(sigma_1**2 + sigma_0**2)

        return d_prime