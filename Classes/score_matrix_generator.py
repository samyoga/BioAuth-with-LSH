import numpy as np
from similarity_calculator import SimilarityCalculator

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
                if probe_key.split('_')[0] == gallery_key.split('_')[0]:
                    self.genuine_scores.append(similarity)
                else:
                    self.impostor_scores.append(similarity)
        
        return score_matrix