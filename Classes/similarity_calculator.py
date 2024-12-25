import numpy as np

class SimilarityCalculator:
    @staticmethod
    def compute_similarity(probe_vector, gallery_vector):
        """ 
        Compute similarity between two vectors using hamming distance
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
