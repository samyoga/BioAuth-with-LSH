from Classes.zip_extractor import ZipExtractor
from Classes.feature_extractor import FeatureExtractor
from Classes.random_vector_generator import RandomVectorGenerator
from Classes.hasher import Hasher
from Classes.score_matrix_generator import ScoreMatrixGenerator
from Classes.visualizer import Visualizer
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_or_generate(file_path, generation_function, *args):
    """Utility to load data from a file or generate it using a provided function."""
    if not os.path.isfile(file_path):
        data = generation_function(*args)

        # Convert NumPy arrays to lists if necessary
        if isinstance(data, np.ndarray):
            data = data.tolist()

        with open(file_path, "w") as outfile:
            json.dump(data, outfile)
        return data
    else:
        with open(file_path, "r") as file:
            return json.load(file)
        
def save_plot(plot_file, plot_func, *args):
    if not os.path.isfile(plot_file):
        plot_func(*args)  # Call the plotting function with arguments
        plt.savefig(plot_file)
        print(f"Plot saved as {plot_file}")
    else:
        print(f"{plot_file} already exists. Plot not saved.")

def main():
    #Initialize paths
    zip_path_gallery = "Dataset/GallerySet.zip"
    zip_path_probe = "Dataset/ProbeSet.zip"
    extract_path_gallery = "Dataset/GallerySet"
    extract_path_probe = "Dataset/ProbeSet"

    #Extract gallery data
    extractor_gallery = ZipExtractor(zip_path_gallery, extract_path_gallery)
    extractor_gallery.extract()

    #Extract probe data
    extractor_probe = ZipExtractor(zip_path_probe, extract_path_probe)
    extractor_probe.extract()

    #Feature Extraction if not already done
    # Create an instance of FeatureExtractor
    feature_extractor = FeatureExtractor()
    feature_dict_gallery_file = "Files/gallery.json"
    features_dict_gallery = load_or_generate(feature_dict_gallery_file, feature_extractor.extract_features, extract_path_gallery, 100, ['_img1.pgm'])

    feature_dict_probe_file = "Files/probe.json"
    features_dict_probe = load_or_generate(feature_dict_probe_file, feature_extractor.extract_features, extract_path_probe, 100, ['_img2.pgm', '_img3.pgm'])

    #Generate random vectors if not already done
    #Create an instance of RandomVectorGenerator
    random_vector_generator = RandomVectorGenerator()
    random_vectors_file = "Files/random_vector.json"
    random_vectors = load_or_generate(random_vectors_file, random_vector_generator.generate)

    #Generate hashed gallery and probe data if not already done
    hashed_gallery_file = "Files/hashed_gallery.json"
    hashed_gallery = load_or_generate(hashed_gallery_file, Hasher.generate_hash, features_dict_gallery, random_vectors)

    hashed_probe_file = "Files/hashed_probe,json"
    hashed_probe = load_or_generate(hashed_probe_file, Hasher.generate_hash, features_dict_probe, random_vectors)

    #Generate score matrix if not already done
    #Create an instance of ScoreMatrixGenerator
    score_matrix_generator = ScoreMatrixGenerator()
    score_matrix_file = "Files/score_matrix.json"
    score_matrix = load_or_generate(score_matrix_file, score_matrix_generator.generate, hashed_gallery, hashed_probe)
    score_matrix = np.array(score_matrix)

    #Extract genuine and impostor scores directly from the above instance
    genuine_scores = score_matrix_generator.genuine_scores
    impostor_scores = score_matrix_generator.impostor_scores

    #Computing decidability index
    d_prime = score_matrix_generator.decidability_index()
    print("Decidability index score for this face recognition system is ", d_prime)

    #Plot histogram if not done
    plot_histogram_file = "Output/histogram.png"
    # Visualizer.plot_histogram(genuine_scores, impostor_scores)
    save_plot(plot_histogram_file, Visualizer.plot_histogram, genuine_scores, impostor_scores)

    #Plot ROC curve if not done
    plot_ROC_file = "Output/ROC.png"
    save_plot(plot_ROC_file, Visualizer.plot_roc_curve, genuine_scores, impostor_scores)
    
# Main function
if __name__ == "__main__":
    main()