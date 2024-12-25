# BioAuth-with-LSH

This project implements a **secure face recognition system** using **SimHash** a variant of **Locality Sensitive Hashing (LSH)**. The architecture is designed to preprocess face images, generate hashed templates using SimHash, calculate similarity scores using Hamming distance, and evaluate system performance with visualizations like ROC curves and histograms.

---

## Table of Contents

- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Steps](#steps)
- [Contributing](#contributing)
- [License](#license)

---

## Project Architecture

The project is modularized into the following components:

1. **Classes**: Contains classes for essential operations:
   - `zip_extractor.py`: Extracts zip files.
   - `feature_extractor.py`: Extracts features of face images.
   - `random_vector_generator.py`: Generates random vectors.
   - `hasher.py`: Implements SimHash for hashing.
   - `similarity_calculator.py`: Calculates similarity using Hamming distance.
   - `score_matrix_generator.py`: Generates the score matrix.
   - `visualizer.py`: Generates histograms and ROC curves for evaluation.
2. **Dataset**: Contains datasets as zip files.
3. **Files**: Stores files generated during preprocessing, hashing, and score matrix generation.
4. **Output**: Contains the ROC curve and histogram generated during evaluation.
5. **Old**: Contains legacy files for previous SimHash implementations.
6. **main.py**: Main function that integrates and executes the pipeline.
7. **README.md**: Project documentation.
8. **.gitignore**: Specifies files to be ignored by Git.

---

## Installation

1. **Install Python**:
   - Ensure Python 3.8 or above is installed on your system. You can download it [here](https://www.python.org/).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/BioAuth-with-LSH.git
   cd BioAuth-with-LSH

3. **Set up a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

## Steps

1. Ensure Python is installed on your system.
2. **Prepare Dataset**: Place your dataset as zip files under the **Dataset/** folder.
3. **Run the main script:**
    ```bash
    python main.py
4. **Access the outputs:**
    * Processed files in Files/.
    * Visualizations in Output/.

