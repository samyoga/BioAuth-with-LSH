import numpy as np

class Hasher:
    @staticmethod
    def generate_hash(feature_template, random_matrix):
        # Intitalize dictionary to store the results
        hashed_result = {}
      
        for random_vector_id, random_vector in random_matrix.items():
            assert len(random_vector) == 2500

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