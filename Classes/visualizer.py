import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_histogram(genuine_scores, impostor_scores):
        plt.figure(figsize=(14.5, 12))

        # Check if both score lists are populated
        if len(genuine_scores) == 0:
            print("Warning: Genuine scores are empty.")
        if len(impostor_scores) == 0:
            print("Warning: Impostor scores are empty.")

        # Plot histogram for genuine scores
        plt.hist(genuine_scores, density=True, bins=30, alpha=0.6, label="Genuine Scores", color="green", edgecolor='black')
        # Plot histogram for impostor scores
        plt.hist(impostor_scores, density=True, bins=30, alpha=0.6, label="Impostor Scores", color="red", edgecolor='black')
        # Customizing font size and family
        font_properties = {'family': 'Arial', 'weight': 'normal', 'size': 46}
        # Add labels and legend
        plt.xlabel('Score', **font_properties)
        plt.ylabel('Frequency', **font_properties)
        plt.title('Histogram of Genuine and Impostor Scores', **font_properties)
        plt.legend(loc='upper right', prop={'family': 'Arial', 'size': 36})
        # Set font properties for axis ticks (x and y axis)
        plt.xticks(fontsize=28, family='Arial')
        plt.yticks(fontsize=28, family='Arial')

        # Display the plot
        plt.show()
    
    @staticmethod
    def plot_roc_curve(genuine_scores, impostor_scores):
        threshold_values = np.linspace(0,1,100)
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

        # Compute EER
        far = np.array(false_acceptance_rate)
        frr = np.array(false_rejection_rate)
        abs_diff = np.abs(far - frr)
        eer_index = np.argmin(abs_diff)
        eer = (far[eer_index] + frr[eer_index]) / 2  # Average FAR and FRR at EER point
        eer_threshold = threshold_values[eer_index]

        print(f"EER: {eer:.4f}, EER Threshold: {eer_threshold:.4f}")
        
        # Plot ROC curve
        plt.figure(figsize=(12.5, 12.5))
        plt.plot(false_acceptance_rate, false_rejection_rate, color='red', linewidth=5, label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='green', lw=3, linestyle='--')

        # Highlight EER point
        plt.scatter([eer], [eer], color='blue', label=f'EER = {eer:.4f}', zorder=5)
        plt.axvline(eer, color='blue', linestyle='--', label='EER Line')
        plt.axhline(eer, color='blue', linestyle='--')

        # Customizing font size and family
        font_properties = {'family': 'Arial', 'weight': 'normal', 'size': 46}

        # Add labels, title, and legend
        plt.title('Receiver Operating Curve with EER', **font_properties)
        plt.xlabel('False Acceptance Rate (FAR)', **font_properties)
        plt.ylabel('False Rejection Rate (FRR)', **font_properties)
        plt.grid(True)
        plt.legend(loc='upper right', prop={'family': 'Arial', 'size': 36})
        # Set font properties for axis ticks (x and y axis)
        plt.xticks(fontsize=28, family='Arial')
        plt.yticks(fontsize=28, family='Arial')
        plt.show()