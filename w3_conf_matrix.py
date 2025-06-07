import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(csv_path, plot_title):
    """
    Plot confusion matrix from results CSV and return accuracy.
    
    Args:
        csv_path (str): Path to the CSV file containing results
        plot_title (str): Title for the confusion matrix plot
    
    Returns:
        float: Classification accuracy
    """
    results_df = pd.read_csv(csv_path)
    
    # Calculate predictions (1 if ramp, 0 if step)
    ramp_predictions = (results_df['ramp_data_ramp_bf'] - results_df['ramp_data_step_bf'] > 0).astype(int)
    step_predictions = (results_df['step_data_ramp_bf'] - results_df['step_data_step_bf'] > 0).astype(int)
    
    ramp_true = np.ones(len(ramp_predictions))
    step_true = np.zeros(len(step_predictions))
    
    y_true = np.concatenate([ramp_true, step_true])
    y_pred = np.concatenate([ramp_predictions, step_predictions])
    
    conf_matrix = np.zeros((2, 2))
    conf_matrix[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    conf_matrix[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    conf_matrix[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    conf_matrix[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Step', 'Ramp'],
                yticklabels=['Step', 'Ramp'])
    plt.title(plot_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    print(f'Overall Accuracy: {accuracy:.2%}')
    return accuracy


if __name__ == '__main__':
    csv_file = 'results/0.25GU_D20_T20.csv'
    title = r'Confusion Matrix - Gaussian Prior, $\sigma_{frac}=0.25$, 5 Trials per Dataset'
    accuracy = plot_confusion_matrix(csv_file, title)