import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# Start timing the script
start_time = time.time()

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 1000

# Function to generate synthetic ncRNA data
def generate_ncRNA_data(samples, features):
    """
    Generates synthetic ncRNA data with added noise.
    
    Parameters:
    samples (int): Number of samples to generate.
    features (int): Number of ncRNA features.
    
    Returns:
    np.ndarray: Generated data with noise.
    """
    data = np.random.rand(samples, features)
    noise = np.random.normal(0, 0.1, data.shape)
    return data + noise

# Function to generate binary labels
def generate_labels(data, noise_level=0.1):
    """
    Generates binary labels based on a weighted sum of the data.
    
    Parameters:
    data (np.ndarray): Input data.
    noise_level (float): Level of noise to add.
    
    Returns:
    np.ndarray: Generated binary labels.
    """
    weights = np.random.rand(data.shape[1])
    linear_combination = np.dot(data, weights) + np.random.randn(data.shape[0]) * noise_level
    threshold = np.percentile(linear_combination, 50)
    return (linear_combination > threshold).astype(int)

# Generate synthetic data for ncRNAs
ncRNA_data = generate_ncRNA_data(num_samples, 50)

# Generate binary labels
labels = generate_labels(ncRNA_data)

# Create DataFrame with ncRNA features and labels
columns = [f'ncRNA_{i}' for i in range(1, 51)]
df = pd.DataFrame(ncRNA_data, columns=columns)
df['label'] = labels

# Function to standardize data
def standardize_data(df, columns):
    """
    Standardizes the given DataFrame columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to standardize.
    
    Returns:
    pd.DataFrame: Standardized DataFrame.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Standardize ncRNA data
df = standardize_data(df, columns)

# Save the dataset to a CSV file
output_file = 'ncRNA_dataset.csv'
df.to_csv(output_file, index=False)

# End timing the script
end_time = time.time()

# Print completion message and runtime
print(f"Dataset ! Saved to {output_file}")
print(f"Runtime: {end_time - start_time:.2f} seconds")

# Display the first few rows of the dataset
print(df.head())
