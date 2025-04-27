import pandas as pd
import os
import glob

def train_test_split(input_df, output_dir, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - input_dir: Directory containing the input files.
    - output_dir: Directory to save the split files.
    - test_size: Proportion of the dataset to include in the test split.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_df)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the number of test samples
    test_size = int(len(df) * test_size)
    
    # Split the DataFrame into train and test sets
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    # Save the train and test sets to CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return train_df, test_df