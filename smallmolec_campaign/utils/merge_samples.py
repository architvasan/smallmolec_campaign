import pandas as pd
import os
import glob

def merge_samples(input_dir, output_file):
    """
    Merges multiple CSV files containing SMILES strings into a single CSV file.

    Args:
        input_dir (str): Directory containing the input CSV files.
        output_file (str): Path to the output merged CSV file.
    """
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Read each CSV file and append it to the list
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files into {output_file}")
    return merged_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge multiple CSV files containing SMILES strings into a single CSV file.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input CSV files.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output merged CSV file.')

    args = parser.parse_args()

    merge_samples(args.input_dir, args.output_file)