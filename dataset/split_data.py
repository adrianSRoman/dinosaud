import os
import argparse
from pathlib import Path

def create_wav_file_lists(input_dir, output_dir, fold_config):
    """
    Create text files containing paths to WAV files for training, validation, and test sets.

    Args:
        input_dir (str): Path to the directory containing WAV files.
        output_dir (str): Path to the directory for output text files.
        fold_config (dict): A dictionary specifying fold numbers for train, val, and test sets.
    """
    # Convert to Path objects
    main_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all WAV files (case insensitive)
    wav_files = list(main_path.glob('mic_dev/**/*.wav'))
    wav_files = sorted([str(path) for path in wav_files])

    # Categorize files into train, val, and test sets
    datasets = {
        'train': [],
        'val': [],
        'test': []
    }

    for wav_file in wav_files:
        for dataset_type, folds in fold_config.items():
            for fold in folds:
                if f"fold{fold}" in wav_file:
                    foa_file = wav_file.replace("mic_dev", "foa_dev").replace(".wav", ".wav")
                    datasets[dataset_type].append(f"{wav_file} {foa_file}\n")

    # Write paths to respective output files
    for dataset_type, file_list in datasets.items():
        output_file = output_path / f"{dataset_type}_dataset.txt"
        with open(output_file, 'w') as f:
            f.writelines(file_list)
        print(f"{dataset_type.capitalize()} set: {len(file_list)} files written to {output_file}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Create dataset lists for WAV files by fold')
    parser.add_argument('input_dir', help='Directory containing WAV files')
    parser.add_argument('output_dir', help='Directory for output text files')

    # Parse arguments
    args = parser.parse_args()

    # Define fold configuration for experiments
    fold_config = {
        'train': [0, 1, 2, 3],
        'val': [4],
        'test': [5]
    }

    # Create the file lists
    create_wav_file_lists(args.input_dir, args.output_dir, fold_config)
