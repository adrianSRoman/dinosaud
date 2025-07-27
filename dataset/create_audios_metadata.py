import json
import os
import argparse
from pathlib import Path
import numpy as np


def create_audio_json(audio_root, output_path, file_extension=".wav", split="train"):
    """
    Create audio metadata JSON file by scanning AudioSet_DCASE directory structure.
    Expected structure: AudioSet_DCASE/<sound_class>/train/assets/*.wav
    
    Args:
        audio_root: Root directory (AudioSet_DCASE)
        output_path: Path to save the JSON file
        file_extension: Audio file extension to look for
        split: Which split to process ('train', 'val', 'test')
    """
    audio_data = []
    audio_root = Path(audio_root)
    
    # Create label mapping from sound classes
    sound_classes = []
    
    # First pass: collect all sound classes
    for class_dir in audio_root.iterdir():
        if class_dir.is_dir():
            split_dir = class_dir / split / "assets"
            if split_dir.exists():
                sound_classes.append(class_dir.name)
    
    # Sort for consistent ordering
    sound_classes.sort()
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sound_classes)}
    
    print(f"Found {len(sound_classes)} sound classes:")
    for i, class_name in enumerate(sound_classes[:10]):  # Show first 10
        print(f"  {i}: {class_name}")
    if len(sound_classes) > 10:
        print(f"  ... and {len(sound_classes) - 10} more")
    
    # Second pass: collect audio files
    for class_dir in audio_root.iterdir():
        if not class_dir.is_dir():
            continue
            
        sound_class = class_dir.name
        split_dir = class_dir / split / "assets"
        
        if not split_dir.exists():
            continue
            
        # Find all audio files in this class
        audio_files = list(split_dir.glob(f"*{file_extension}"))
        
        for audio_file in audio_files:
            filename = audio_file.stem  # filename without extension
            
            # Folder path relative to audio_root for this structure
            folder = f"{sound_class}/{split}/assets"
            
            # Get label index
            label = class_to_idx[sound_class]
            
            audio_data.append({
                "id": filename,
                "label": label,
                "folder": folder
            })
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(audio_data, f, indent=2)
    
    # Also save class mapping for reference
    class_mapping_path = output_path.replace('.json', '_class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {str(idx): class_name for class_name, idx in class_to_idx.items()},
            'num_classes': len(sound_classes)
        }, f, indent=2)
    
    print(f"Created audio JSON with {len(audio_data)} entries at {output_path}")
    print(f"Created class mapping with {len(sound_classes)} classes at {class_mapping_path}")
    
    # Print statistics
    class_counts = {}
    for item in audio_data:
        class_name = sound_classes[item['label']]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nDataset statistics for {split} split:")
    print(f"Total samples: {len(audio_data)}")
    print(f"Classes: {len(sound_classes)}")
    print(f"Average samples per class: {len(audio_data) / len(sound_classes):.1f}")
    
    # Show top 10 classes by sample count
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 classes by sample count:")
    for class_name, count in top_classes:
        print(f"  {class_name}: {count} samples")
    
    return audio_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create audio metadata JSON from AudioSet_DCASE directory structure')
    parser.add_argument('--audio_root', type=str, default="/scratch/ssd1/audio_datasets/AudioSet_DCASE", help='Root directory of AudioSet_DCASE')
    parser.add_argument('--output_path', type=str, default="/scratch/data/repos/dinosaud/train_metadata/audio_metadata.json", help='Path to save the output JSON file')
    parser.add_argument('--file_extension', type=str, default='.wav', help='Audio file extension to look for')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train', help='Which split to process')
    
    args = parser.parse_args()
    
    create_audio_json(args.audio_root, args.output_path, args.file_extension, args.split)