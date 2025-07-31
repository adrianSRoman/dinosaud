import os
import json
import argparse
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import random


def normalize_audio(waveform, target_lufs=-14.0):
    """Normalize audio to target LUFS"""
    # Simple RMS-based normalization as approximation
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 0:
        target_rms = 10 ** (target_lufs / 20)
        waveform = waveform * (target_rms / rms)
    return waveform


def load_audio(audio_path, sample_rate=44100, audio_length_seconds=1, normalize=True):
    """Load and preprocess audio file"""
    # Load audio
    waveform, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = waveform[:, 0]
    
    # Resample if necessary
    if sr != sample_rate:
        waveform = signal.resample_poly(waveform, sample_rate, sr)
    
    # Normalize audio
    if normalize:
        waveform = normalize_audio(torch.from_numpy(waveform), -14.0).numpy()
    
    # Convert to tensor and reshape
    waveform = torch.from_numpy(waveform).reshape(1, -1).float()
    
    # Pad or trim to fixed length
    audio_length_samples = sample_rate * audio_length_seconds
    padding = audio_length_samples - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), 'constant', 0)
    elif padding < 0:
        waveform = waveform[:, :audio_length_samples]
        
    return waveform


def load_reverb(reverb_path, sample_rate=44100):
    """Load reverb impulse response"""
    # Load reverb
    reverb = torch.from_numpy(np.load(reverb_path)).float()
    
    # Ensure reverb has correct shape [channels, time]
    if reverb.dim() == 3:
        reverb = reverb.squeeze(1)
    
    # Pad or trim reverb to 2 seconds
    reverb_length = sample_rate * 2
    reverb_padding = reverb_length - reverb.shape[1]
    if reverb_padding > 0:
        reverb = F.pad(reverb, (0, reverb_padding), 'constant', 0)
    elif reverb_padding < 0:
        reverb = reverb[:, :reverb_length]
        
    return reverb


def get_spatial_targets(reverb_metadata):
    """Extract spatial information from reverb metadata"""
    # Distance classes (discretized)
    distance = reverb_metadata['distance']
    distance_class = min(int(distance * 10), 20)  # 21 classes (0-20)
    
    # Azimuth (0-359 degrees)
    azimuth = reverb_metadata['azimuth']
    azimuth_class = int((azimuth + 180) % 360)  # Convert [-180, 180] to [0, 359]
    
    # Elevation (0-179 degrees) 
    elevation = reverb_metadata['elevation']
    elevation_class = int(elevation + 90)  # Convert [-90, 90] to [0, 179]
    elevation_class = max(0, min(179, elevation_class))
    
    return {
        "distance": distance_class,
        "azimuth": azimuth_class, 
        "elevation": elevation_class
    }


def generate_test_data(args):
    """Generate test multi-channel audio data from groups metadata"""
    
    # Load audio metadata (test set)
    print(f"Loading test audio metadata from {args.audio_json}")
    with open(args.audio_json, 'r') as f:
        audio_data = json.load(f)
    print(f"Found {len(audio_data)} test audio samples")
    
    # Load reverb groups metadata
    print(f"Loading reverb groups metadata from {args.reverb_json}")
    with open(args.reverb_json, 'r') as f:
        reverb_metadata = json.load(f)
    
    # Extract groups
    if 'groups' not in reverb_metadata:
        raise ValueError("No 'groups' found in reverb metadata file")
    
    groups = reverb_metadata['groups']
    print(f"Found {len(groups)} groups in metadata")
    
    # Count total reverbs across all groups
    total_reverbs = sum(len(group['rooms']) for group in groups.values())
    print(f"Total reverbs across all groups: {total_reverbs}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of channels based on reverb type
    channel_num = 2 if args.reverb_type == 'binaural' else 4 if args.reverb_type == 'mic' else 1
    print(f"Generating {args.reverb_type} audio with {channel_num} channels")
    
    # Limit audio samples if requested
    if args.max_audio_samples > 0:
        audio_data = audio_data[:args.max_audio_samples]
        print(f"Limited to {len(audio_data)} audio samples")
    
    # Generate test data
    test_samples = []
    
    # Calculate total combinations for progress bar
    if args.convolution_strategy == 'all_combinations':
        total_combinations = len(audio_data) * total_reverbs
    else:  # one_per_reverb
        total_combinations = total_reverbs
    
    print(f"Generating {total_combinations} test samples using '{args.convolution_strategy}' strategy...")
    
    reverb_counter = 0
    
    with tqdm(total=total_combinations, desc="Generating test data") as pbar:
        for group_id, group_data in groups.items():
            if int(group_id.split("_")[1]) <= 493:
                continue
            group_name = group_data.get('group_name', f'group_{group_id}')
            reverbs = group_data['rooms']
            
            print(f"\nProcessing group '{group_name}' with {len(reverbs)} reverbs")
            
            for reverb_idx, reverb_meta in enumerate(reverbs):
                # Load reverb
                reverb_path = os.path.join(args.reverb_path_root, reverb_meta['fname'])
                
                try:
                    reverb = load_reverb(reverb_path, args.sample_rate)
                except Exception as e:
                    print(f"Error loading reverb {reverb_path}: {e}")
                    if args.convolution_strategy == 'all_combinations':
                        pbar.update(len(audio_data))
                    else:
                        pbar.update(1)
                    continue
                
                # Ensure reverb has correct number of channels
                if reverb.shape[0] != channel_num:
                    if reverb.shape[0] > channel_num:
                        reverb = reverb[:channel_num]  # Take first N channels
                    else:
                        # Duplicate channels if needed
                        reverb = reverb.repeat(channel_num // reverb.shape[0] + 1, 1)[:channel_num]
                
                # Determine which audio samples to convolve with this reverb
                if args.convolution_strategy == 'all_combinations':
                    # Convolve with all test audio samples
                    audio_samples = audio_data
                elif args.convolution_strategy == 'one_per_reverb':
                    # Convolve with one randomly selected audio sample
                    audio_samples = [random.choice(audio_data)]
                else:
                    raise ValueError(f"Unknown convolution strategy: {args.convolution_strategy}")
                
                for audio_item in audio_samples:
                    # Load base audio
                    audio_path = os.path.join(
                        args.audio_path_root, 
                        audio_item['folder'], 
                        audio_item['id'] + args.audio_extension
                    )
                    
                    try:
                        base_audio = load_audio(
                            audio_path, 
                            args.sample_rate, 
                            args.audio_length_seconds, 
                            args.normalize
                        )
                    except Exception as e:
                        print(f"Error loading audio {audio_path}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Convolve audio with reverb
                    try:
                        convolved = torchaudio.functional.fftconvolve(
                            base_audio, reverb, mode='full'
                        )[..., :base_audio.shape[-1]]
                    except Exception as e:
                        print(f"Error convolving audio: {e}")
                        pbar.update(1)
                        continue
                    
                    # Get spatial targets
                    spatial_targets = get_spatial_targets(reverb_meta)
                    
                    # Create output filename
                    output_filename = f"{audio_item['id']}_group{group_id}_reverb{reverb_counter:04d}.wav"
                    output_path = output_dir / output_filename
                    
                    # Save multi-channel audio
                    try:
                        # Convert to numpy and transpose for soundfile (time, channels)
                        audio_np = convolved.squeeze(0).T.numpy()  # [time, channels]
                        sf.write(output_path, audio_np, args.sample_rate)
                    except Exception as e:
                        print(f"Error saving audio {output_path}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Store metadata
                    sample_info = {
                        'audio_id': audio_item['id'],
                        'group_id': group_id,
                        'group_name': group_name,
                        'reverb_idx': reverb_counter,
                        'reverb_fname': reverb_meta['fname'],
                        'output_file': output_filename,
                        'audio_label': audio_item['label'],
                        'spatial_targets': spatial_targets,
                        'reverb_metadata': reverb_meta
                    }
                    test_samples.append(sample_info)
                    
                    pbar.update(1)
                
                reverb_counter += 1
    
    # Save test metadata
    test_metadata = {
        'test_samples': test_samples,
        'groups_info': {
            group_id: {
                'group_name': group_data.get('group_name', f'group_{group_id}'),
                'num_reverbs': len(group_data['rooms']),
                'sample_count': len([s for s in test_samples if s['group_id'] == group_id])
            }
            for group_id, group_data in groups.items()
        },
        'config': {
            'sample_rate': args.sample_rate,
            'audio_length_seconds': args.audio_length_seconds,
            'reverb_type': args.reverb_type,
            'channel_num': channel_num,
            'normalize': args.normalize,
            'convolution_strategy': args.convolution_strategy,
            'max_audio_samples': args.max_audio_samples,
            'seed': args.seed
        },
        'statistics': {
            'total_samples': len(test_samples),
            'unique_audio_files': len(set(s['audio_id'] for s in test_samples)),
            'unique_reverbs': len(set(s['reverb_idx'] for s in test_samples)),
            'unique_groups': len(set(s['group_id'] for s in test_samples))
        }
    }
    
    metadata_path = output_dir / 'test_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Generated {len(test_samples)} test samples")
    print(f"Saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print detailed statistics
    if test_samples:
        print(f"\nDetailed Statistics:")
        print(f"Total samples: {len(test_samples)}")
        print(f"Unique audio files: {len(set(s['audio_id'] for s in test_samples))}")
        print(f"Unique reverbs: {len(set(s['reverb_idx'] for s in test_samples))}")
        print(f"Groups processed: {len(set(s['group_id'] for s in test_samples))}")
        
        # Group-wise statistics
        print(f"\nSamples per group:")
        for group_id in sorted(set(s['group_id'] for s in test_samples)):
            group_samples = [s for s in test_samples if s['group_id'] == group_id]
            group_name = group_samples[0]['group_name']
            print(f"  Group {group_id} ({group_name}): {len(group_samples)} samples")
        
        # Spatial statistics
        distances = [s['spatial_targets']['distance'] for s in test_samples]
        azimuths = [s['spatial_targets']['azimuth'] for s in test_samples]
        elevations = [s['spatial_targets']['elevation'] for s in test_samples]
        
        print(f"\nSpatial parameter ranges:")
        print(f"Distance: {min(distances)} - {max(distances)} (classes)")
        print(f"Azimuth: {min(azimuths)} - {max(azimuths)} (degrees)")
        print(f"Elevation: {min(elevations)} - {max(elevations)} (degrees)")
        
        # Distribution statistics
        print(f"\nSpatial parameter distributions:")
        print(f"Distance classes: {len(set(distances))} unique values")
        print(f"Azimuth values: {len(set(azimuths))} unique values")
        print(f"Elevation values: {len(set(elevations))} unique values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate test multi-channel audio data from reverb groups')
    
    # Input paths
    parser.add_argument('--audio_json', default="/scratch/data/repos/dinosaud/test_metadata/audio_metadata.json", type=str, 
                        help='Path to test audio metadata JSON file')
    parser.add_argument('--reverb_json', default="/scratch/data/repos/dinosaud/test_metadata/metadata.json", type=str,
                        help='Path to reverb groups metadata JSON file')
    parser.add_argument('--audio_path_root', default="/scratch/ssd1/audio_datasets/AudioSet_DCASE", type=str,
                        help='Root directory containing audio files')
    parser.add_argument('--reverb_path_root', default="/scratch/ssd1/DINOSAUD_Dataset", type=str,
                        help='Root directory containing reverb files')
    parser.add_argument('--output_dir', default="/scratch/ssd1/DINOSAUD_test", type=str,
                        help='Output directory for generated test data')
    
    # Audio parameters
    parser.add_argument('--sample_rate', default=44100, type=int,
                        help='Target sample rate')
    parser.add_argument('--audio_length_seconds', default=1, type=int,
                        help='Length of each audio sample in seconds')
    parser.add_argument('--reverb_type', default='mic', type=str,
                        choices=['binaural', 'mic', 'mono'],
                        help='Type of reverb/spatial audio')
    parser.add_argument('--audio_extension', default='.wav', type=str,
                        help='Audio file extension')
    parser.add_argument('--normalize', default=True, type=bool,
                        help='Whether to normalize audio')
    
    # Generation strategy
    parser.add_argument('--convolution_strategy', default='one_per_reverb', type=str,
                        choices=['all_combinations', 'one_per_reverb'],
                        help='Strategy for convolving audio with reverbs')
    parser.add_argument('--max_audio_samples', default=0, type=int,
                        help='Maximum number of audio samples to use (0 = all)')
    
    # Random seed
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    generate_test_data(args)