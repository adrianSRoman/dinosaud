import os
import sys
import argparse
import json
from pathlib import Path
import pickle

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from model.soundrain import SoundRain
from dataset.dataloader import SpatialAudioDataset


class SpatialAudioWrapper(nn.Module):
    """
    Wrapper for spatial audio processing - same as in main_dinosaud.py
    """
    def __init__(self, backbone, head=None):
        super(SpatialAudioWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, audio_tensor, return_features=False):
        """
        Args:
            audio_tensor: [batch_size, channels, time]
            return_features: if True, return backbone features instead of head output
        """
        # Process audio through backbone
        features = self.backbone(audio_tensor, mode='encode')
        
        # Global average pooling over time dimension
        features = torch.mean(features, dim=1)  # [batch_size, D]
        
        if return_features or self.head is None:
            return features
        
        # Apply projection head
        output = self.head(features)
        return output


class TestSpatialAudioDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-generated test data using test_metadata.json
    """
    def __init__(self, test_data_dir, sample_rate=44100):
        self.test_data_dir = Path(test_data_dir)
        self.sample_rate = sample_rate
        
        # Load test metadata
        metadata_path = self.test_data_dir / 'test_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.test_samples = self.metadata['test_samples']
        self.config = self.metadata['config']
        
        print(f'TestSpatialAudioDataset loaded:')
        print(f'  Total samples: {len(self.test_samples)}')
        print(f'  Sample rate: {self.sample_rate}')
        print(f'  Audio length: {self.config["audio_length_seconds"]}s')
        print(f'  Channels: {self.config["channel_num"]}')
        print(f'  Reverb type: {self.config["reverb_type"]}')
    
    def __getitem__(self, index):
        """
        Load pre-generated test audio sample
        """
        sample_info = self.test_samples[index]
        
        # Load pre-generated audio file
        audio_path = self.test_data_dir / sample_info['output_file']
        
        # Load audio using soundfile
        import soundfile as sf
        audio_data, sr = sf.read(str(audio_path))
        
        # Convert to tensor and ensure correct format
        if len(audio_data.shape) == 1:
            # Mono audio
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()  # [1, time]
        else:
            # Multi-channel audio: transpose from [time, channels] to [channels, time]
            audio_tensor = torch.from_numpy(audio_data.T).float()  # [channels, time]
        
        # Resample if necessary
        if sr != self.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
        
        return {
            'audio': audio_tensor,
            'spatial_targets': sample_info['spatial_targets'],
            'label': sample_info['audio_label'],
            'audio_id': sample_info['audio_id'],
            'group_id': sample_info['group_id'],
            'reverb_idx': sample_info['reverb_idx']
        }
    
    def __len__(self):
        return len(self.test_samples)


def build_model(args):
    """Build and load the model"""
    backbone = SoundRain(
        n_q=args.n_q,
        codebook_size=args.codebook_size,
        D=args.D,
        C=args.C,
        strides=args.strides
    )
    
    model = SpatialAudioWrapper(backbone)
    model.cuda()
    
    # Load pretrained weights
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, "soundrain", None)
    model.eval()
    
    return model


def extract_train_features_chunked(args):
    """Extract training features in chunks and save to disk"""
    print("=" * 50)
    print("EXTRACTING TRAINING FEATURES")
    print("=" * 50)
    
    # Create output directory
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    train_chunks_dir = features_dir / "train_chunks"
    train_chunks_dir.mkdir(exist_ok=True)
    
    # Check if features already exist
    train_features_path = features_dir / "train_features.pth"
    train_labels_path = features_dir / "train_spatial_labels.pth"
    
    if train_features_path.exists() and train_labels_path.exists() and not args.overwrite_features:
        print(f"Training features already exist at {train_features_path}")
        print("Use --overwrite_features to recompute")
        return
    
    # Load training dataset
    print("Loading training dataset...")
    dataset_train = SpatialAudioDataset(
        audio_json=args.train_audio_json,
        reverb_json=args.reverb_json,
        audio_path_root=args.audio_path_root,
        reverb_path_root=args.reverb_path_root,
        reverb_type=args.reverb_type,
        sample_rate=args.sample_rate,
        audio_length_seconds=args.audio_length_seconds,
        normalize=args.normalize,
        mode="train"
    )
    
    # Create data loader
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_train.collate_fn
    )
    
    print(f"Training dataset loaded: {len(dataset_train)} samples")
    
    # Build model
    model = build_model(args)
    
    # Extract features in chunks
    print("Extracting training features in chunks...")
    chunk_idx = 0
    all_chunk_info = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_train):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader_train)}")
            
            # Extract teacher audio and spatial targets
            teacher_audio = batch['teacher_audio'].cuda(non_blocking=True)
            spatial_targets = batch['teacher_spatial_targets']
            
            # Extract features
            features = model(teacher_audio, return_features=True)
            features = nn.functional.normalize(features, dim=1, p=2)
            
            # Save chunk
            chunk_data = {
                'features': features.cpu(),
                'spatial_targets': {
                    'distance': spatial_targets['distance'].cpu(),
                    'azimuth': spatial_targets['azimuth'].cpu(),
                    'elevation': spatial_targets['elevation'].cpu()
                },
                'batch_idx': batch_idx,
                'num_samples': features.shape[0]
            }
            
            chunk_path = train_chunks_dir / f"chunk_{chunk_idx:04d}.pth"
            torch.save(chunk_data, chunk_path)
            
            all_chunk_info.append({
                'chunk_idx': chunk_idx,
                'chunk_path': str(chunk_path),
                'num_samples': features.shape[0],
                'feature_dim': features.shape[1]
            })
            
            chunk_idx += 1
            
            # Clear GPU memory
            del features, teacher_audio
            torch.cuda.empty_cache()
    
    # Combine all chunks into single files
    print("Combining chunks into single files...")
    total_samples = sum(info['num_samples'] for info in all_chunk_info)
    feature_dim = all_chunk_info[0]['feature_dim']
    
    # Initialize combined tensors
    combined_features = torch.zeros(total_samples, feature_dim)
    combined_spatial_labels = {
        'distance': torch.zeros(total_samples, dtype=torch.long),
        'azimuth': torch.zeros(total_samples, dtype=torch.long),
        'elevation': torch.zeros(total_samples, dtype=torch.long)
    }
    
    # Load and combine chunks
    current_idx = 0
    for chunk_info in all_chunk_info:
        chunk_data = torch.load(chunk_info['chunk_path'])
        num_samples = chunk_info['num_samples']
        
        # Copy features
        combined_features[current_idx:current_idx + num_samples] = chunk_data['features']
        
        # Copy spatial labels
        for key in combined_spatial_labels:
            combined_spatial_labels[key][current_idx:current_idx + num_samples] = chunk_data['spatial_targets'][key]
        
        current_idx += num_samples
        
        # Clean up chunk file
        os.remove(chunk_info['chunk_path'])
    
    # Save combined files
    torch.save(combined_features, train_features_path)
    torch.save(combined_spatial_labels, train_labels_path)
    
    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'feature_dim': feature_dim,
        'spatial_classes': {
            'distance': 21,  # 0-20
            'azimuth': 360,  # 0-359
            'elevation': 180  # 0-179
        }
    }
    
    with open(features_dir / "train_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up chunks directory
    train_chunks_dir.rmdir()
    
    print(f"Training features saved:")
    print(f"  Features: {train_features_path} ({total_samples} x {feature_dim})")
    print(f"  Labels: {train_labels_path}")
    print(f"  Metadata: {features_dir / 'train_metadata.json'}")


def extract_test_features_chunked(args):
    """Extract test features in chunks and save to disk"""
    print("=" * 50)
    print("EXTRACTING TEST FEATURES")
    print("=" * 50)
    
    # Create output directory
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    test_chunks_dir = features_dir / "test_chunks"
    test_chunks_dir.mkdir(exist_ok=True)
    
    # Check if features already exist
    test_features_path = features_dir / "test_features.pth"
    test_labels_path = features_dir / "test_spatial_labels.pth"
    
    if test_features_path.exists() and test_labels_path.exists() and not args.overwrite_features:
        print(f"Test features already exist at {test_features_path}")
        print("Use --overwrite_features to recompute")
        return
    
    # Load test dataset
    print("Loading test dataset...")
    dataset_test = TestSpatialAudioDataset(
        test_data_dir=args.test_data_dir,
        sample_rate=args.sample_rate
    )
    
    # Custom collate function for test data
    def test_collate_fn(batch):
        audio_batch = torch.stack([item['audio'] for item in batch])
        spatial_targets = {
            'distance': torch.LongTensor([item['spatial_targets']['distance'] for item in batch]),
            'azimuth': torch.LongTensor([item['spatial_targets']['azimuth'] for item in batch]),
            'elevation': torch.LongTensor([item['spatial_targets']['elevation'] for item in batch])
        }
        return {
            'audio': audio_batch,
            'spatial_targets': spatial_targets,
            'labels': torch.LongTensor([item['label'] for item in batch]),
            'audio_ids': [item['audio_id'] for item in batch],
            'group_ids': torch.LongTensor([int(item['group_id'].split("_")[1]) for item in batch]),
            'reverb_indices': torch.LongTensor([item['reverb_idx'] for item in batch])
        }
    
    # Create data loader
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_collate_fn
    )
    
    print(f"Test dataset loaded: {len(dataset_test)} samples")
    
    # Build model
    model = build_model(args)
    
    # Extract features in chunks
    print("Extracting test features in chunks...")
    chunk_idx = 0
    all_chunk_info = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_test):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader_test)}")
            
            # Extract audio and spatial targets
            audio = batch['audio'].cuda(non_blocking=True)
            spatial_targets = batch['spatial_targets']
            
            # Extract features
            features = model(audio, return_features=True)
            features = nn.functional.normalize(features, dim=1, p=2)
            
            # Save chunk
            chunk_data = {
                'features': features.cpu(),
                'spatial_targets': {
                    'distance': spatial_targets['distance'].cpu(),
                    'azimuth': spatial_targets['azimuth'].cpu(),
                    'elevation': spatial_targets['elevation'].cpu()
                },
                'batch_idx': batch_idx,
                'num_samples': features.shape[0]
            }
            
            chunk_path = test_chunks_dir / f"chunk_{chunk_idx:04d}.pth"
            torch.save(chunk_data, chunk_path)
            
            all_chunk_info.append({
                'chunk_idx': chunk_idx,
                'chunk_path': str(chunk_path),
                'num_samples': features.shape[0],
                'feature_dim': features.shape[1]
            })
            
            chunk_idx += 1
            
            # Clear GPU memory
            del features, audio
            torch.cuda.empty_cache()
    
    # Combine all chunks into single files
    print("Combining chunks into single files...")
    total_samples = sum(info['num_samples'] for info in all_chunk_info)
    feature_dim = all_chunk_info[0]['feature_dim']
    
    # Initialize combined tensors
    combined_features = torch.zeros(total_samples, feature_dim)
    combined_spatial_labels = {
        'distance': torch.zeros(total_samples, dtype=torch.long),
        'azimuth': torch.zeros(total_samples, dtype=torch.long),
        'elevation': torch.zeros(total_samples, dtype=torch.long)
    }
    
    # Load and combine chunks
    current_idx = 0
    for chunk_info in all_chunk_info:
        chunk_data = torch.load(chunk_info['chunk_path'])
        num_samples = chunk_info['num_samples']
        
        # Copy features
        combined_features[current_idx:current_idx + num_samples] = chunk_data['features']
        
        # Copy spatial labels
        for key in combined_spatial_labels:
            combined_spatial_labels[key][current_idx:current_idx + num_samples] = chunk_data['spatial_targets'][key]
        
        current_idx += num_samples
        
        # Clean up chunk file
        os.remove(chunk_info['chunk_path'])
    
    # Save combined files
    torch.save(combined_features, test_features_path)
    torch.save(combined_spatial_labels, test_labels_path)
    
    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'feature_dim': feature_dim,
        'spatial_classes': {
            'distance': 21,  # 0-20
            'azimuth': 360,  # 0-359
            'elevation': 180  # 0-179
        }
    }
    
    with open(features_dir / "test_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up chunks directory
    test_chunks_dir.rmdir()
    
    print(f"Test features saved:")
    print(f"  Features: {test_features_path} ({total_samples} x {feature_dim})")
    print(f"  Labels: {test_labels_path}")
    print(f"  Metadata: {features_dir / 'test_metadata.json'}")


@torch.no_grad()
def knn_classifier_spatial(train_features, train_spatial_labels, test_features, test_spatial_labels, k, T, spatial_type='azimuth'):
    """
    k-NN classifier for spatial audio parameters
    
    Args:
        spatial_type: 'azimuth', 'elevation', or 'distance'
    """
    print(f"Running {k}-NN classification for {spatial_type}...")
    
    # Get the specific spatial labels
    train_labels = train_spatial_labels[spatial_type]
    test_labels = test_spatial_labels[spatial_type]
    
    # Determine number of classes
    if spatial_type == 'azimuth':
        num_classes = 360  # 0-359 degrees
    elif spatial_type == 'elevation':
        num_classes = 180  # 0-179 degrees (mapped from -90 to 90)
    elif spatial_type == 'distance':
        num_classes = 21   # 0-20 (discretized distance * 10)
    else:
        raise ValueError(f"Unknown spatial_type: {spatial_type}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features = train_features.to(device)
    test_features = test_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    
    top1, top5, total = 0.0, 0.0, 0
    train_features_t = train_features.t()
    num_test_samples = test_labels.shape[0]
    
    # Process in chunks to avoid memory issues
    chunk_size = min(1000, num_test_samples)
    retrieval_one_hot = torch.zeros(k, num_classes).to(device)
    
    for idx in range(0, num_test_samples, chunk_size):
        end_idx = min(idx + chunk_size, num_test_samples)
        features = test_features[idx:end_idx, :]
        targets = test_labels[idx:end_idx]
        batch_size = targets.shape[0]

        # Calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features_t)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # Find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)
    
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def run_knn_evaluation(args):
    """Run k-NN evaluation using preprocessed features"""
    print("=" * 50)
    print("RUNNING K-NN EVALUATION")
    print("=" * 50)
    
    features_dir = Path(args.features_dir)
    
    # Load preprocessed features
    print("Loading preprocessed features...")
    
    train_features_path = features_dir / "train_features.pth"
    train_labels_path = features_dir / "train_spatial_labels.pth"
    test_features_path = features_dir / "test_features.pth"
    test_labels_path = features_dir / "test_spatial_labels.pth"

    print(f"Train features: {train_features_path}")
    print(f"Train labels: {train_labels_path}")
    print(f"Test features: {test_features_path}")
    print(f"Test labels: {test_labels_path}")
    
    if not all(p.exists() for p in [train_features_path, train_labels_path, test_features_path, test_labels_path]):
        print("ERROR: Preprocessed features not found. Run with --extract_features first.")
        return
    
    train_features = torch.load(train_features_path)
    train_spatial_labels = torch.load(train_labels_path)
    test_features = torch.load(test_features_path)
    test_spatial_labels = torch.load(test_labels_path)
    
    print(f"Loaded features:")
    print(f"  Train: {train_features.shape}")
    print(f"  Test: {test_features.shape}")
    
    # Run k-NN evaluation for each spatial parameter
    results = {}
    for spatial_type in ['azimuth', 'elevation', 'distance']:
        print(f"\n=== {spatial_type.upper()} Classification ===")
        spatial_results = {}
        
        for k in args.nb_knn:
            top1, top5 = knn_classifier_spatial(
                train_features, train_spatial_labels,
                test_features, test_spatial_labels, 
                k, args.temperature, spatial_type=spatial_type
            )
            print(f"{k}-NN classifier result for {spatial_type}: Top1: {top1:.2f}%, Top5: {top5:.2f}%")
            spatial_results[k] = {'top1': top1, 'top5': top5}
        
        results[spatial_type] = spatial_results
    
    # Save results
    results_path = features_dir / "knn_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with k-NN on Spatial Audio')
    
    # Model parameters
    parser.add_argument('--n_q', default=4, type=int, help='Number of quantizers in SoundRain')
    parser.add_argument('--codebook_size', default=1024, type=int, help='Codebook size in SoundRain')
    parser.add_argument('--D', default=32, type=int, help='Embedding dimension in SoundRain')
    parser.add_argument('--C', default=32, type=int, help='Channel dimension in SoundRain')
    parser.add_argument('--strides', default=[2, 4, 5, 8], type=int, nargs='+', help='Strides in SoundRain')
    
    # Audio parameters
    parser.add_argument('--sample_rate', default=44100, type=int, help='Audio sample rate')
    parser.add_argument('--audio_length_seconds', default=1, type=int, help='Audio length in seconds for training')
    parser.add_argument('--reverb_type', default='mic', type=str, choices=['binaural', 'mic', 'mono'])
    parser.add_argument('--normalize', default=True, type=utils.bool_flag, help='Normalize audio')
    
    # Dataset parameters
    parser.add_argument('--train_audio_json', default="/scratch/data/repos/dinosaud/train_metadata/audio_metadata.json", type=str, help='Path to training audio metadata JSON')
    parser.add_argument('--test_data_dir', default="/scratch/ssd1/DINOSAUD_test", type=str, help='Path to directory containing pre-generated test data and test_metadata.json')
    parser.add_argument('--reverb_json', default="/scratch/data/repos/dinosaud/train_metadata/metadata.json", type=str, help='Path to reverb groups JSON (for training data)')
    parser.add_argument('--audio_path_root', default="/scratch/ssd1/audio_datasets/AudioSet_DCASE", type=str, help='Root path for audio files')
    parser.add_argument('--reverb_path_root', default="/scratch/ssd1/DINOSAUD_Dataset", type=str, help='Root path for reverb files')

    
    # Feature extraction parameters
    parser.add_argument('--features_dir', default="/scratch/ssd1/DINOSAUD_features", type=str, help='Directory to save/load preprocessed features')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--overwrite_features', action='store_true', help='Overwrite existing preprocessed features')
    
    # Model parameters
    parser.add_argument('--pretrained_weights', default='/scratch/data/repos/dinosaud/outputs/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="student", type=str,
        help='Key to use in the checkpoint (example: "student")')
    
    # Evaluation parameters
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    
    # Execution mode
    parser.add_argument('--extract_features', action='store_true', help='Extract and save features')
    parser.add_argument('--run_knn', action='store_true', help='Run k-NN evaluation using preprocessed features')
    
    # Distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.extract_features:
        print("Extracting features...")
        extract_train_features_chunked(args)
        extract_test_features_chunked(args)
        print("Feature extraction complete!")
    
    if args.run_knn:
        print("Running k-NN evaluation...")
        run_knn_evaluation(args)
        print("k-NN evaluation complete!")
    
    if not args.extract_features and not args.run_knn:
        print("Please specify --extract_features and/or --run_knn")
    
    dist.barrier()