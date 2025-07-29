import os
import sys
import argparse
import json
from pathlib import Path

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


def extract_feature_pipeline(args):
    """Extract features from both training and test datasets"""
    
    # ============ preparing training data ... ============
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
    
    # ============ preparing test data ... ============
    print("Loading test dataset...")
    dataset_test = TestSpatialAudioDataset(
        test_data_dir=args.test_data_dir,
        sample_rate=args.sample_rate
    )
    
    # Create samplers and data loaders
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=False)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_train.collate_fn
    )
    
    def test_collate_fn(batch):
        """Custom collate function for test data"""
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
            'group_ids': torch.LongTensor([item['group_id'] for item in batch]),
            'reverb_indices': torch.LongTensor([item['reverb_idx'] for item in batch])
        }
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_collate_fn
    )
    
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_test)} test samples.")

    # ============ building network ... ============
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

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_spatial_labels = extract_features_spatial(model, data_loader_train, args.use_cuda, mode="train")
    
    print("Extracting features for test set...")
    test_features, test_spatial_labels = extract_features_spatial(model, data_loader_test, args.use_cuda, mode="test")

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # Save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_spatial_labels, os.path.join(args.dump_features, "train_spatial_labels.pth"))
        torch.save(test_spatial_labels, os.path.join(args.dump_features, "test_spatial_labels.pth"))
    
    return train_features, test_features, train_spatial_labels, test_spatial_labels


@torch.no_grad()
def extract_features_spatial(model, data_loader, use_cuda=True, mode="train"):
    """Extract features and spatial labels from spatial audio data"""
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    all_spatial_labels = []
    
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10)):
        if mode == "train":
            # For training data, we use teacher audio (2 crops per sample)
            audio = batch['teacher_audio'].cuda(non_blocking=True)
            spatial_targets = batch['teacher_spatial_targets']
            batch_size = audio.shape[0]
        else:
            # For test data, single audio sample
            audio = batch['audio'].cuda(non_blocking=True)
            spatial_targets = batch['spatial_targets']
            batch_size = audio.shape[0]
        
        # Extract features
        feats = model(audio, return_features=True).clone()

        # Initialize storage feature matrix
        if dist.get_rank() == 0 and features is None:
            if mode == "train":
                # Training data has 2 crops per sample
                total_samples = len(data_loader.dataset) * 2
            else:
                # Test data has 1 sample per item
                total_samples = len(data_loader.dataset)
            
            features = torch.zeros(total_samples, feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # Create indices for this batch
        if mode == "train":
            # Each sample produces 2 features (teacher crops)
            start_idx = batch_idx * data_loader.batch_size * 2
            indices = torch.arange(start_idx, start_idx + batch_size).cuda()
        else:
            start_idx = batch_idx * data_loader.batch_size
            indices = torch.arange(start_idx, start_idx + batch_size).cuda()

        # Gather indices from all processes
        y_all = torch.empty(dist.get_world_size(), indices.size(0), dtype=indices.dtype, device=indices.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, indices, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # Gather features from all processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # Update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        
        # Collect spatial labels
        all_spatial_labels.append({
            'distance': spatial_targets['distance'].cpu(),
            'azimuth': spatial_targets['azimuth'].cpu(),
            'elevation': spatial_targets['elevation'].cpu(),
        })
    
    # Concatenate all spatial labels
    spatial_labels = {
        'distance': torch.cat([labels['distance'] for labels in all_spatial_labels]),
        'azimuth': torch.cat([labels['azimuth'] for labels in all_spatial_labels]),
        'elevation': torch.cat([labels['elevation'] for labels in all_spatial_labels]),
    }
    
    return features, spatial_labels


@torch.no_grad()
def knn_classifier_spatial(train_features, train_spatial_labels, test_features, test_spatial_labels, k, T, spatial_type='azimuth'):
    """
    k-NN classifier for spatial audio parameters
    
    Args:
        spatial_type: 'azimuth', 'elevation', or 'distance'
    """
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
    
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_samples, num_chunks = test_labels.shape[0], 100
    samples_per_chunk = max(1, num_test_samples // num_chunks)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    
    for idx in range(0, num_test_samples, samples_per_chunk):
        # Get the features for test samples
        end_idx = min(idx + samples_per_chunk, num_test_samples)
        features = test_features[idx:end_idx, :]
        targets = test_labels[idx:end_idx]
        batch_size = targets.shape[0]

        # Calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
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
    
    # Evaluation parameters
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='/scratch/data/repos/dinosaud/outputs/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU?")
    parser.add_argument("--checkpoint_key", default="student", type=str,
        help='Key to use in the checkpoint (example: "student")')
    parser.add_argument('--dump_features', default="/scratch/ssd1/DINOSAUD_features",
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_spatial_labels = torch.load(os.path.join(args.load_features, "train_spatial_labels.pth"))
        test_spatial_labels = torch.load(os.path.join(args.load_features, "test_spatial_labels.pth"))
    else:
        # Extract features
        train_features, test_features, train_spatial_labels, test_spatial_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            for key in train_spatial_labels:
                train_spatial_labels[key] = train_spatial_labels[key].cuda()
                test_spatial_labels[key] = test_spatial_labels[key].cuda()

        print("Features are ready!\nStart the k-NN classification for spatial parameters.")
        
        # Evaluate each spatial parameter
        for spatial_type in ['azimuth', 'elevation', 'distance']:
            print(f"\n=== {spatial_type.upper()} Classification ===")
            for k in args.nb_knn:
                top1, top5 = knn_classifier_spatial(
                    train_features, train_spatial_labels,
                    test_features, test_spatial_labels, 
                    k, args.temperature, spatial_type=spatial_type
                )
                print(f"{k}-NN classifier result for {spatial_type}: Top1: {top1:.2f}%, Top5: {top5:.2f}%")
    
    dist.barrier()