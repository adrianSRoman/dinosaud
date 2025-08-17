import torch
from torch.utils.data import Dataset
import json
import numpy as np
import random
import os
import soundfile as sf
import torchaudio
from scipy import signal
import torch.nn.functional as F


def normalize_audio(waveform, target_lufs=-14.0):
    """Normalize audio to target LUFS"""
    # Simple RMS-based normalization as approximation
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 0:
        target_rms = 10 ** (target_lufs / 20)
        waveform = waveform * (target_rms / rms)
    return waveform


class SpatialAudioDataset(Dataset):
    def __init__(
        self, 
        audio_json, 
        reverb_json, 
        audio_path_root,
        reverb_path_root,
        reverb_type='mic',
        sample_rate=44100,
        audio_length_seconds=5,
        teacher_num=2,
        student_num=4,
        normalize=True,
        _ext_audio=".wav",
        mode="train"
    ):
        # Load audio metadata
        with open(audio_json, 'r') as f:
            self.audio_data = json.load(f)
        
        # Load reverb metadata with groups
        with open(reverb_json, 'r') as f:
            reverb_metadata = json.load(f)
            self.reverb_groups = reverb_metadata['groups']
        
        self.audio_path_root = audio_path_root
        self.reverb_path_root = reverb_path_root
        self.reverb_type = reverb_type
        self.sample_rate = sample_rate
        self.audio_length_seconds = audio_length_seconds
        self.audio_length_samples = sample_rate * audio_length_seconds
        self.normalize = normalize
        self._ext_audio = _ext_audio
        self.mode = mode
        self.teacher_num = teacher_num # number of augmentations for teacher
        self.student_num = student_num # number of augmentations for student
        
        # Determine number of channels based on reverb type
        self.channel_num = 2 if reverb_type == 'binaural' else 4 if reverb_type == 'mic' else 1
        
        # Convert groups dict to list for easier sampling
        self.group_list = list(self.reverb_groups.values())

        # Filter groups to only include those with at least 6 reverbs
        self.group_list = [group for group in self.group_list if len(group['rooms']) >= (self.teacher_num + self.student_num)]
        
        if not self.group_list:
            raise ValueError("No groups have at least 6 reverbs")
        
        print(f'---------------the {mode} dataloader---------------')
        print(f'Audio samples: {len(self.audio_data)}')
        print(f'Reverb groups: {len(self.reverb_groups)}')
        print(f'Channel count: {self.channel_num}')
        print(f'Sample rate: {self.sample_rate}')
        print(f'Audio length: {self.audio_length_seconds}s')

    def load_audio(self, audio_item):
        """Load and preprocess audio file"""
        audio_path = os.path.join(
            self.audio_path_root, 
            audio_item['folder'], 
            audio_item['id'] + self._ext_audio
        )
        
        # Load audio
        waveform, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0]
        
        # Resample if necessary
        if sr != self.sample_rate:
            waveform = signal.resample_poly(waveform, self.sample_rate, sr)
        
        # Normalize audio
        if self.normalize:
            waveform = normalize_audio(torch.from_numpy(waveform), -14.0).numpy()
        
        # Convert to tensor and reshape
        waveform = torch.from_numpy(waveform).reshape(1, -1).float()
        
        # Pad or trim to fixed length
        padding = self.audio_length_samples - waveform.shape[1]
        if padding > 0:
            waveform = F.pad(waveform, (0, padding), 'constant', 0)
        elif padding < 0:
            waveform = waveform[:, :self.audio_length_samples]
            
        return waveform

    def load_reverb(self, reverb_item):
        """Load reverb impulse response"""
        reverb_path = os.path.join(
            self.reverb_path_root, 
            reverb_item['fname']
        )
        
        # Load reverb
        reverb = torch.from_numpy(np.load(reverb_path)).float()
        
        # Ensure reverb has correct shape [channels, time]
        if reverb.dim() == 3:
            reverb = reverb.squeeze(1)
        
        # Pad or trim reverb to 2 seconds
        reverb_length = self.sample_rate * 2
        reverb_padding = reverb_length - reverb.shape[1]
        if reverb_padding > 0:
            reverb = F.pad(reverb, (0, reverb_padding), 'constant', 0)
        elif reverb_padding < 0:
            reverb = reverb[:, :reverb_length]
            
        return reverb

    def get_spatial_targets(self, reverb_item):
        """Extract spatial information from reverb metadata"""
        # Distance classes (discretized)
        distance = reverb_item['distance']
        distance_class = min(int(distance * 10), 20)  # 21 classes (0-20)
        
        # Azimuth (0-359 degrees)
        azimuth = reverb_item['azimuth']
        azimuth_class = int((azimuth + 180) % 360)  # Convert [-180, 180] to [0, 359]
        
        # Elevation (0-179 degrees) 
        elevation = reverb_item['elevation']
        elevation_class = int(elevation + 90)  # Convert [-90, 90] to [0, 179]
        elevation_class = max(0, min(179, elevation_class))
        
        return {
            "distance": distance_class,
            "azimuth": azimuth_class, 
            "elevation": elevation_class
        }

    def sample_reverbs_from_group(self, group, num_reverbs, exclude=None):
        """Sample reverbs from a group, optionally excluding certain reverbs"""
        available_reverbs = group['rooms']
        
        if exclude: # ensure if we sample again, we get disjoint reverbs
            available_reverbs = [r for r in available_reverbs if r not in exclude]
        
        if len(available_reverbs) < num_reverbs:
            # If not enough reverbs, sample with replacement
            return random.choices(available_reverbs, k=num_reverbs)
        else:
            return random.sample(available_reverbs, num_reverbs)

    def __getitem__(self, index):
        """
        Returns:
        dict: Contains 'teacher_data' and 'student_data' with reverbs and spatial targets
        """
        # Get audio item
        audio_item = self.audio_data[index]
        # Load base audio
        base_audio = self.load_audio(audio_item)
        
        # Sample a random group for this audio sample
        selected_group = random.choice(self.group_list)
        
        # Sample teacher reverbs first
        teacher_reverbs_meta = self.sample_reverbs_from_group(selected_group, self.teacher_num)
        
        # Sample student reverbs, excluding teacher reverbs
        student_reverbs_meta = self.sample_reverbs_from_group(
            selected_group, self.student_num, exclude=teacher_reverbs_meta
        )
        
        # Load teacher reverbs and create convolved audio
        teacher_reverbs = []
        teacher_spatial_targets = []
        teacher_audio = []
        
        for reverb_meta in teacher_reverbs_meta:
            reverb = self.load_reverb(reverb_meta)
            teacher_reverbs.append(reverb)
            teacher_spatial_targets.append(self.get_spatial_targets(reverb_meta))
            
            # Convolve audio with reverb
            convolved = torchaudio.functional.fftconvolve(
                base_audio, reverb, mode='full'
            )[..., :base_audio.shape[-1]]
            teacher_audio.append(convolved)
        
        # Load student reverbs and create convolved audio  
        student_reverbs = []
        student_spatial_targets = []
        student_audio = []
        
        for reverb_meta in student_reverbs_meta:
            reverb = self.load_reverb(reverb_meta)
            student_reverbs.append(reverb)
            student_spatial_targets.append(self.get_spatial_targets(reverb_meta))
            
            # Convolve audio with reverb
            convolved = torchaudio.functional.fftconvolve(
                base_audio, reverb, mode='full'
            )[..., :base_audio.shape[-1]]
            student_audio.append(convolved)
        
        # Get label
        label = audio_item['label']
        
        return {
            'teacher_audio': teacher_audio,  # List of 2 convolved audio tensors
            'student_audio': student_audio,  # List of 4 convolved audio tensors  
            'teacher_spatial_targets': teacher_spatial_targets,  # List of 2 spatial target dicts
            'student_spatial_targets': student_spatial_targets,  # List of 4 spatial target dicts
            'label': label,
            'audio_id': audio_item['id'],
            'group_id': selected_group['group_id']
        }

    def __len__(self):
        return len(self.audio_data)

    def collate_fn(self, batch):
        """Custom collate function for batching"""
        # Separate teacher and student data
        teacher_audio_batch = []
        student_audio_batch = []
        teacher_spatial_batch = []
        student_spatial_batch = []
        labels = []
        audio_ids = []
        group_ids = []
        
        for item in batch:
            # Stack teacher audio (2 per sample)
            teacher_audio_batch.extend(item['teacher_audio'])
            # Stack student audio (4 per sample)  
            student_audio_batch.extend(item['student_audio'])
            
            teacher_spatial_batch.extend(item['teacher_spatial_targets'])
            student_spatial_batch.extend(item['student_spatial_targets'])
            
            labels.append(item['label'])
            audio_ids.append(item['audio_id'])
            group_ids.append(item['group_id'])
        
        # Stack audio tensors
        teacher_audio_tensor = torch.stack(teacher_audio_batch)  # [batch_size*2, channels, time]
        student_audio_tensor = torch.stack(student_audio_batch)  # [batch_size*4, channels, time]
        
        # Convert spatial targets to tensors
        def collate_spatial_targets(spatial_list):
            distances = torch.LongTensor([s['distance'] for s in spatial_list])
            azimuths = torch.LongTensor([s['azimuth'] for s in spatial_list]) 
            elevations = torch.LongTensor([s['elevation'] for s in spatial_list])
            return {'distance': distances, 'azimuth': azimuths, 'elevation': elevations}
        
        teacher_spatial_tensor = collate_spatial_targets(teacher_spatial_batch)
        student_spatial_tensor = collate_spatial_targets(student_spatial_batch)
        
        return {
            'teacher_audio': teacher_audio_tensor,
            'student_audio': student_audio_tensor,
            'teacher_spatial_targets': teacher_spatial_tensor,
            'student_spatial_targets': student_spatial_tensor,
            'labels': torch.LongTensor(labels),
            'audio_ids': audio_ids,
            'group_ids': torch.LongTensor(group_ids)
        }