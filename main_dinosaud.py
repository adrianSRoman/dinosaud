# Code adapted from https://github.com/facebookresearch/dino

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random

import utils
from model.soundrain import SoundRain
from dataset.dataloader import Dataset
from util.utils import sample_fixed_length_data_aligned

def get_args_parser():
    parser = argparse.ArgumentParser('Audio DINO', add_help=False)

    # Model parameters
    parser.add_argument('--backbone', default='soundrain', type=str, choices=['soundrain'],
        help="""Name of audio backbone architecture to train.""")
    parser.add_argument('--n_q', default=4, type=int, help='Number of quantizers in SoundRain')
    parser.add_argument('--codebook_size', default=1024, type=int, help='Codebook size in SoundRain')
    parser.add_argument('--D', default=32, type=int, help='Embedding dimension in SoundRain')
    parser.add_argument('--C', default=32, type=int, help='Channel dimension in SoundRain')
    parser.add_argument('--strides', default=[2, 4, 5, 8], type=int, nargs='+', help='Strides in SoundRain')
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Audio parameters
    parser.add_argument('--sample_rate', default=16000, type=int, help='Audio sample rate')
    parser.add_argument('--sample_length', default=16384, type=int, help='Fixed length for audio samples')
    parser.add_argument('--n_channels', default=4, type=int, help='Number of audio channels')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct audio samples loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters for audio
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range for global crops (fraction of total audio length).""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local crops to generate. Set to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range for local crops (fraction of total audio length).""")

    # Misc
    parser.add_argument('--data_path', default='/scratch/data/repos/dinosaud/dataset/val.hdf5', type=str,
        help='Please specify path to the audio dataset list file.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


class AudioDINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class AudioMultiCropWrapper(nn.Module):
    """
    Wrapper for audio multi-crop processing
    """
    def __init__(self, backbone, head):
        super(AudioMultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # x is a list of audio crops
        # Each crop shape: [batch_size, n_channels, sample_length]

        if not isinstance(x, list):
            x = [x]
        
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            # Process crops of the same size together
            _out = self.backbone(torch.cat(x[start_idx: end_idx]), mode='encode')
            # Global average pooling over time dimension
            _out = torch.mean(_out, dim=1)  # [batch_size, D]
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        
        # Apply projection head
        return self.head(output)


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationAudioDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.sample_length,
        args.sample_rate
    )
    dataset = Dataset(
        hdf5_path=args.data_path, transform=None
    )
    
    # Create wrapper dataset for transforms
    dataset = AudioDatasetWrapper(dataset, transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} audio samples.")

    # ============ building student and teacher networks ... ============
    if args.backbone == 'soundrain':
        student_backbone = SoundRain(
            n_q=args.n_q,
            codebook_size=args.codebook_size,
            D=args.D,
            C=args.C,
            strides=args.strides
        )
        teacher_backbone = SoundRain(
            n_q=args.n_q,
            codebook_size=args.codebook_size,
            D=args.D,
            C=args.C,
            strides=args.strides
        )
        embed_dim = args.D
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Wrap with multi-crop processing and DINO head
    student = AudioMultiCropWrapper(student_backbone, AudioDINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = AudioMultiCropWrapper(teacher_backbone, AudioDINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head
    ))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    
    student = nn.parallel.DistributedDataParallel(
            student, 
            device_ids=[args.gpu], 
            find_unused_parameters=True
    )
    
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.backbone} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting Audio DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (audio_crops, _, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move audio crops to gpu
        audio_crops = [crop.cuda(non_blocking=True) for crop in audio_crops]
        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(audio_crops[:2])  # only the 2 global crops pass through the teacher
            student_output = student(audio_crops)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class AudioDatasetWrapper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mic_sig, foa_sig, filename = self.dataset[idx]
        audio_crops = self.transform(mic_sig)
        return audio_crops, foa_sig, filename


class DataAugmentationAudioDINO:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, sample_length, sample_rate):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.sample_length = sample_length
        self.sample_rate = sample_rate

    def __call__(self, audio):
        # audio: [n_channels, time_samples]
        crops = []
        
        # Global crops
        for _ in range(2):
            crop = audio.astype(np.float32) # NOTE: for now no augmentation and make the data float32
            #crop = self.random_crop(audio, self.global_crops_scale)
            #crop = self.apply_augmentations(crop)
            crops.append(crop)
        
        # Local crops
        for _ in range(self.local_crops_number):
            crop = audio.astype(np.float32) # NOTE: for now no augmentation and make the data float32
            #crop = self.random_crop(audio, self.local_crops_scale)
            #crop = self.apply_augmentations(crop)
            crops.append(crop)
        
        return crops

    def random_crop(self, audio, scale_range):
        # audio: [n_channels, time_samples]
        n_channels, total_samples = audio.shape
        
        # Determine crop length
        min_scale, max_scale = scale_range
        scale = random.uniform(min_scale, max_scale)
        crop_length = int(total_samples * scale)
        crop_length = max(crop_length, 1024)  # Minimum crop length
        
        # Random start position
        if crop_length >= total_samples:
            start_pos = 0
            crop_length = total_samples
        else:
            start_pos = random.randint(0, total_samples - crop_length)
        
        # Extract crop
        crop = audio[:, start_pos:start_pos + crop_length]
        
        # Pad or trim to fixed length for consistent processing
        if crop_length < self.sample_length:
            # Pad with zeros
            pad_length = self.sample_length - crop_length
            crop = F.pad(crop, (0, pad_length), mode='constant', value=0)
        elif crop_length > self.sample_length:
            # Trim to sample_length
            crop = crop[:, :self.sample_length]
        
        return crop

    def apply_augmentations(self, audio):
        # audio: [n_channels, time_samples]
        audio = torch.from_numpy(audio).to(torch.float32)
        
        # Time shifting
        if random.random() < 0.5:
            shift = random.randint(-audio.shape[1]//10, audio.shape[1]//10)
            audio = torch.roll(audio, shift, dims=1)
        
        # Amplitude scaling
        if random.random() < 0.8:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        # Add noise
        if random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.01
            audio = audio + noise
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        return audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO Spatial Audio', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
