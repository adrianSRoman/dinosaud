# Forked from DINO repo https://github.com/facebookresearch/dino

Nothing but experiments...

```
python main_dinosaud.py \
    --reverb_type mic \
    --backbone soundrain \
    --n_q 4 \
    --codebook_size 1024 \
    --D 32 \
    --C 32 \
    --out_dim 65536 \
    --batch_size_per_gpu 8 \
    --epochs 100 \
    --lr 0.0005 \
    --warmup_epochs 10 \
    --output_dir ./outputs \
    --num_workers 8 \
    --sample_rate 44100 \
    --audio_length_seconds 1
```
