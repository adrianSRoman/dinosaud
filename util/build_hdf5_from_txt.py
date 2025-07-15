import soundfile as sf
import numpy as np
import os
import h5py
from tqdm import tqdm
import argparse

def sample_fixed_length_data_aligned(data_a, data_b, sample_length, hop_length=None):
    if hop_length is None:
        hop_length = sample_length  # no overlap

    frames_total = data_a.shape[1]
    num_segments = max(1, int(np.ceil((frames_total - sample_length) / hop_length)) + 1)

    mic_segments = []
    foa_segments = []

    for i in range(num_segments):
        start = i * hop_length
        end = start + sample_length
        mic_chunk = data_a[:, start:end]
        foa_chunk = data_b[:, start:end]

        # Zero-pad if needed
        if mic_chunk.shape[1] < sample_length:
            pad_width = sample_length - mic_chunk.shape[1]
            mic_chunk = np.pad(mic_chunk, ((0, 0), (0, pad_width)))
            foa_chunk = np.pad(foa_chunk, ((0, 0), (0, pad_width)))

        mic_segments.append(mic_chunk)
        foa_segments.append(foa_chunk)

    return np.stack(mic_segments), np.stack(foa_segments)

def build_hdf5_from_txt(txt_path, out_path, sample_length=16384, hop_length=None):
    with open(txt_path, 'r') as f:
        lines = [line.strip().split() for line in f]

    mic_data = []
    foa_data = []
    names = []

    for mic_path, foa_path in tqdm(lines, desc="Processing files"):
        mic_path = os.path.expanduser(mic_path)
        foa_path = os.path.expanduser(foa_path)

        mic, _ = sf.read(mic_path, always_2d=True)
        foa, _ = sf.read(foa_path, always_2d=True)

        mic = mic.T  # (C, T)
        foa = foa.T

        mic_chunks, foa_chunks = sample_fixed_length_data_aligned(mic, foa, sample_length, hop_length)

        mic_data.append(mic_chunks)
        foa_data.append(foa_chunks)
        names.extend([os.path.basename(mic_path)] * mic_chunks.shape[0])

    mic_data = np.concatenate(mic_data, axis=0)  # (N, C, T)
    foa_data = np.concatenate(foa_data, axis=0)

    print(f"Saving to {out_path}, total chunks: {mic_data.shape[0]}")

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('mic', data=mic_data, compression='gzip')
        f.create_dataset('foa', data=foa_data, compression='gzip')
        f.create_dataset('filenames', data=np.string_(names), dtype=h5py.string_dtype())

def main():
    parser = argparse.ArgumentParser(description="Convert paired audio paths to HDF5 chunks.")
    parser.add_argument('--txt_path', type=str, required=True, help='Path to text file with paired mic/foa paths.')
    parser.add_argument('--out_path', type=str, required=True, help='Output HDF5 file path.')
    parser.add_argument('--sample_length', type=int, default=16384, help='Number of samples per chunk.')
    parser.add_argument('--hop_length', type=int, default=None, help='Hop length between chunks (overlap if < sample_length).')

    args = parser.parse_args()

    build_hdf5_from_txt(
        txt_path=args.txt_path,
        out_path=args.out_path,
        sample_length=args.sample_length,
        hop_length=args.hop_length
    )

if __name__ == '__main__':
    main()
