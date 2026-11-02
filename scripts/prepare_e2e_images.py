"""Prepare OCTMNIST PNGs for integration tests, not medical evaluation."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'storage' / 'e2e-images'
URL = 'https://zenodo.org/records/10519652/files/octmnist.npz?download=1'
MD5 = 'c68d92d5b585d8d81f7112f81e2d0842'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'octmnist.npz'
    if not source.exists():
        partial = source.with_suffix('.part')
        print('Downloading official OCTMNIST (54.9 MB)...', flush=True)
        urlretrieve(URL, partial)
        if hashlib.md5(partial.read_bytes()).hexdigest() != MD5:
            raise RuntimeError('Download checksum mismatch; remove .part and retry.')
        partial.replace(source)
    if hashlib.md5(source.read_bytes()).hexdigest() != MD5:
        raise RuntimeError('Existing dataset checksum mismatch.')
    with np.load(source, allow_pickle=False) as data:
        for split, count, stage in [('train', 10, 'augmentation'), ('test', 2, 'inference')]:
            images, labels = data[split + '_images'], data[split + '_labels'].reshape(-1)
            for name, label in [('normal', 3), ('anomaly_cnv', 0)]:
                folder = ROOT / stage / name
                folder.mkdir(parents=True, exist_ok=True)
                indices = np.flatnonzero(labels == label)[:count]
                if len(indices) != count:
                    raise RuntimeError('Not enough samples for ' + name)
                for index in indices:
                    Image.fromarray(images[index]).save(folder / f'{split}_{index:06d}.png')
                print(folder, ':', count, 'PNGs', flush=True)
    (ROOT / 'SOURCE.txt').write_text(
        'OCTMNIST / MedMNIST. Official source: ' + URL + '\n'
        'Metadata: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py\n'
        'License: CC BY 4.0; retain dataset attribution when sharing.\n'
        'Normal = original class 3; anomaly = class 0 (CNV) only.\n'
        'Original 28x28 PNGs; server resizes them. Integration testing only.\n'
        'Augmentation uses train split; inference uses separate test split.\n'
        'Dataset reference: Yang et al., MedMNIST v2, Scientific Data 10, 41 (2023), '
        'https://doi.org/10.1038/s41597-022-01721-8\n', encoding='utf-8')


if __name__ == '__main__':
    main()
