"""Replay one public CT correction using original surface-grid coordinates."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import tifffile


VOLUME = (
    'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/'
    'volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr'
)


def load_mesh(path):
    points = np.stack([tifffile.imread(path / f'{axis}.tif') for axis in 'xyz'], axis=-1)
    visible = (np.isfinite(points) & (points >= 0)).all(axis=-1)
    row, col = np.nonzero(visible)
    cropped = points[row.min():row.max()+1, col.min():col.max()+1].astype(float)
    if cropped.shape != (24, 64, 3) or not (np.isfinite(cropped) & (cropped >= 0)).all():
        raise ValueError(f'Unexpected mesh dimensions or validity in {path}')
    return cropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--volume', default=VOLUME)
    parser.add_argument('--expected-mesh', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    config = json.loads((args.fixture / 'replay_params.json').read_text(encoding='utf-8'))
    corrections = args.fixture.resolve() / 'correction.json'
    anchor = json.loads(corrections.read_text(encoding='utf-8'))['collections']['1']
    col, row = anchor['anchor2d']
    target = np.asarray(anchor['points']['1']['p'])
    source = args.fixture.resolve() / 'untrimmed.tifxyz'
    orig = load_mesh(source)
    before = float(np.linalg.norm(orig[row, col] - target))
    expected = load_mesh(args.expected_mesh) if args.expected_mesh else None
    reports = {}
    for name, extra_cols, extra_rows in (('default', 0, 0), ('asymmetric', 7, 3)):
        params = {**config, 'grow_max_extra_cols': extra_cols, 'grow_max_extra_rows': extra_rows}
        config_path = args.out / f'{name}.json'
        config_path.write_text(json.dumps(params, indent=2), encoding='utf-8')
        log_path = args.out / f'{name}.log'
        output = args.out / f'{name}.tifxyz'
        command = [str(args.binary.resolve()), '--volume', args.volume, '--target-dir', str(output),
                   '--params', str(config_path), '--resume', str(source), '--correct', str(corrections),
                   '--segment-name', name, '--rng-seed', '3915']
        with log_path.open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        written = load_mesh(output)
        if expected is None:
            expected = written
        residual = float(np.linalg.norm(written[row, col] - target))
        expected_grid = f'work grid: [{col+25+extra_cols}, {row+25+extra_rows}]'
        translated = expected_grid in log_path.read_text(encoding='utf-8', errors='replace')
        same_seed = bool(np.array_equal(written[:, :24], orig[:, :24]))
        reports[name] = {'before_vx': before, 'after_vx': residual, 'seed_unchanged': same_seed,
                         'expected_trace': expected_grid, 'trace_present': translated,
                         'full_mesh_equal': bool(np.array_equal(written, expected))}
        print(name, reports[name], flush=True)
    (args.out / 'regression.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    if not all(d['seed_unchanged'] and d['after_vx'] < d['before_vx']/2 and d['trace_present'] and d['full_mesh_equal']
               for d in reports.values()):
        raise SystemExit('Anchor correction regression failed')


if __name__ == '__main__':
    main()
