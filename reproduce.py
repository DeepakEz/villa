"""Reproduce the GrowPatch resume-anchor bug with two Linux VC3D CLI builds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from verify_anchor_fix import VOLUME, load_mesh


def environment(library_path):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'CUDA_VISIBLE_DEVICES': '-1'}
    if library_path:
        env['LD_LIBRARY_PATH'] = library_path
    return env


def run_stock(binary, fixture, out, name, correction, env):
    mesh = out / f'{name}.tifxyz'
    command = [
        str(binary), '--volume', VOLUME, '--target-dir', str(mesh),
        '--params', str(fixture/'replay_params.json'), '--resume', str(fixture/'untrimmed.tifxyz'),
        '--correct', str(fixture/correction), '--segment-name', name, '--rng-seed', '3915',
    ]
    with (out/f'{name}.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stock', 'patched', 'out'):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument('--stock-library-path')
    parser.add_argument('--patched-library-path')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    package = Path(__file__).resolve().parent
    fixture = package/'fixture'
    baseline = load_mesh(fixture/'untrimmed.tifxyz')
    anchor = json.loads((fixture/'correction.json').read_text(encoding='utf-8'))['collections']['1']
    col, row = anchor['anchor2d']
    target = np.asarray(anchor['points']['1']['p'])
    stock_env = environment(args.stock_library_path)
    stock = run_stock(args.stock, fixture, args.out, 'stock', 'correction.json', stock_env)
    translated = run_stock(args.stock, fixture, args.out, 'translated', 'correction_padded_grid_control.json', stock_env)
    stock_mesh = load_mesh(stock)
    stock_seed_motion = np.linalg.norm(stock_mesh[:, :24] - baseline[:, :24], axis=-1)
    if not np.array_equal(stock_mesh[row, col], baseline[row, col]) or not stock_seed_motion.any():
        raise RuntimeError('Stock build did not reproduce the unchanged anchor and displaced seed')

    patched_dir = args.out/'patched'
    subprocess.run([
        sys.executable, str(package/'verify_anchor_fix.py'), str(args.patched), str(fixture), str(patched_dir),
        '--expected-mesh', str(translated),
    ], env=environment(args.patched_library_path), check=True)
    report = {
        'stock_anchor_error_vox': float(np.linalg.norm(stock_mesh[row, col] - target)),
        'stock_seed_vertices_moved': int(np.count_nonzero(stock_seed_motion)),
        'stock_seed_max_motion_vox': float(stock_seed_motion.max()),
        'patched': json.loads((patched_dir/'regression.json').read_text(encoding='utf-8')),
    }
    (args.out/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
