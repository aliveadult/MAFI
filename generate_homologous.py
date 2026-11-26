"""Generate ten ESM-3 sequence variants per unique target in a MAFI CSV.

Input: a processed CSV with target_key and target_sequence columns.
Output: the same rows plus target_homologous_1 ... target_homologous_10.
Existing variants are preserved. No PDB structures or contact maps are created.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


# Replace this placeholder with your actual data directory, or use --data-dir.
DATA_DIR = Path('REPLACE_WITH_YOUR_ACTUAL_PATH').expanduser()
DATASET_NAME = 'davis'
MODEL_NAME = 'esm3-sm-open-v1'
NUM_STEPS = 30
TEMPERATURE = 0.8

VARIANT_COLUMNS = [f'target_homologous_{i}' for i in range(1, 11)]


def mask_segment(sequence, index):
    """Mask one of ten consecutive segments, including the final remainder."""
    start = index * len(sequence) // 10
    end = (index + 1) * len(sequence) // 10
    if start == end:
        raise ValueError('A sequence must have at least ten residues')
    return sequence[:start] + '_' * (end - start) + sequence[end:]


def save_csv(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def load_input(input_csv, output_csv):
    source = pd.read_csv(input_csv, dtype={'target_key': str})
    required = {'target_key', 'target_sequence'}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f'Missing input CSV columns: {sorted(missing)}')
    if source[['target_key', 'target_sequence']].isna().any().any():
        raise ValueError('Input CSV has empty target_key or target_sequence values')
    source['target_key'] = source['target_key'].str.strip()
    source['target_sequence'] = source['target_sequence'].str.strip().str.upper()
    if (source['target_key'] == '').any() or (source['target_sequence'] == '').any():
        raise ValueError('Input CSV has blank target_key or target_sequence values')
    for target_key, group in source.groupby('target_key', sort=False):
        if group['target_sequence'].nunique() != 1:
            raise ValueError(f'Conflicting sequences for target_key {target_key}')

    # Resume an existing output only when it has the same rows and targets.
    if output_csv.exists() and output_csv.resolve() != input_csv.resolve():
        frame = pd.read_csv(output_csv, dtype={'target_key': str})
        if len(frame) != len(source):
            raise ValueError('Existing output has a different row count')
        for column in ('target_key', 'target_sequence'):
            if column not in frame or not frame[column].fillna('').astype(str).equals(
                source[column].fillna('').astype(str)
            ):
                raise ValueError(f'Existing output does not match input column {column}')
    else:
        frame = source.copy()

    for column in VARIANT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--dataset', default=DATASET_NAME, choices=('davis', 'kiba'))
    parser.add_argument('--input-csv', type=Path,
                        help='Default: <data-dir>/<dataset>_processed.csv')
    parser.add_argument('--output-csv', type=Path,
                        help='Default: <data-dir>/<dataset>_processeds.csv')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-residues', type=int, default=1500)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--limit', type=int, help='Generate variants for the first N targets only.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate input and count missing variants without loading ESM-3.')
    args = parser.parse_args()
    if args.max_residues < 10 or args.save_every < 1 or (args.limit is not None and args.limit < 1):
        parser.error('max-residues must be >= 10; save-every and limit must be positive')
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    input_csv = args.input_csv or data_dir / f'{args.dataset}_processed.csv'
    output_csv = args.output_csv or data_dir / f'{args.dataset}_processeds.csv'
    if not input_csv.is_file():
        raise FileNotFoundError(f'Input CSV not found: {input_csv}')
    frame = load_input(input_csv, output_csv)
    targets = frame[['target_key', 'target_sequence']].drop_duplicates('target_key')
    if args.limit is not None:
        targets = targets.head(args.limit)

    tasks = []
    for target_key, sequence in targets.itertuples(index=False, name=None):
        rows = frame.index[frame['target_key'] == target_key]
        for index, column in enumerate(VARIANT_COLUMNS):
            values = frame.loc[rows, column].dropna().astype(str).str.strip()
            values = values[values != ''].unique().tolist()
            if len(values) > 1:
                raise ValueError(f'Conflicting existing values for {target_key}, {column}')
            if values:
                if len(values[0]) != len(sequence) or '_' in values[0]:
                    raise ValueError(f'Invalid existing variant for {target_key}, {column}')
                frame.loc[rows, column] = values[0]
            else:
                tasks.append((target_key, sequence, rows, index, column))

    print(f'Input: {input_csv}\nOutput: {output_csv}')
    print(f'Unique targets selected: {len(targets)}; missing variants: {len(tasks)}')
    if args.check_only:
        return
    if not tasks:
        save_csv(frame, output_csv)
        return

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from tqdm import tqdm

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    model = ESM3.from_pretrained(MODEL_NAME).eval().to(args.device)
    completed_since_save = 0
    failures = []
    try:
        with torch.inference_mode():
            for target_key, sequence, rows, index, column in tqdm(tasks, desc='ESM-3 variants'):
                try:
                    if not 10 <= len(sequence) <= args.max_residues:
                        raise ValueError(f'Sequence length {len(sequence)} is outside 10..{args.max_residues}')
                    masked = mask_segment(sequence, index)
                    generated = model.generate(
                        ESMProtein(sequence=masked),
                        GenerationConfig(track='sequence', num_steps=NUM_STEPS,
                                         temperature=TEMPERATURE),
                    )
                    if not isinstance(generated, ESMProtein) or not isinstance(generated.sequence, str):
                        raise RuntimeError('ESM-3 did not return a protein sequence')
                    variant = generated.sequence.upper()
                    if len(variant) != len(sequence) or '_' in variant:
                        raise RuntimeError('Generated sequence has a different length or unresolved masks')
                    frame.loc[rows, column] = variant
                    completed_since_save += 1
                    if completed_since_save >= args.save_every:
                        save_csv(frame, output_csv)
                        completed_since_save = 0
                except Exception as exc:
                    failures.append((target_key, column, str(exc)))
                    tqdm.write(f'{target_key} {column}: {exc}')
                    if args.device.startswith('cuda') and 'out of memory' in str(exc).lower():
                        torch.cuda.empty_cache()
    finally:
        # Preserve successfully generated variants even after interruption.
        save_csv(frame, output_csv)

    print(f'Saved {output_csv}; failed variants: {len(failures)}')
    if failures:
        for target_key, column, error in failures[:10]:
            print(f'  {target_key} {column}: {error}')
        raise RuntimeError('Some variants failed; rerun the script to resume missing values')


if __name__ == '__main__':
    main()
