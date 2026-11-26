"""Encode the original sequence and ten ESM-3 variants already in a wide CSV.

Output columns: target_key, variant_index, sequence, esm2_embedding.
The last column is a JSON array of 1280 floats. Existing output is resumable.
"""
import argparse
import json
import os
from pathlib import Path

import esm
import pandas as pd
import torch
from tqdm import tqdm

SEQUENCE_COLUMNS = ["target_sequence"] + [f"target_homologous_{i}" for i in range(1, 11)]
OUTPUT_COLUMNS = ["target_key", "variant_index", "sequence", "esm2_embedding"]


def save_atomic(records, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description="Generate pooled ESM-2 embeddings from MAFI wide CSV")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.save_every < 1 or not 1 <= args.max_length <= 1022:
        parser.error("batch-size and save-every must be positive; max-length must be 1..1022")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source = pd.read_csv(args.input_csv, dtype={"target_key": str})
    required = ["target_key", *SEQUENCE_COLUMNS]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Input CSV is missing: {missing}")

    # Affinity CSV repeats each target for many drugs. Embed each target once.
    targets = source[required].dropna(subset=["target_key"]).drop_duplicates("target_key")
    tasks = []
    for row in targets.itertuples(index=False, name=None):
        target_key = str(row[0])
        for variant_index, raw_sequence in enumerate(row[1:]):
            if pd.isna(raw_sequence) or not str(raw_sequence).strip():
                continue  # Useful when processing a deliberately partial CSV.
            tasks.append((target_key, variant_index, str(raw_sequence).strip()))

    destination = Path(args.output_csv)
    if destination.exists():
        existing = pd.read_csv(destination, dtype={"target_key": str})
        if not set(OUTPUT_COLUMNS).issubset(existing.columns):
            raise ValueError("Existing output CSV has an incompatible schema")
        records = existing[OUTPUT_COLUMNS].to_dict("records")
    else:
        records = []
    finished = {}
    for record in records:
        key = (str(record["target_key"]), int(record["variant_index"]))
        finished[key] = str(record["sequence"])
    pending = []
    for target_key, variant_index, sequence in tasks:
        key = (target_key, variant_index)
        if key in finished:
            if finished[key] != sequence:
                raise ValueError(f"Sequence changed since previous run: {key}")
        else:
            pending.append((target_key, variant_index, sequence))
    if not pending:
        print(f"All {len(tasks)} embeddings already exist in {destination}")
        return

    print(f"Loading ESM-2 650M on {args.device}; {len(pending)} sequences to encode")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()
    completed_since_save = 0
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(pending), args.batch_size), desc="ESM-2 batches"):
                batch = pending[start:start + args.batch_size]
                clipped = [sequence[:args.max_length] for _, _, sequence in batch]
                _, _, tokens = batch_converter([(str(i), seq) for i, seq in enumerate(clipped)])
                result = model(tokens.to(args.device), repr_layers=[33], return_contacts=False)
                representations = result["representations"][33]
                for index, (target_key, variant_index, sequence) in enumerate(batch):
                    length = len(clipped[index])
                    embedding = representations[index, 1:length + 1].mean(dim=0)
                    vector = embedding.float().cpu().tolist()
                    if len(vector) != 1280:
                        raise RuntimeError(f"Expected 1280 dimensions, got {len(vector)}")
                    records.append({
                        "target_key": target_key,
                        "variant_index": variant_index,
                        "sequence": sequence,
                        "esm2_embedding": json.dumps(vector, separators=(",", ":")),
                    })
                    completed_since_save += 1
                if completed_since_save >= args.save_every:
                    save_atomic(records, destination)
                    completed_since_save = 0
    finally:
        if completed_since_save:
            save_atomic(records, destination)
    print(f"Saved {len(records)} embeddings to {destination}")


if __name__ == "__main__":
    main()
