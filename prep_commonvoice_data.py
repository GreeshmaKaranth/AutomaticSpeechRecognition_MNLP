#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torchaudio
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import regex as re
import string
import torch
import numpy as np

import pandas as pd
from typing import Any, Dict, List, Optional, Union

from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import COMMONVOICE
from datasets import load_dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import (
    convert_waveform, _get_kaldi_fbank, _get_torchaudio_fbank, is_npy_data,
    is_sf_audio_data
)

log = logging.getLogger(__name__)

SPLITS = [
    "train",
    "test",
    "validation",
    "invalidated"
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


# PATH_TO_DATASET = "/content/fairseq/examples/speech_to_text/cv-corpus-15.0-2023-09-08/gn"

def asr_normalize(text):
    return " ".join(re.sub(r'\([^ ]*\)', '', text).translate(str.maketrans('', '', string.punctuation)).lower().split())


def process(args):
    out_root = Path(args.output_root).absolute()
    input_dir = Path(args.input_dir).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)

    for split in SPLITS:
        print(f"Fetching split {split}...")
        # dataset = COMMONVOICE(input_dir.as_posix(), tsv=f"{split}.tsv")
        # for wav, sample_rate, sample in tqdm(dataset):
        #   print(wav, wav.shape)
        #   path = sample["path"]
        #   client_id = sample["client_id"]
        #   sentence = sample["sentence"]
        #   sample_id = f"{client_id}-{path}"

        #   extract_fbank_features(
        #         wav, sample_rate, feature_root / f"{sample_id}.npy"
        #     )

        dataset = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split=split, token=True)
        # print(dataset)
        print("Extracting log mel filter bank features...")
        for sample in tqdm(dataset):
            path = sample["path"]
            file_name = path.split('/')[-1]

            client_id = sample["client_id"]
            sentence = sample["sentence"]
            sample_id = f"{client_id}_{file_name}"

            sample_rate = sample["audio"]["sampling_rate"]
            wav = torch.FloatTensor(sample["audio"]["array"].reshape(1, -1))

            features = extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )

    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        # dataset = COMMONVOICE(input_dir.as_posix(), tsv=f"{split}.tsv")
        dataset = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split=split, token=True)
        for sample in tqdm(dataset):
            path = sample["path"]
            file_name = path.split('/')[-1]

            client_id = sample["client_id"]
            sentence = sample["sentence"]
            sample_id = f"{client_id}_{file_name}"

            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(sentence.lower())
            manifest["speaker"].append(client_id)

        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--input-dir", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
