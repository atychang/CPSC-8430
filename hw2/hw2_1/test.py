#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from data import (
    MsvdTestingDataset,
    build_vocab,
    build_word_index_map,
    get_video_feats,
)
from torch.utils.data import DataLoader
from utils import dir_path, is_text_file
from data import PAD, BOS, EOS, UNK

DATA_DIR = Path(__file__).parent / "data"
DATASET_DIR = DATA_DIR / "MLDS_hw2_1_data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "testing_data_dir", type=dir_path, help="The testing data directory"
    )
    parser.add_argument(
        "output_filename",
        type=is_text_file,
        help="The test data output filename (format:.txt)",
    )

    args = parser.parse_args()
    return args


def get_testing_data(testing_data_dir):
    word_count = build_vocab(
        [DATA_DIR / "training_label.json", DATA_DIR / "testing_label.json"]
    )
    _, index2word = build_word_index_map(word_count)
    video_feats = get_video_feats(Path(testing_data_dir))

    return MsvdTestingDataset(video_feats), index2word


def main(args):
    testing_data, index2word = get_testing_data(args.testing_data_dir)
    test_dataloader = DataLoader(dataset=testing_data)

    model = torch.load(DATA_DIR / "s2vt_model.pkl")
    model.eval()

    all_predictions = []
    for batch_idx, (vid, video_feats) in enumerate(test_dataloader):
        video_feats = video_feats.float()

        cap_out = model(video_feats)

        words = []
        for tensor in cap_out:
            word = tensor.item()
            if word in [PAD, BOS, EOS]:
                continue
            elif word == UNK:
                words.append("<UNK>")
            else:
                words.append(index2word[word])
            caption = " ".join(words)

        all_predictions.append((vid[0], caption))

    with open(args.output_filename, "w") as f:
        for id, cap in all_predictions:
            f.write(id + "," + cap + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
