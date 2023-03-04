#!/usr/bin/env python
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from data import (
    MsvdTrainingDataset,
    build_vocab,
    build_word_index_map,
    get_annotated_captions,
    get_video_feats,
    minibatch,
)
from model_seq2seq import S2VT
from torch.autograd import Variable
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).parent / "data"
DATASET_DIR = DATA_DIR / "MLDS_hw2_1_data"
DATASET_COMPRESSED_FILE = DATA_DIR / "MLDS_hw2_1_data.tar.gz"

EPOCH = 200
BATCH_SIZE = 10
LEARNING_RATE = 1e-4


def check_dataset_is_downloaded():
    if DATASET_COMPRESSED_FILE.is_file():
        print("Dataset already exists.\n")
    else:
        print("Downloading dataset...")

        import gdown

        url = "https://drive.google.com/uc?id=1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ"
        output = DATASET_COMPRESSED_FILE.name
        gdown.download(url, output, quiet=False)

        print("Download complete.\n")


def check_dataset_is_extracted():
    if (DATASET_DIR / "training_label.json").is_file():
        print("Dataset already extracted.\n")
    else:
        print("Extracting dataset...")

        import tarfile

        with tarfile.open(DATASET_COMPRESSED_FILE) as tar:
            tar.extractall(DATA_DIR)

        print("Extraction complete.\n")


def get_training_data():
    word_count = build_vocab(
        [DATASET_DIR / "training_label.json", DATASET_DIR / "testing_label.json"]
    )
    word2index, index2word = build_word_index_map(word_count)
    video_feats = get_video_feats(DATASET_DIR / "training_data" / "feat")
    annotated_captions = get_annotated_captions(
        DATASET_DIR / "training_label.json", word2index
    )

    return MsvdTrainingDataset(video_feats, annotated_captions), word2index, index2word


def calculate_loss(x, y, lengths, loss_fn):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss / batch_size

    return loss


def train(model, epoch, data_loader, loss_func):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for batch_idx, (video_feats, captions, cap_mask) in enumerate(data_loader):
        video_feats, captions, cap_mask = (
            torch.FloatTensor(video_feats),
            torch.LongTensor(captions),
            torch.FloatTensor(cap_mask),
        )

        # Compute prediction error
        cap_out = model(video_feats, captions)
        cap_labels = captions[:, 1:].contiguous().view(-1)
        cap_mask = cap_mask[:, 1:].contiguous().view(-1)
        logit_loss = loss_func(cap_out, cap_labels)
        masked_loss = logit_loss * cap_mask
        loss = torch.sum(masked_loss) / torch.sum(cap_mask)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(video_feats),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )


def main():
    check_dataset_is_downloaded()
    check_dataset_is_extracted()

    training_data, word2index, index2word = get_training_data()
    training_data_loader = DataLoader(
        dataset=training_data,
        batch_size=BATCH_SIZE,
        collate_fn=minibatch,
    )

    model = S2VT(len(index2word), word2index)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        start_time = time.time()
        train(model, epoch + 1, training_data_loader, loss_func)
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\nTraining complete.\n")
    torch.save(model, DATA_DIR / "s2vt_model.pkl")
    print("Model saved.\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Total %s seconds ---" % (time.time() - start_time))
