import json
import re

import numpy as np
from torch.utils.data import Dataset

PAD: int = 0
BOS: int = 1
EOS: int = 2
UNK: int = 3


def build_vocab(file_paths, min_count=5):
    word_count = {}

    for file_path in file_paths:
        with open(file_path, "r") as f:
            videos = json.load(f)

        for video in videos:
            for caption in video["caption"]:
                caption = "<BOS> " + normalize_caption(caption) + " <EOS>"
                for word in caption.split():
                    word_count[word] = word_count.get(word, 0) + 1

    word_count = {
        word: count for word, count in word_count.items() if count >= min_count
    }

    return word_count


def build_word_index_map(word_count):
    word2index = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    index2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

    for idx, word in enumerate(word_count.keys()):
        word2index[word] = idx + 4
        index2word[idx + 4] = word

    return word2index, index2word


def normalize_caption(caption):
    caption = caption.lower()
    caption = re.sub("[.!,;?]]", " ", caption)
    return caption


def preprocess_caption(caption, word2index):
    caption = caption.split()
    caption = [word2index.get(word, UNK) for word in caption]
    caption = [BOS] + caption + [EOS]
    return caption


def get_video_feats(file_path):
    feat = {}
    for video in file_path.iterdir():
        if video.suffix != ".npy":
            continue
        try:
            video_feat = np.load(video)
        except ValueError:
            print(f"Error loading {video}")
            continue

        feat[video.stem] = video_feat

    return feat


def get_annotated_captions(file_path, word2index):
    with open(file_path, "r") as f:
        videos = json.load(f)

    annotated_captions = []
    for video in videos:
        for caption in video["caption"]:
            caption = normalize_caption(caption)
            caption = preprocess_caption(caption, word2index)
            annotated_captions.append((video["id"], caption))

    return annotated_captions


class MsvdTrainingDataset(Dataset):
    def __init__(self, video_feats, annotated_captions):
        self.video_feats = video_feats
        self.annotated_captions = annotated_captions

    def __len__(self):
        return len(self.annotated_captions)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        vid, caption = self.annotated_captions[idx]
        return self.video_feats[vid], caption


class MsvdTestingDataset(Dataset):
    def __init__(self, video_feats):
        self.video_feats = list(video_feats.items())

    def __len__(self):
        return len(self.video_feats)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        return self.video_feats[idx]


def minibatch(data):
    video_feats = [d[0] for d in data]
    captions = [d[1] for d in data]
    max_len = 80
    cap_mask = [[1.0] * len(cap) + [0.0] * (max_len - len(cap)) for cap in captions]
    caps = []
    for cap in captions:
        n_word = len(cap)
        cap = cap + [PAD] * (max_len - n_word)
        caps.append(cap)
    return np.array(video_feats), np.array(caps), np.array(cap_mask)
