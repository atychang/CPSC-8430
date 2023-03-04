# Homework 2 - Video Caption Generation

## Usage

```bash
pip install -r requirements.txt
```

### For training

```bash
./hw2/hw2_1/train.py
```

This script would check whether `MLDS_hw2_1_data.tar.gz` is existing in `data/` directory. If not, it would download `MLDS_hw2_1_data.tar.gz` to `data/` from Google Drive and extract it to `data/`.

### For testing

```bash
./hw2_seq2seq.sh hw2/hw2_1/data/MLDS_hw2_1_data/testing_data/feat testset_output.txt
```
