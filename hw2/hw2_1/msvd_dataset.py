import pandas as pd
from torch.utils.data import Dataset


class MsvdDataset(Dataset):
    def __init__(self, label_file, avi_dir):
        self.avi_labels = pd.read_json(label_file)
        self.avi_dir = avi_dir
