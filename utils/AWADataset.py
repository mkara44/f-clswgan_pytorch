import torch
from torch.utils.data import Dataset
import scipy.io as sio

# Additional Scripts
from config import cfg


class AWADataset(Dataset):
    res_mat = sio.loadmat(cfg.res_path)
    atts_mat = sio.loadmat(cfg.atts_path)

    def __init__(self, set):
        super().__init__()

        loc = self.atts_mat[set].squeeze() - 1

        self.features = torch.from_numpy(self.res_mat['features'][..., loc]).float().T
        self.atts = torch.from_numpy(self.atts_mat['att']).float().T
        self.labels = torch.from_numpy((self.res_mat['labels'] - 1)[loc]).long()

    def __getitem__(self, idx):
        return {'feature': self.features[idx, :],
                'label': self.labels[idx],
                'attribute': self.atts[self.labels[idx][0]]}

    def __len__(self):
        return self.labels.shape[0]
