import torch
import numpy as np


def calculate_correct_cls(cls_pred, cls_true):
    _, cls_pred = torch.max(cls_pred.data, 1)
    return (cls_pred == cls_true).sum().item()


def calculate_label_acc(cls_pred, cls_true, label_acc):
    _, cls_pred = torch.max(cls_pred.data, 1)
    for c_p, c_t in zip(cls_pred, cls_true):
        c_p = int(c_p)
        c_t = int(c_t)

        if label_acc.get(c_t) is None:
            label_acc[c_t] = [0, 0]

        label_acc[c_t][1] += 1
        if c_p == c_t:
            label_acc[c_t][0] += 1

    return label_acc


class EpochCallback:
    monitor_value = np.inf

    def __init__(self, model_name, total_epoch_num, model, optimizer, monitor=None):
        if isinstance(model_name, str):
            model_name = [model_name]
            model = [model]
            optimizer = [optimizer]

        self.model_name = model_name
        self.total_epoch_num = total_epoch_num
        self.monitor = monitor
        self.model = model
        self.optimizer = optimizer

    def __save_model(self):
        for m_name, m, opt in zip(self.model_name, self.model, self.optimizer):
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': opt.state_dict()},
                       m_name)

            print(f'Model saved to {m_name}')

    def epoch_end(self, epoch_num, hash):
        epoch_end_str = f'Epoch {epoch_num}/{self.total_epoch_num} - '
        for name, value in hash.items():
            epoch_end_str += f'{name}: {round(value, 3)} '

        print(epoch_end_str)

        if self.monitor is None:
            self.__save_model()

        elif hash[self.monitor] < self.monitor_value:
            print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')

            self.monitor_value = hash[self.monitor]
            self.__save_model()
        else:
            print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')
