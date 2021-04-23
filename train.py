import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Additional Scripts
from train_clswgan import TrainCLSWGAN
from utils.utils import calculate_correct_cls, calculate_label_acc, EpochCallback
from utils.AWADataset import AWADataset
from config import cfg


class TrainTestPipe:
    def __init__(self, device):
        self.device = device

        self.train_seen_loader = self.__load_dataset('trainval_loc')
        self.val_seen_loader = self.__load_dataset('test_seen_loc')
        self.unseen_loader = self.__load_dataset('test_unseen_loc')

        self.clswgan = TrainCLSWGAN(self.device)

    def __load_dataset(self, zsl_set):
        animal_set = AWADataset(zsl_set)
        return DataLoader(animal_set, batch_size=cfg.batch_size, shuffle=False)

    def __loop_train(self, loader, step_func, t, val=False, set=None):
        total_loss = None
        total_correct = None

        for step, data in enumerate(loader):
            feat, atts, cls_true = data['feature'], data['attribute'], data['label']
            feat = torch.autograd.Variable(feat, requires_grad=True).to(self.device)
            atts = torch.autograd.Variable(atts, requires_grad=True).to(self.device)
            cls_true = cls_true.to(self.device).squeeze_()

            loss, cls_pred = step_func(feat=feat, atts=atts, cls_true=cls_true, val=val, set=set, step=step)

            if cls_pred is not None:
                n_correct = calculate_correct_cls(cls_pred, cls_true)
                if total_correct is None:
                    total_correct = 0

                total_correct += n_correct

            if isinstance(loss, list):
                if total_loss is None:
                    total_loss = [0] * len(loss)

                total_loss = np.add(total_loss, loss).tolist()
            else:
                if total_loss is None:
                    total_loss = 0

                total_loss += loss

            t.update()

        return total_loss, total_correct

    def load_model(self, paths, model_types):
        if isinstance(paths, str) or paths is None:
            paths = [paths]
            model_types = [model_types]

        for model_type, path in zip(model_types, paths):
            if path is None:
                print(f'{model_type}_path cannot be loaded, it is not defined!')
                break

            elif not os.path.exists(path):
                print(f'Path ({path}) does not exist!')
                break

            if model_type == 'g_cls':
                model = self.clswgan.G_cls
                optimizer = self.clswgan.G_cls_optimizer
            elif model_type == 'wgan_G':
                model = self.clswgan.G
                optimizer = self.clswgan.G_optimizer
            elif model_type == 'wgan_D':
                model = self.clswgan.D
                optimizer = self.clswgan.D_optimizer
            elif model_type == 'projection':
                model = self.clswgan.projection
                optimizer = self.clswgan.projection_optimizer
            else:
                print(f'Unexpected model_type! ({model_type})')
                break

            ckpt = torch.load(path)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f'{model_type} model has been loaded!')

    def train_g_cls(self):
        callback = EpochCallback(cfg.g_cls.model_name, cfg.g_cls.epoch,
                                 self.clswgan.G_cls, self.clswgan.G_cls_optimizer, 'val_loss')

        for epoch in range(cfg.g_cls.epoch):
            with tqdm(total=len(self.train_seen_loader) + len(self.val_seen_loader)) as t:
                train_loss, train_correct = self.__loop_train(self.train_seen_loader, self.clswgan.step_g_cls, t)

                val_loss, val_correct = self.__loop_train(self.val_seen_loader, self.clswgan.step_g_cls, t, val=True)

            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_seen_loader),
                                'acc': train_correct / (len(self.train_seen_loader) * cfg.batch_size),
                                'val_loss': val_loss / len(self.val_seen_loader),
                                'val_acc': val_correct / (len(self.val_seen_loader) * cfg.batch_size)})

    def train_wgan(self):
        callback = EpochCallback([cfg.wgan.G_model_name, cfg.wgan.D_model_name], cfg.wgan.epoch,
                                 [self.clswgan.G, self.clswgan.D],
                                 [self.clswgan.G_optimizer, self.clswgan.D_optimizer])

        for epoch in range(cfg.wgan.epoch):
            with tqdm(total=len(self.train_seen_loader)) as t:
                loss, _ = self.__loop_train(self.train_seen_loader, self.clswgan.step_wgan, t)

            callback.epoch_end(epoch + 1,
                               {'d_loss': loss[0] / len(self.train_seen_loader),
                                'g_loss': loss[1] / (len(self.train_seen_loader) / cfg.wgan.n_step)})

    def train_projection(self):
        callback = EpochCallback(cfg.projection.model_name, cfg.projection.epoch,
                                 self.clswgan.projection, self.clswgan.projection_optimizer,
                                 'unseen_loss')

        for epoch in range(cfg.projection.epoch):
            with tqdm(total=len(self.train_seen_loader) + len(self.unseen_loader)) as t:
                seen_train_loss, seen_train_correct = self.__loop_train(self.train_seen_loader,
                                                                        self.clswgan.step_projection, t, set='seen')

                unseen_loss, unseen_correct = self.__loop_train(self.unseen_loader,
                                                                self.clswgan.step_projection, t, set='unseen')

            callback.epoch_end(epoch + 1,
                               {'seen_train_loss': seen_train_loss / len(self.train_seen_loader),
                                'seen_train_acc': seen_train_correct / (len(self.train_seen_loader) * cfg.batch_size),
                                'unseen_loss': unseen_loss / len(self.unseen_loader),
                                'unseen_acc': unseen_correct / (len(self.unseen_loader) * cfg.batch_size)})

    def __loop_test(self, loader, t):
        label_acc = {}
        for data in loader:
            feat, atts, cls_true = data['feature'], data['attribute'], data['label']
            feat = feat.to(self.device)
            cls_true = cls_true.to(self.device).squeeze_()

            cls_pred = self.clswgan.inference(feat=feat)
            label_acc = calculate_label_acc(cls_pred, cls_true, label_acc)
            t.update()

        ay = sum(n_correct / n for n_correct, n in label_acc.values()) / len(label_acc)
        return ay

    def test(self):
        with tqdm(total=len(self.val_seen_loader) + len(self.unseen_loader)) as t:
            ays = self.__loop_test(self.val_seen_loader, t)
            ayu = self.__loop_test(self.unseen_loader, t)

            H = (2 * ayu * ays) / (ayu + ays)

        print(f'Seen Set Accuracy: {ays}\nUnseen Set Accuracy: {ayu}\nH: {H}')
