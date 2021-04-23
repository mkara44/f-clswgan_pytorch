import torch
import argparse

# Additional Scripts
from train import TrainTestPipe


def main_pipeline(parser):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'

    ttp = TrainTestPipe(device)
    ttp.load_model(parser.g_cls_path, 'g_cls')
    ttp.load_model([parser.wgan_G_path, parser.wgan_D_path], ['wgan_G', 'wgan_D'])
    ttp.load_model(parser.projection_path, 'projection')

    if parser.train:
        print('G_cls training process has been started!')
        ttp.train_g_cls()

        print('Wgan training process has been started!')
        ttp.train_wgan()

        print('Projection training process has been started!')
        ttp.train_projection()

    print('Test has been started!')
    ttp.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--g_cls_path', type=str, default=None)
    parser.add_argument('--wgan_G_path', type=str, default=None)
    parser.add_argument('--wgan_D_path', type=str, default=None)
    parser.add_argument('--projection_path', type=str, default=None)
    parser = parser.parse_args()

    main_pipeline(parser)
