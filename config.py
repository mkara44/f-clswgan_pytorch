from easydict import EasyDict

cfg = EasyDict()
cfg.beta = 0.01
cfg.lambd = 10.
cfg.latent_dim = 128
cfg.batch_size = 64
cfg.attr_number = 85
cfg.output_size = 224
cfg.seen_class_number = 40
cfg.unseen_class_number = 10
cfg.atts_path = './AWA2/att_splits.mat'
cfg.res_path = './AWA2/res101.mat'

# g_cls settings
cfg.g_cls = EasyDict()
cfg.g_cls.epoch = 30
cfg.g_cls.learning_rate = 1e-4
cfg.g_cls.model_name = 'g_cls_model_1e4.pt'

# wgan settings
cfg.wgan = EasyDict()
cfg.wgan.epoch = 100
cfg.wgan.n_step = 5
cfg.wgan.learning_rate = 1e-4
cfg.wgan.G_model_name = 'wgan_G_model_1e4.pt'
cfg.wgan.D_model_name = 'wgan_D_model_1e4.pt'

# projection settings
cfg.projection = EasyDict()
cfg.projection.epoch = 30
cfg.projection.learning_rate = 1e-4
cfg.projection.model_name = 'projection_model_1e4.pt'
