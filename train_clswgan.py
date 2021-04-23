import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import uniform, normal

# Additional Scripts
from utils.nn import Generator, Discriminator, MLP
from config import cfg


class TrainCLSWGAN:
    atts_dim = 0
    batch_size = 64
    output_size = 224
    eps = uniform.Uniform(0, 1)
    Z_sampler = normal.Normal(0, 1)
    beta = cfg.beta
    lambd = cfg.lambd

    def __init__(self, device):
        self.device = device

        # self.G_cls = MLP(cfg.seen_class_number).to(self.device)
        self.G_cls = MLP(cfg.seen_class_number + cfg.unseen_class_number).to(self.device)
        self.G = Generator(cfg.attr_number + cfg.latent_dim).to(self.device)
        self.D = Discriminator(cfg.attr_number).to(self.device)
        self.projection = MLP(cfg.seen_class_number + cfg.unseen_class_number).to(self.device)

        self.projection_criterion = nn.NLLLoss()
        self.G_cls_criterion = nn.NLLLoss()

        self.G_cls_optimizer = optim.Adam(self.G_cls.parameters(), lr=cfg.g_cls.learning_rate)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=cfg.wgan.learning_rate)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=cfg.wgan.learning_rate)
        self.projection_optimizer = optim.Adam(self.projection.parameters(), lr=cfg.projection.learning_rate)

    def get_noise(self, batch_size):
        return torch.autograd.Variable(self.Z_sampler.sample(torch.Size([batch_size, cfg.latent_dim])).to(self.device))

    def get_gradient_penalty(self, d_real, d_fake, batch_size, atts):
        eps = self.eps.sample(torch.Size([batch_size, 1])).to(self.device)
        X_penalty = eps * d_real + (1 - eps) * d_fake

        X_penalty = autograd.Variable(X_penalty, requires_grad=True).to(self.device)
        d_pred = self.D(X_penalty, atts)
        grad_outputs = torch.ones(d_pred.size()).to(self.device)
        gradients = autograd.grad(
            outputs=d_pred, inputs=X_penalty,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambd
        return grad_penalty

    def step_g_cls(self, **params):
        if params['val']:
            self.G_cls.eval()
            with torch.no_grad():
                cls_pred = self.G_cls(params['feat'])
                loss = self.G_cls_criterion(F.log_softmax(cls_pred, dim=1), params['cls_true'])

                return loss.item(), cls_pred

        self.G_cls.train()
        self.G_cls_optimizer.zero_grad()
        cls_pred = self.G_cls(params['feat'])
        loss = self.G_cls_criterion(F.log_softmax(cls_pred, dim=1), params['cls_true'])
        loss.backward()
        self.G_cls_optimizer.step()

        return loss.item(), cls_pred

    def step_wgan(self, **params):
        loss_g = None
        self.G_cls.eval()
        for p in self.D.parameters():
            p.requires_grad = True

        batch_size = params['atts'].shape[0]
        self.D_optimizer.zero_grad()
        d_real = self.D(params['feat'], params['atts'])
        d_real = torch.mean(d_real)
        d_real.backward(torch.tensor(-1.))

        Z = self.get_noise(batch_size)
        fake_feat = self.G(Z, params['atts'])

        d_fake = self.D(fake_feat, params['atts'])
        d_fake = torch.mean(d_fake)
        d_fake.backward(torch.tensor(1.))

        gradient_penalty = self.get_gradient_penalty(params['feat'], fake_feat, batch_size, params['atts'])
        gradient_penalty.backward()

        loss_d = d_fake - d_real + gradient_penalty
        self.D_optimizer.step()

        if params['step'] % cfg.wgan.n_step == 0:
            for p in self.D.parameters():
                p.requires_grad = False
            self.G_optimizer.zero_grad()
            Z = self.get_noise(batch_size)
            fake_feat = self.G(Z, params['atts'])

            d_fake = self.D(fake_feat, params['atts'])
            d_fake = -1 * torch.mean(d_fake)

            g_cls_pred = self.G_cls(fake_feat)
            loss_cls = self.G_cls_criterion(F.log_softmax(g_cls_pred, dim=1), params['cls_true'])
            loss_g = d_fake + self.beta * loss_cls

            loss_g.backward()
            self.G_optimizer.step()

        return [loss_d.item(), loss_g.item() if loss_g is not None else 0], None

    def step_projection(self, **params):
        with torch.no_grad():
            if params['set'] == 'seen':
                feat = params['feat']

                if params['val']:
                    self.projection.eval()
                    cls_pred = self.projection(params['feat'])
                    loss = self.projection_criterion(cls_pred, params['cls_true'])

                    return loss.item(), cls_pred

            elif params['set'] == 'unseen':
                batch_size = params['atts'].shape[0]
                Z = self.Z_sampler.sample(torch.Size([batch_size, cfg.latent_dim])).to(self.device)
                feat = self.G(Z, params['atts'])

        self.projection.train()
        self.projection_optimizer.zero_grad()
        cls_pred = self.projection(feat)
        loss = self.projection_criterion(F.log_softmax(cls_pred, dim=1), params['cls_true'])
        loss.backward()
        self.projection_optimizer.step()

        return loss.item(), cls_pred

    def inference(self, **params):
        self.projection.eval()
        with torch.no_grad():
            cls_pred = self.projection(params['feat'])

        return cls_pred
