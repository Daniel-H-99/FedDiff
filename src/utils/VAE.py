import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, label="PathMNIST", image_size=32, channel_num=3, kernel_num=128, z_size=128):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        # self.halfpc = False
        
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear')
        
        # encoder
        self.private_encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )
        # self.encoder = nn.Sequential(
        #     self._conv(channel_num, kernel_num // 4)
        # )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.private_q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.private_q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        # self.private_project = self._linear(z_size, 768, relu=False)

        self.private_project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        raw_size = x.shape[-2:]
        x = self.upsample(x)
        # x = x.half()

        # encoded = self.encoder(x)
        encoded = self.private_encoder(x)
        # print(f'x type: {x.dtype}')
        # while True:
        #     continue
        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        # z_projected = self.private_project(z)
        z_projected = self.private_project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = 2 * self.decoder(z_projected) - 1
        x_reconstructed = F.interpolate(x_reconstructed, size=raw_size)
        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return x_reconstructed, (mean, logvar), z

    def encode_z(self, x):
        # encode x
        raw_size = x.shape[-2:]
        x = self.upsample(x)
        encoded = self.private_encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        # z_projected = self.private_project(z).view(
        #     -1, self.kernel_num,
        #     self.feature_size,
        #     self.feature_size,
        # )

        # reconstruct x from z
        # x_reconstructed = self.decoder(z_projected)
        # x_reconstructed = F.interpolate(x_reconstructed, size=raw_size)
        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return z
    
    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.private_q_mean(unrolled), self.private_q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).to(mean.device) if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        # if self.half:
        #     eps = eps.half()
        return eps.mul(std).add_(mean)

    # def tohalf(self):
    #     self.halfpc = True
    #     for p in self.parameters():
    #         p.data = p.half()


    # def unhalf(self):
    #     self.halfpc = False
    #     for p in self.parameters():
    #         p.data = p.float()
            
    # def reconstruction_loss(self, x_reconstructed, x):
    #     return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    # def kl_divergence_loss(self, mean, logvar):
    #     return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    # # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size, glob=False):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        gen = self.private_project(z, glob=glob)
        
        return {
            "z": z,
            "gen": gen
        }

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)