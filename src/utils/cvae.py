import torch
import torch.nn as nn
import torch.nn.functional as F



class CEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=[32, 64, 128, 256, 256], num_layer=5, num_classes=9, act_fn=nn.ReLU, input_padding=0):
        super(CEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.act_fn = act_fn
        self.input_padding = input_padding
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Upsample(size=(28, 28), mode='bilinear'),
            nn.Conv2d(self.input_dim + self.num_classes, self.hidden_dim[0], 3, stride=2, padding=(1, 1)), 
            self.act_fn(),
             nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1], 3, stride=2, padding=(1, 1)), 
            self.act_fn(), 
            nn.Conv2d(self.hidden_dim[1], self.hidden_dim[2], 3, stride=2, padding=(1, 1)), 
            self.act_fn(),  
            nn.Conv2d(self.hidden_dim[2], self.hidden_dim[3], 3, stride=2, padding=(2, 2)), 
            self.act_fn(),
            nn.Conv2d(self.hidden_dim[3], self.hidden_dim[4], 3, stride=1), 
            self.act_fn(),      
        )

        # self.layers = nn.Sequential(
        #     nn.Conv2d(self.input_dim, self.hidden_dim[0], 5, stride=3, padding=(2, 2), bias=False), 
        #     # self.act_fn(),
        #      nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1], 5, stride=3, padding=(2, 2), bias=False), 
        #     # self.act_fn(), 
        #     nn.Conv2d(self.hidden_dim[1], self.hidden_dim[2], 3, stride=2, padding=(1, 1), bias=False), 
        #     # self.act_fn(),            
        # )
    
    def forward(self, x, c):
        # assert x.shape[2] == 32, f'x shape: {x.shape}'
        assert len(c.shape) == 2, f'{c.shape}'
        x = torch.cat([x, c.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
        latent = self.layers(x).flatten(1)
        # print(f'latent shape: {latent.shape}')
        # while True:
        #     continue
        assert len(latent.shape) == 2, f"latent shape: {latent.shape}"
        return latent
    
class CDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=[256, 128, 64, 32, 3], num_layer=5, num_classes=9, act_fn=nn.ReLU):
        super(CDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.act_fn = act_fn 
        self.num_classes = num_classes
        self.private_layers = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim + self.num_classes, self.hidden_dim[0], 3),
            self.act_fn()
        )
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim[0], self.hidden_dim[1], 3, stride=2, padding=2, output_padding=1),
            self.act_fn(),
            nn.ConvTranspose2d(self.hidden_dim[1], self.hidden_dim[2], 3, stride=2, padding=1),
            self.act_fn(),
            nn.ConvTranspose2d(self.hidden_dim[2], self.hidden_dim[3], 3, stride=2, padding=1, output_padding=1),
            self.act_fn(),           
            nn.ConvTranspose2d(self.hidden_dim[3], self.hidden_dim[4], 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid(),
            nn.Upsample(size=(28, 28), mode='bilinear')     
        )
    
    def forward(self, x, c, glob=False):
        assert len(c.shape) == 2, f'{c.shape}'
        assert x.shape[1] == self.input_dim, f'x shape: {x.shape}'
        # print(f'z shape: {x.shape}')
        # print(f'zc shape: {c.shape}')
        x = x.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, c.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 1)], dim=1)
        x = self.private_layers(x)
        if glob:
            x = self.act_fn()(torch.randn_like(x))
        out = self.layers(x)
        assert len(out.shape) == 4, f"latent shape: {out.shape}"
        return out
    
class CVAE(nn.Module):
    def __init__(self, input_shape=(3, 28, 28), hidden_dim=[32, 64, 128, 256, 512], act_fn=nn.ReLU, num_classes=9, device='cuda'):
        super(CVAE, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.z_dim = self.hidden_dim[-1]
        self.device = device
        rev_hidden_dim = list(reversed([input_shape[0]] + hidden_dim))
        self.num_classes = num_classes
        self.private_encoder = CEncoder(input_dim=input_shape[0], hidden_dim=hidden_dim, num_layer=len(hidden_dim), act_fn=self.act_fn, num_classes=self.num_classes)
        self.decoder = CDecoder(input_dim=rev_hidden_dim[0], hidden_dim=rev_hidden_dim[1:], num_layer=len(rev_hidden_dim[1:]), act_fn=self.act_fn, num_classes=self.num_classes)
        self.private_proj_muvar = nn.Linear(self.hidden_dim[-1], self.z_dim * 2, bias=False)
        self.num_classes = num_classes
        # self.deproj_z = nn.Sequential(
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        # )
        
    def sample_z(self, B):
        return torch.randn(B, self.z_dim).to(self.device)
    
    def forward(self, x, c):
        # print(f'x shape: {x.shape}')
        # print(f'c shape: {c.shape}')
        if len(c.shape) == 2:
            c = c.squeeze(1)
        assert len(c.shape) == 1, f'{c.shape}'
        c = torch.nn.functional.one_hot(c, self.num_classes)
        latent = self.private_encoder(x, c)
        muvar = self.private_proj_muvar(latent)
        assert len(muvar.shape) == 2, f'muvar shape: {muvar.shape}'
        assert muvar.shape[1] == self.z_dim * 2, f'muvar shape1: {muvar.shape}'
        mu, var = muvar.split([self.z_dim, self.z_dim], dim=-1)
        sig = torch.exp(0.5 * var)
        var = sig ** 2
        z = self.sample_z(len(mu))
        z_tilde = mu + sig * z
        # recon = self.decoder(z_tilde)
        # dec_in = self.deproj_z(z_tilde)
        dec_in = z_tilde
        recon = self.decoder(dec_in, c)
        assert len(recon.shape) == 4, f'recon shape: {recon.shape}'
        assert recon.shape[1] == 3, f'recon shape: {recon.shape[1]}'
        # assert recon.shape[2] == 32, f'recon shape: {recon.shape[2]}'
        # assert recon.shape[3] == 32, f'recon shape: {recon.shape[3]}'
        
        return {
            "input": x,
            "recon": recon,
            "mu": mu,
            "var": var,
            "z_tilde": z_tilde,
        }
        
    def compute_loss(self, out):
        # print(f'recon shapes: {out["recon"].shape}')
        # print(f'input shapes: {out["input"].shape}')
        # loss_recon = ((out["recon"] - out["input"])** 2).sum(dim=[1, 2, 3]).mean(dim=0)
        loss_recon = ((out["recon"] - out["input"])** 2).mean()
        # print(f'var: {out["var"]}')
        # print(f'mu: {out["mu"]}')
        

        loss_vae = - 0.5 * torch.log(out["var"]).sum(dim=1).mean(dim=0) + 0.5 * ((out["mu"] ** 2).sum(dim=1).mean(dim=0) + out["var"].sum(dim=1).mean(dim=0) - out["var"].shape[1])
        # loss_vae = torch.zeros(1).to(loss_recon.device)
        # print(f'loss vae: {loss_vae}') 
        # while True:
        #     continue
        # print(f'loss_recon: {loss_recon}')
        return loss_recon + 1e-4 * loss_vae, {"loss_recon": loss_recon.item(), "loss_vae": loss_vae.item()}
    
    def sample(self, B, c,  glob=False):
        z = self.sample_z(B)
        if len(c.shape) == 2:
            c = c.squeeze(1)
        c = torch.nn.functional.one_hot(c, self.num_classes)
        # dec_in = self.deproj_z(z)
        dec_in = z
        gen = self.decoder(dec_in, c, glob=glob)
        return {
            "z": z,
            "gen": gen
        }
    
    
class CVAEGenerator(nn.Module):
    def __init__(self, input_shape=(3, 28, 28), hidden_dim=[32, 64, 128, 256, 512], act_fn=nn.ReLU, num_classes=9, device='cuda'):
        super(CVAE, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.z_dim = self.hidden_dim[-1]
        self.device = device
        rev_hidden_dim = list(reversed([input_shape[0]] + hidden_dim))
        self.num_classes = num_classes
        self.private_encoder = CEncoder(input_dim=input_shape[0], hidden_dim=hidden_dim, num_layer=len(hidden_dim), act_fn=self.act_fn, num_classes=self.num_classes)
        # self.decoder = CDecoder(input_dim=rev_hidden_dim[0], hidden_dim=rev_hidden_dim[1:], num_layer=len(rev_hidden_dim[1:]), act_fn=self.act_fn, num_classes=self.num_classes)
        self.private_proj_muvar = nn.Linear(self.hidden_dim[-1], self.z_dim * 2, bias=False)
        self.private_dec_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.LeakyReLU()
        )
        self.num_classes = num_classes
        # self.deproj_z = nn.Sequential(
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]),
        #     nn.ReLU(),
        # )
        
    def sample_z(self, B):
        return torch.randn(B, self.z_dim).to(self.device)
    
    def forward(self, x, c):
        # print(f'x shape: {x.shape}')
        # print(f'c shape: {c.shape}')
        if len(c.shape) == 2:
            c = c.squeeze(1)
        assert len(c.shape) == 1, f'{c.shape}'
        c = torch.nn.functional.one_hot(c, self.num_classes)
        latent = self.private_encoder(x, c)
        muvar = self.private_proj_muvar(latent)
        assert len(muvar.shape) == 2, f'muvar shape: {muvar.shape}'
        assert muvar.shape[1] == self.z_dim * 2, f'muvar shape1: {muvar.shape}'
        mu, logvar = muvar.split([self.z_dim, self.z_dim], dim=-1)
        sig = torch.exp(0.5 * logvar)
        var = sig ** 2
        z = self.sample_z(len(mu))
        z_tilde = mu + sig * z
        # recon = self.decoder(z_tilde)
        context = self.private_dec_layer(z_tilde)
        # dec_in = z_tilde
        # recon = self.decoder(dec_in, c)
        # assert len(recon.shape) == 4, f'recon shape: {recon.shape}'
        # assert recon.shape[1] == 3, f'recon shape: {recon.shape[1]}'
        # assert recon.shape[2] == 32, f'recon shape: {recon.shape[2]}'
        # assert recon.shape[3] == 32, f'recon shape: {recon.shape[3]}'
        
        return {
            "input": x,
            "mu": mu,
            "var": var,
            "z_tilde": z_tilde,
            "context": context
        }
        
    def compute_loss(self, out):
        # print(f'recon shapes: {out["recon"].shape}')
        # print(f'input shapes: {out["input"].shape}')
        # loss_recon = ((out["recon"] - out["input"])** 2).sum(dim=[1, 2, 3]).mean(dim=0)
        # loss_recon = ((out["recon"] - out["input"])** 2).mean()
        # print(f'var: {out["var"]}')
        # print(f'mu: {out["mu"]}')
        

        loss_vae = - 0.5 * torch.log(out["var"]).sum(dim=1).mean(dim=0) + 0.5 * ((out["mu"] ** 2).sum(dim=1).mean(dim=0) + out["var"].sum(dim=1).mean(dim=0) - out["var"].shape[1])
        # loss_vae = torch.zeros(1).to(loss_recon.device)
        # print(f'loss vae: {loss_vae}') 
        # while True:
        #     continue
        # print(f'loss_recon: {loss_recon}')
        return loss_vae
    
    def sample(self, B, c,  glob=False):
        z = self.sample_z(B)
        if len(c.shape) == 2:
            c = c.squeeze(1)
        c = torch.nn.functional.one_hot(c, self.num_classes)
        # dec_in = self.deproj_z(z)
        dec_in = z
        gen = self.decoder(dec_in, c, glob=glob)
        return {
            "z": z,
            "gen": gen
        }
    
    
        
    
        