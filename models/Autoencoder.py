from importlib.metadata import requires
from turtle import forward
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        return to_4d(x, h, w)

class Encoder(nn.Module):
    def __init__(self, input_channels, h_channels, bias=True) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.norm_layer = WithBias_LayerNorm(input_channels)

        self.qk = nn.Conv2d(input_channels, input_channels*2, kernel_size=1, bias=1)
        self.v = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=1)
        self.qk_dwconv = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, groups=input_channels*2, padding=1,bias=bias)
        self.v_dwconv = nn.Conv2d(input_channels, input_channels, kernel_size=3, groups=input_channels, padding=1,bias=bias)

        self.w = DimSelect(input_channels, h_channels)
        self.project_out = nn.Conv2d(h_channels, h_channels, kernel_size=1, bias=bias)

        
    def forward(self, x):
        b,c,h,w = x.shape
        x1 = self.norm_layer(x)
        qk = self.qk(x1)
        v = self.v_dwconv(self.v(x))
        q,k = self.qk_dwconv(qk).chunk(2, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.w(attn)
        attn = attn.transpose(-2, -1)
        attn = nn.functional.softmax(attn, dim=-1)
        # print(torch.sum(attn,dim=-1))
        out = attn @ v

        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)

        out = self.project_out(out)
        return out
    
    def freeze_all(self):
        for params in self.parameters():
            params.requires_grad = False

class Decoder(nn.Module):
    def __init__(self, h_channels, inputs_channels, bias=True) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(h_channels, h_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(h_channels, inputs_channels, kernel_size=3, padding=1, bias=bias)
        )
    
    def forward(self, h_features):
        return self.projector(h_features)

    def freeze_all(self):
        for params in self.parameters():
            params.requires_grad = False
    

class DimSelect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = torch.nn.Parameter(torch.ones((in_channels, out_channels), requires_grad=True) / (in_channels))
        
        
    def forward(self, x):
        return x @ self.W


class Adjustor(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()
        self.projector = nn.ModuleList()
        for _ in range(3):
            projector = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
                nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=bias)
            )
            self.projector.append(projector)
    
    def forward(self, x, f):
        out = []
        for index in range(3):
            out.append(x + self.projector[index](f[:, 64*index: 64*index+64, :, :]))
        return out

    def freeze_all(self):
        for params in self.parameters():
            params.requires_grad = False