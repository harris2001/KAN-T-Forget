# from torch import nn
# import sys
# import torch.nn.functional as F

# # sys.path.append('../kan_convolutional')

# from KANLinear import KANLinear

# class NormalConvsKAN(nn.Module):
#     def __init__(self):
#         super(NormalConvsKAN, self).__init__()
#         # Convolutional layer, assuming an input with 1 channel (grayscale image)
#         # and producing 16 output channels, with a kernel size of 3x3
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

#         # Max pooling layer
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
        
#         # Flatten layer
#         self.flatten = nn.Flatten()

#         # KAN layer
#         # self.kan1 = KANLinear(
#         #     245,
#         #     10,
#         #     grid_size=10,
#         #     spline_order=50,
#         #     scale_noise=0.01,
#         #     scale_base=1,
#         #     scale_spline=1,
#         #     base_activation=nn.SiLU,
#         #     grid_eps=0.02,
#         #     grid_range=[0,1])
#         self.kan1 = nn.Linear(245, 10)


#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.flatten(x)
#         x = self.kan1(x)
#         x = F.log_softmax(x, dim=1)

#         return x
import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, embed_dim, nb_total_classes, nb_base_classes, increment, nb_tasks, bias=True, complete=True, cosine=False, norm=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_base_classes
        self.cosine = cosine

        if self.cosine not in (False, None, ''):
            self.scale = nn.Parameter(torch.tensor(1.))
        else:
            self.scale = 1
        self.head = nn.Linear(embed_dim, nb_base_classes, bias=not cosine)
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identitty()
        self.increment = increment

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.norm(x)

        if self.cosine not in (False, None, ''):
            w = self.head.weight  # (c, d)

            if self.cosine == 'pcc':
                x = x - x.mean(dim=1, keepdims=True)
                w = w - w.mean(dim=1, keepdims=True)
            x = F.normalize(x, p=2, dim=1)  # (bs, d)
            w = F.normalize(w, p=2, dim=1)  # (c, d)
            return self.scale * torch.mm(x, w.T)

        return self.head(x)

    def init_prev_head(self, head):
        w, b = head.weight.data, head.bias.data
        self.head.weight.data[:w.shape[0], :w.shape[1]] = w
        self.head.bias.data[:b.shape[0]] = b

    def init_prev_norm(self, norm):
        w, b = norm.weight.data, norm.bias.data
        self.norm.weight.data[:w.shape[0]] = w
        self.norm.bias.data[:b.shape[0]] = b

    @torch.no_grad()
    def weight_align(self, nb_new_classes):
        w = self.head.weight.data
        norms = torch.norm(w, dim=1)

        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        w[-nb_new_classes:] = gamma * w[-nb_new_classes:]

    def add_classes(self):
        self.add_new_outputs(self.increment)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=not self.cosine)
        head.weight.data[:-n] = self.head.weight.data
        if not self.cosine:
            head.bias.data[:-n] = self.head.bias.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n