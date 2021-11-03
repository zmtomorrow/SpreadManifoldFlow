import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PermuteRandom(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

#         np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)
#         np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x):
        return x[:, self.perm],0

    def backward(self,x):
        return x[:, self.perm_inv],0
    

class Flatten(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.size=input_shape
        self.total_size = input_shape[0]*input_shape[1]*input_shape[2]

    def forward(self, x):
        return x.view(-1,self.total_size),0
            
    def backward(self,x):
        return x.view(-1,*self.size),0
    
    
class DownSampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 2
        self.block_size_sq = self.block_size**2

    def backward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return input.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl), 0

    def forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w), 0

    
#     class IRevNetDownsampling(nn.Module):
#     '''The invertible spatial downsampling used in i-RevNet, adapted from
#     https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py'''

#     def __init__(self, dims_in):
#         super().__init__()
#         self.block_size = 2
#         self.block_size_sq = self.block_size**2

#     def forward(self, x):
#         input=x
#         output = input.permute(0, 2, 3, 1)
#         (batch_size, s_height, s_width, s_depth) = output.size()
#         d_depth = s_depth * self.block_size_sq
#         d_height = int(s_height / self.block_size)
#         t_1 = output.split(self.block_size, 2)
#         stack = [t_t.contiguous().view(batch_size, d_height, d_depth)
#                      for t_t in t_1]
#         output = torch.stack(stack, 1)
#         output = output.permute(0, 2, 1, 3)
#         output = output.permute(0, 3, 1, 2)
#         return [output.contiguous()],0

#      def backward(self, x):
#         input = x
#         output = input.permute(0, 2, 3, 1)
#         (batch_size, d_height, d_width, d_depth) = output.size()
#         s_depth = int(d_depth / self.block_size_sq)
#         s_width = int(d_width * self.block_size)
#         s_height = int(d_height * self.block_size)
#         t_1 = output.contiguous().view(batch_size, d_height, d_width,
#                                            self.block_size_sq, s_depth)
#         spl = t_1.split(self.block_size, 3)
#         stack = [t_t.contiguous().view(batch_size, d_height, s_width,
#                                            s_depth) for t_t in spl]
#         output = torch.stack(stack, 0).transpose(0, 1)
#         output = output.permute(0, 2, 1, 3, 4).contiguous()
#         output = output.view(batch_size, s_height, s_width, s_depth)
#         output = output.permute(0, 3, 1, 2)
#         return [output.contiguous()],0

#     def output_dims(self, input_dims):
#         assert len(input_dims) == 1, "Can only use 1 input"
#         c, w, h = input_dims[0]
#         c2, w2, h2 = c*4, w//2, h//2
#         assert c*h*w == c2*h2*w2, "Uneven input dimensions"
#         return [(c2, w2, h2)]
