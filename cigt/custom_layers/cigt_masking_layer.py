import numpy as np
import torch
from torch import nn


class CigtMaskingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, net, routing_matrix):
        channel_dim = net.shape[1]
        path_count = routing_matrix.shape[1]
        repeat_count = channel_dim // path_count
        routing_matrix_repeated = torch.repeat_interleave(routing_matrix, repeat_count, dim=1)
        for _ in range(len(net.shape) - len(routing_matrix_repeated.shape)):
            routing_matrix_repeated = torch.unsqueeze(routing_matrix_repeated, dim=-1)
        masked_net = net * routing_matrix_repeated
        return masked_net


# Testing script
# if __name__ == "__main__":
#     masking_layer = CigtMaskingLayer()
#
#     for path_count in [2, 4]:
#         for tens_shape in [(1024, 32, 16, 16), (1024, 64)]:
#             x_conv = np.random.random(size=tens_shape)
#
#             path_length = x_conv.shape[1] // path_count
#             routing_activations = np.random.random(size=(1024, path_count))
#             routing_mat = np.zeros_like(routing_activations)
#             routing_mat[np.arange(routing_activations.shape[0]), np.argmax(routing_activations, axis=1)] = 1.0
#
#             masked_result = masking_layer(torch.from_numpy(x_conv), torch.from_numpy(routing_mat))
#             masked_result = masked_result.numpy()
#
#             for i in range(routing_mat.shape[0]):
#                 for j in range(routing_mat.shape[1]):
#                     partial_result = masked_result[i, j*path_length:(j + 1)*path_length, ...]
#                     if routing_mat[i, j] == 0.0:
#                         assert np.array_equal(partial_result, np.zeros_like(partial_result))
#                     else:
#                         assert np.array_equal(partial_result, x_conv[i, j*path_length:(j + 1)*path_length, ...])
#         print("X")


