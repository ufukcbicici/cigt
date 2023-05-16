import torch
import numpy as np

batch_size = 1024
path_count = 4


def divide_tensor_wrt_routing_matrix(tens, routing_matrix):
    tens_num_of_non_batch_dims = np.prod(tens.shape[1:]).item()
    masked_tensors = []
    p_count = routing_matrix.shape[1]
    for p_id in range(p_count):
        s_tensor = routing_matrix[:, p_id].to(torch.bool)
        for _ in range(len(tens.shape) - 1):
            s_tensor = s_tensor.unsqueeze(-1)
        tens_part = torch.masked_select(tens, s_tensor)
        tens_part = torch.reshape(input=tens_part,
                                  shape=(int(tens_part.shape[0] // tens_num_of_non_batch_dims), *tens.shape[1:]))
        masked_tensors.append(tens_part)
    return masked_tensors


X = torch.rand(size=(batch_size, 32, 32, 64), dtype=torch.float32)
Y = torch.rand(size=(batch_size, 128), dtype=torch.float32)
Z = torch.rand(size=(batch_size, ), dtype=torch.float32)

activations_matrix = torch.rand(size=(batch_size, path_count), dtype=torch.float32)
rm = torch.zeros_like(activations_matrix)
b = activations_matrix.ge(0.5)
rm[b] = 1.0

masked_X = divide_tensor_wrt_routing_matrix(tens=X, routing_matrix=rm)
masked_Y = divide_tensor_wrt_routing_matrix(tens=Y, routing_matrix=rm)
masked_Z = divide_tensor_wrt_routing_matrix(tens=Z, routing_matrix=rm)

for path_id in range(path_count):
    selection_tensor = rm[:, path_id].to(torch.bool)
    selected_idx = 0
    for idx in range(selection_tensor.shape[0]):
        if selection_tensor[idx]:
            A = X[idx].numpy()
            B = masked_X[path_id][selected_idx].numpy()
            assert np.array_equal(A, B)

            A = Y[idx].numpy()
            B = masked_Y[path_id][selected_idx].numpy()
            assert np.array_equal(A, B)

            A = Z[idx].numpy()
            B = masked_Z[path_id][selected_idx].numpy()
            assert np.array_equal(A, B)
            selected_idx += 1
print("X")
