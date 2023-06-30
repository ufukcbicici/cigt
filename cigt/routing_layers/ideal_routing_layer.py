import torch
import torch.nn as nn


class IdealRoutingLayer(nn.Module):
    def __init__(self, ideal_routes, class_count, path_count, error_ratio, device):
        super().__init__()
        self.idealRoutes = ideal_routes
        self.classCount = class_count
        self.pathCount = path_count
        self.errorRatio = error_ratio
        self.routeMatrix = torch.zeros(size=(self.classCount, self.pathCount), dtype=torch.float32,
                                       device=device)
        for p_id, route_tpl in enumerate(self.idealRoutes):
            for lbl in route_tpl:
                self.routeMatrix[lbl, p_id] = 1.0

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        # Layer input is the labels
        batch_size = labels.size(0)
        label_one_hot_matrix = torch.nn.functional.one_hot(labels, self.classCount).type(self.routeMatrix.dtype)
        p_n_given_x = torch.matmul(label_one_hot_matrix, self.routeMatrix)

        # Make errors equivalent to the size of the error ratio.
        if self.training:
            random_matrix = torch.randint(low=0, high=1000, size=(p_n_given_x.shape[0], p_n_given_x.shape[1]))
            random_matrix[torch.arange(p_n_given_x.shape[0]), torch.argmax(p_n_given_x, dim=1)] = -1
            error_routing_matrix = torch.zeros_like(p_n_given_x)
            error_routing_matrix[torch.arange(p_n_given_x.shape[0]), torch.argmax(random_matrix, dim=1)] = 1

            error_index_count = int(self.errorRatio * p_n_given_x.shape[0])
            error_indices = torch.randint(low=0, high=p_n_given_x.shape[0], size=(error_index_count,))
            selection_vec = torch.ones(size=(p_n_given_x.shape[0], ))
            selection_vec[error_indices] = 0
            noisy_routing_matrix = torch.where(selection_vec.to(torch.bool),  p_n_given_x, error_routing_matrix)
            p_n_given_x = noisy_routing_matrix

        # p_n_given_x2 = torch.zeros_like(p_n_given_x)
        # for lid, lbl in enumerate(labels):
        #     for route_id, ss in enumerate(self.idealRoutes):
        #         if lbl.numpy().item() in ss:
        #             p_n_given_x2[lid, route_id] = 1.0
        # assert np.array_equal(p_n_given_x.numpy(), p_n_given_x2.numpy())

        return p_n_given_x, p_n_given_x
