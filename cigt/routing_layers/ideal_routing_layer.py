import torch
import torch.nn as nn


class IdealRoutingLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.idealRoutes = kwargs["ideal_routes"]
        self.classCount = kwargs["class_count"]
        self.pathCount = kwargs["path_count"]
        self.routeMatrix = torch.zeros(size=(self.classCount, self.pathCount), dtype=torch.float32,
                                       device="cuda")
        for p_id, route_tpl in enumerate(self.idealRoutes):
            for lbl in route_tpl:
                self.routeMatrix[lbl, p_id] = 1.0

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        # Layer input is the labels
        batch_size = labels.size(0)
        label_one_hot_matrix = torch.nn.functional.one_hot(labels, self.classCount).type(self.routeMatrix.dtype)
        p_n_given_x = torch.matmul(label_one_hot_matrix, self.routeMatrix)

        # p_n_given_x2 = torch.zeros_like(p_n_given_x)
        # for lid, lbl in enumerate(labels):
        #     for route_id, ss in enumerate(self.idealRoutes):
        #         if lbl.numpy().item() in ss:
        #             p_n_given_x2[lid, route_id] = 1.0
        # assert np.array_equal(p_n_given_x.numpy(), p_n_given_x2.numpy())

        return p_n_given_x
