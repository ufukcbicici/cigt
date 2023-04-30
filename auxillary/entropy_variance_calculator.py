import numpy as np
import torch
import torch.nn as nn


class EntropyVarianceCalculator(nn.Module):
    def __init__(self, routing_activations):
        super().__init__()
        self.routingActivations = torch.from_numpy(routing_activations)
        self.temperature = nn.Parameter(torch.ones(size=(), dtype=torch.float32))

    def forward(self):
        eps = 1e-30
        routing_arrs_for_block_tempered = self.routingActivations / self.temperature
        routing_probs = torch.softmax(routing_arrs_for_block_tempered, dim=1)
        log_prob = torch.log(routing_probs + eps)
        prob_log_prob = routing_probs * log_prob
        entropies = -1.0 * torch.sum(prob_log_prob, dim=1)
        loss = -1.0 * torch.var(entropies)
