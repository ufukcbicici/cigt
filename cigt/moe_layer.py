import torch
import torch.nn as nn


class MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.identityLayer = nn.Identity()

    def forward(self, curr_outputs, routing_matrices):
        # Calculate the weighted sum of block outputs
        weighted_outputs = []
        for path, logits in curr_outputs.items():
            path_probabilities = []
            for l_, i_ in enumerate(path):
                path_probabilities.append(routing_matrices[l_][:, i_])
            path_probabilities = torch.stack(path_probabilities, dim=1)
            probs = torch.prod(path_probabilities, dim=1)
            probs_exp = self.identityLayer(probs)
            for _ in range(len(logits.shape) - len(probs_exp.shape)):
                probs_exp = torch.unsqueeze(probs_exp, -1)
            # Calculate softmax with exponent normalization
            max_logits = torch.unsqueeze(torch.max(logits, dim=1)[0], dim=1)
            logits_reduced = logits - max_logits
            class_probs = torch.softmax(logits_reduced, dim=1)
            # print(class_probs.device)
            # print(probs_exp.device)
            weighted_probs = probs_exp * class_probs
            weighted_outputs.append(weighted_probs)
            # weighted_logits = probs_exp * logits
            # weighted_outputs.append(weighted_logits)
        weighted_outputs = torch.stack(weighted_outputs, dim=1)
        moe_probs = torch.sum(weighted_outputs, dim=1)
        return moe_probs

        # log_moe_probs = torch.log(moe_probs)
        # nll_loss = self.nllLoss(log_moe_probs, labels)
        # return nll_loss
