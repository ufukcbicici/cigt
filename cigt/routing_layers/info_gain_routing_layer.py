import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoGainRoutingLayer(nn.Module):
    def __init__(self,
                 feature_dim,
                 avg_pool_stride,
                 path_count,
                 class_count,
                 input_feature_map_count,
                 input_feature_map_size,
                 device,
                 from_logits=True):
        super().__init__()
        self.featureDim = feature_dim
        self.avgPoolStride = avg_pool_stride
        self.pathCount = path_count
        self.classCount = class_count
        self.device = device
        self.fromLogits = True
        self.inputFeatureMapCount = input_feature_map_count
        self.inputFeatureMapSize = input_feature_map_size
        self.linearLayerInputDim = \
            int(
                self.inputFeatureMapCount * \
                (self.inputFeatureMapSize / self.avgPoolStride) *
                (self.inputFeatureMapSize / self.avgPoolStride))
        self.identityLayer = nn.Identity()
        #  Change the GAP Layer with average pooling with size
        self.avgPool = nn.AvgPool2d(self.avgPoolStride, stride=self.avgPoolStride)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.linearLayerInputDim, self.featureDim)
        self.fc2 = nn.Linear(self.featureDim, self.pathCount)

        self.igBatchNorm = nn.BatchNorm2d(self.featureDim)

    def calculate_entropy(self, prob_distribution, eps=1e-30):
        log_prob = torch.log(prob_distribution + eps)
        # is_inf = tf.is_inf(log_prob)
        # zero_tensor = tf.zeros_like(log_prob)
        # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
        prob_log_prob = prob_distribution * log_prob
        entropy = -1.0 * torch.sum(prob_log_prob)
        return entropy, log_prob

    def forward(self, curr_outputs, routing_matrices,
                labels, temperature, balance_coefficient):
        # Calculate the weighted sum of block outputs
        weighted_outputs = []
        for path, output in curr_outputs.items():
            path_probabilities = []
            for l_, i_ in enumerate(path):
                path_probabilities.append(routing_matrices[l_][:, i_])
            path_probabilities = torch.stack(path_probabilities, dim=1)
            probs = torch.prod(path_probabilities, dim=1)
            probs_exp = self.identityLayer(probs)
            for _ in range(len(output.shape) - len(probs_exp.shape)):
                probs_exp = torch.unsqueeze(probs_exp, -1)
            weighted_output = probs_exp * output
            weighted_outputs.append(weighted_output)
        weighted_outputs = torch.stack(weighted_outputs, dim=1)
        block_output_final = torch.sum(weighted_outputs, dim=1)

        # Feed it into information gain calculation
        h_out = self.avgPool(block_output_final)
        h_out = self.flatten(h_out)
        h_out = F.relu(self.fc1(h_out))

        # Calculate the information gain
        activations = self.fc2(h_out)
        weight_vector = torch.ones(size=(block_output_final.shape[0], ), dtype=torch.float32, device=self.device)
        # probability_vector = tf.cast(weight_vector / tf.reduce_sum(weight_vector), dtype=activations.dtype)
        sample_count = torch.sum(weight_vector)
        probability_vector = torch.div(weight_vector, sample_count)
        batch_size = activations.shape[0]
        node_degree = activations.shape[1]
        joint_distribution = torch.ones(size=(batch_size, self.classCount, node_degree), dtype=activations.dtype,
                                        device=self.device)

        # Calculate p(x)
        joint_distribution = joint_distribution * torch.unsqueeze(torch.unsqueeze(probability_vector, dim=-1), dim=-1)

        # Calculate p(c|x) * p(x) = p(x,c)
        p_c_given_x = torch.nn.functional.one_hot(labels, self.classCount)
        joint_distribution = joint_distribution * torch.unsqueeze(p_c_given_x, dim=2)

        # Calculate p(n|x,c) * p(x,c) = p(x,c,n). Note that p(n|x,c) = p(n|x) since we assume conditional independence
        if self.fromLogits:
            activations_with_temperature = activations / temperature
            p_n_given_x = torch.softmax(activations_with_temperature, dim=1)
        else:
            p_n_given_x = self.identityLayer(activations)
        p_xcn = joint_distribution * torch.unsqueeze(p_n_given_x, dim=1)

        # Calculate p(c,n)
        marginal_p_cn = torch.sum(p_xcn, dim=0)
        # Calculate p(n)
        marginal_p_n = torch.sum(marginal_p_cn, dim=0)
        # Calculate p(c)
        marginal_p_c = torch.sum(marginal_p_cn, dim=1)
        # Calculate entropies
        entropy_p_cn, log_prob_p_cn = self.calculate_entropy(prob_distribution=marginal_p_cn)
        entropy_p_n, log_prob_p_n = self.calculate_entropy(prob_distribution=marginal_p_n)
        entropy_p_c, log_prob_p_c = self.calculate_entropy(prob_distribution=marginal_p_c)
        # Calculate the information gain
        information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
        information_gain = -1.0 * information_gain
        return information_gain, p_n_given_x
