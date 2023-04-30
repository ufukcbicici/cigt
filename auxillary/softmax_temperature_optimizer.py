from collections import deque
import torch
import math
import numpy as np
import torch.nn as nn

from auxillary.entropy_variance_calculator import EntropyVarianceCalculator


class SoftmaxTemperatureOptimizer(object):
    def __init__(self):
        pass

    def run(self, routing_activations):
        temperature = torch.ones(size=(), dtype=torch.float32, requires_grad=True)
        A_ = torch.from_numpy(routing_activations)
        alpha = 1.2
        beta = 0.5
        lr = 0.001
        min_lr = 1e-10
        loss_history = deque(maxlen=10000)
        grad_history = deque(maxlen=10000)
        eps = 1e-30

        for iteration_id in range(1000000):
            routing_arrs_for_block_tempered = A_ / temperature
            routing_probs = torch.softmax(routing_arrs_for_block_tempered, dim=1)
            log_prob = torch.log(routing_probs + eps)
            prob_log_prob = routing_probs * log_prob
            entropies = -1.0 * torch.sum(prob_log_prob, dim=1)
            loss = -1.0 * torch.var(entropies)
            loss.backward()

            with torch.no_grad():
                curr_loss = np.copy(loss.detach().numpy())
                delta = np.copy(temperature.grad.numpy())
                loss_history.append(curr_loss)
                grad_history.append(delta)
                print("curr_loss={0}".format(curr_loss))
                print("curr_temperature={0}".format(temperature))
                if len(grad_history) > 1:
                    sgn_t = grad_history[-1] / abs(grad_history[-1])
                    sgn_t_minus_1 = grad_history[-2] / abs(grad_history[-2])
                    if (sgn_t * sgn_t_minus_1) > 0:
                        lr = min(lr * alpha, 1.0)
                    elif (sgn_t * sgn_t_minus_1) < 0:
                        lr = max(lr * beta, min_lr)
                    else:
                        return temperature.detach().numpy()
                step = lr * (temperature.grad / torch.abs(temperature.grad))
                temperature -= step
                if lr < min_lr or np.allclose(0.0, step.numpy(), atol=1e-7):
                    break

                temperature.grad = None
        return temperature.detach().numpy()
