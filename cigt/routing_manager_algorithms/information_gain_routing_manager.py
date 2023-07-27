import torch
import numpy as np

# Standard Information Gain routing.
# Route to torch.argmax(p_n_given_x_soft, dim=1) only.
# Corresponds to Approximate Training.
# No temperature decay during warmup.


class InformationGainRoutingManager(object):
    def __init__(self):
        super().__init__()

    # OK x2
    @staticmethod
    def get_enforced_routing_matrix(model, layer_id, p_n_given_x_soft):
        if model.enforcedRoutingMatrices is None or len(model.enforcedRoutingMatrices) == 0:
            return None
        else:
            enforced_routing_matrix = model.enforcedRoutingMatrices[layer_id]
            enforced_routing_matrix = enforced_routing_matrix[0:p_n_given_x_soft.shape[0], :]
            return enforced_routing_matrix

    # OK x2
    def get_warmup_routing_matrix(self, model, p_n_given_x_soft):
        if model.warmUpKind == "RandomRouting":
            random_routing_matrix = torch.rand(size=p_n_given_x_soft.shape)
            arg_max_entries = torch.argmax(random_routing_matrix, dim=1)
            random_routing_matrix_hard = torch.zeros_like(p_n_given_x_soft)
            random_routing_matrix_hard[torch.arange(random_routing_matrix_hard.shape[0]), arg_max_entries] = 1.0
            p_n_given_x_hard = random_routing_matrix_hard
        elif model.warmUpKind == "FullRouting":
            # assert isinstance(model, CigtIgGatherScatterImplementation) \
            #        or issubclass(model, CigtIgGatherScatterImplementation)
            p_n_given_x_hard = torch.ones_like(p_n_given_x_soft)
        else:
            raise NotImplementedError()
        return p_n_given_x_hard

    # OK x2
    def get_hard_routing_matrix(self, model, layer_id, p_n_given_x_soft):
        # Enforced routing has always the first priority, if there are enforced routing matrices,
        # return them before anything else.
        enforced_routing_matrix = InformationGainRoutingManager.get_enforced_routing_matrix(
            model=model, layer_id=layer_id, p_n_given_x_soft=p_n_given_x_soft)
        if enforced_routing_matrix is not None:
            return enforced_routing_matrix

        # Calculate the pure information gain based hard routing matrix.
        ig_routing_matrix_hard = torch.zeros_like(p_n_given_x_soft)
        ig_arg_max_entries = torch.argmax(p_n_given_x_soft, dim=1)
        ig_routing_matrix_hard[torch.arange(ig_routing_matrix_hard.shape[0]), ig_arg_max_entries] = 1.0

        # If we are in the evaluation mode, always use the pure information gain based hard routing matrix.
        if not model.training:
            return ig_routing_matrix_hard

        # If we are in the training mode and in the warm up period, use the warm up routing matrix.
        if model.isInWarmUp:
            warmup_routing_matrix = \
                self.get_warmup_routing_matrix(model=model, p_n_given_x_soft=p_n_given_x_soft)
            return warmup_routing_matrix

        # If we are in the training mode and we don´t use random routing regularization, just return
        # the pure information gain based hard routing matrix.
        if model.routingRandomizationRatio <= 0.0:
            return ig_routing_matrix_hard

        # If we are in the training mode and use random routing regularization,
        # create a random routing matrix with given probabilities. If we additionally want a strictly
        # random matrix such that no ig route is included, apply this as well.
        random_routing_matrix = torch.rand(size=p_n_given_x_soft.shape)
        # If we strictly want to send our samples to non-ig paths, set ig entries to -inf.
        if model.enableStrictRandomRouting:
            random_routing_matrix[
                torch.arange(random_routing_matrix.shape[0]), ig_arg_max_entries] = -float('inf')
        random_arg_max_entries = torch.argmax(random_routing_matrix, dim=1)
        random_routing_matrix_hard = torch.zeros_like(p_n_given_x_soft)
        random_routing_matrix_hard[
            torch.arange(random_routing_matrix_hard.shape[0]), random_arg_max_entries] = 1.0
        random_entries = torch.from_numpy(
            np.random.choice(np.arange(ig_routing_matrix_hard.shape[0]),
                             replace=True,
                             size=int(ig_routing_matrix_hard.shape[0] *
                                      model.routingRandomizationRatio)).astype(np.int64)).to(model.device)
        selection_vec = torch.zeros(size=(ig_routing_matrix_hard.shape[0],), dtype=torch.bool,
                                    device=model.device)
        selection_vec[random_entries] = True
        p_n_given_x_hard = torch.where(torch.unsqueeze(selection_vec, dim=1),
                                       random_routing_matrix_hard,
                                       ig_routing_matrix_hard)
        return p_n_given_x_hard

    # OK x2
    def adjust_temperature(self, model):
        # If "enableInformationGainDuringWarmUp" is True, this means we are always optimizing the information
        # gain even during the warm up. Start the temperature decaying from 0.
        if model.enableInformationGainDuringWarmUp:
            model.temperatureController.update(iteration=model.numOfTrainingIterations)
            temperature = model.temperatureController.get_value()
            return temperature
        else:
            # If "enableInformationGainDuringWarmUp" is False, this means during the warm up period
            # we will not optimizing the information gain, hence temperature will stay constant.
            if model.isInWarmUp:
                return 1.0
            # If "enableInformationGainDuringWarmUp" is False and warm up period is over, then
            # we start decaying the temperature by taking the warm up ending as the zero point.
            else:
                decay_t = model.numOfTrainingIterations - model.warmUpEndingIteration
                model.temperatureController.update(iteration=decay_t)
                temperature = model.temperatureController.get_value()
                return temperature

    # OK x2
    def adjust_decision_loss_coeff(self, model):
        # First of all, assert that we are in the training phase. This method is not supposed to be
        # called during evaluation.
        assert model.training
        # If "enableInformationGainDuringWarmUp" is True, this means we are always optimizing the information
        # gain even during the warm up. Return the original decision coefficient.
        if model.enableInformationGainDuringWarmUp:
            return model.decisionLossCoeff
        else:
            # If "enableInformationGainDuringWarmUp" is False, this means during the warm up period
            # we will not be optimizing the information gain, hence DISABLE THE DECISION LOSS.
            if model.isInWarmUp:
                return 0.0
            # If "enableInformationGainDuringWarmUp" is False and warm up period is over, then
            # we will be optimizing the decision loss in a regular fashion, return the original decision loss
            # coefficient.
            else:
                return model.decisionLossCoeff
