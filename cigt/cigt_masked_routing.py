from cigt.cigt_ig_refactored import CigtIgHardRoutingX


class CigtMaskedRouting(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)
