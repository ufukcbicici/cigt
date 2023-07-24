from collections import OrderedDict

import torch.nn as nn

from cigt.cigt_model import BasicBlock, Sequential_ext


class LenetConfigInterpreter:

    def __init__(self):
        pass

    @staticmethod
    def interpret_config_list(configs):
        return configs.layer_config_list

    @staticmethod
    def get_explanation(model, explanation, kv_rows):
        for block_id, block_dict in enumerate(model.blockParametersList):
            block_elements = block_dict["layer_structure"]
            assert isinstance(block_elements, list)
            for layer_id, layer_description_dict in enumerate(block_elements):
                layer_description_kvs = [(k, v) for k, v in layer_description_dict.items() if k != "layer_type"]
                layer_description_kvs_sorted = [("layer_type", layer_description_dict["layer_type"])]
                layer_description_kvs_sorted.extend(sorted(layer_description_kvs, key=lambda tpl: tpl[0]))
                for k, v in layer_description_kvs_sorted:
                    explanation = model.add_explanation(
                        name_of_param="Block {0} Layer {1} {2}".format(block_id, layer_id, k),
                        value=v,
                        explanation=explanation, kv_rows=kv_rows)
        return kv_rows, explanation

    @staticmethod
    def get_loss_layer(model):
        model.lossLayers = nn.ModuleList()
        if model.lossCalculationKind == "SingleLogitSingleLoss":
            end_module = nn.Sequential(OrderedDict([
                ('logits', nn.LazyLinear(out_features=model.numClasses))]))
            model.lossLayers.append(end_module)
        elif model.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                or model.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
            for block_id in range(model.pathCounts[-1]):
                end_module = nn.Sequential(OrderedDict([('logits', nn.LazyLinear(out_features=model.numClasses))]))
                model.lossLayers.append(end_module)

    @staticmethod
    def create_cigt_blocks(model):
        block_in_planes = model.inputDims[0]
        for block_id, block_dict in enumerate(model.blockParametersList):
            path_count_in_layer = block_dict["path_count"]
            block_elements = block_dict["layer_structure"]
            cigt_layer_blocks = nn.ModuleList()

            in_planes_list = []
            for path_id in range(path_count_in_layer):
                layers = []
                in_planes_list = [block_in_planes]
                for layer_dict in block_elements:
                    if layer_dict["layer_type"] == "conv":
                        out_planes = layer_dict["feature_map_count"]
                        stride = layer_dict["strides"]
                        kernel_size = layer_dict["kernel_size"]
                        use_max_pool = layer_dict["use_max_pool"]
                        use_batch_normalization = layer_dict["use_batch_normalization"]

                        conv_layer = nn.Conv2d(in_planes_list[-1], out_planes, kernel_size=kernel_size,
                                               stride=stride, padding="same", bias=True)
                        in_planes_list.append(out_planes)
                        layers.append(conv_layer)

                        if use_batch_normalization:
                            bn_layer = nn.BatchNorm2d(num_features=out_planes)
                            layers.append(bn_layer)

                        activation_layer = nn.ReLU()
                        layers.append(activation_layer)

                        if use_max_pool:
                            max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                            layers.append(max_pool_layer)
                    elif layer_dict["layer_type"] == "flatten":
                        flatten_layer = nn.Flatten()
                        layers.append(flatten_layer)
                    elif layer_dict["layer_type"] == "fc":
                        # Apply flatten if the previous layer was a convolution
                        dimension = layer_dict["dimension"]
                        use_dropout = layer_dict["use_dropout"]
                        use_batch_normalization = layer_dict["use_batch_normalization"]

                        fc_layer = nn.LazyLinear(dimension, bias=True)
                        layers.append(fc_layer)

                        if use_batch_normalization:
                            bn_layer = nn.BatchNorm1d(num_features=dimension)
                            layers.append(bn_layer)

                        activation_layer = nn.ReLU()
                        layers.append(activation_layer)

                        if use_dropout:
                            dropout_layer = nn.Dropout(p=model.classificationDropout)
                            layers.append(dropout_layer)
                    else:
                        raise NotImplementedError()
                block_obj = Sequential_ext(*layers)
                cigt_layer_blocks.append(block_obj)

            model.cigtLayers.append(cigt_layer_blocks)
            # Block end layers: Routing layers for inner layers, loss layer for the last one.
            if block_id < len(model.blockParametersList) - 1:
                assert block_elements[-1]["layer_type"] == "conv"
                assert in_planes_list[-1] == block_elements[-1]["feature_map_count"]
                routing_layer = model.get_routing_layer(cigt_layer_id=block_id,
                                                        input_feature_map_count=in_planes_list[-1])
                model.blockEndLayers.append(routing_layer)
            block_in_planes = in_planes_list[-1]

        LenetConfigInterpreter.get_loss_layer(model=model)
