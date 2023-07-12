from collections import OrderedDict

import torch.nn as nn

from cigt.cigt_model import BasicBlock, Sequential_ext


class ResnetConfigInterpreter:

    def __init__(self):
        pass

    @staticmethod
    def interpret_config_list(configs):
        block_list = []
        # Unravel the configuration information into a complete block by block list.
        for block_id, block_config_dict in enumerate(configs.layer_config_list):
            path_count = block_config_dict["path_count"]
            for idx, d_ in enumerate(block_config_dict["layer_structure"]):
                for idy in range(d_["layer_count"]):
                    block_list.append((block_id, path_count, d_["feature_map_count"]))

        block_parameters_dict = {}
        for layer_id, layer_info in enumerate(block_list):
            block_id = layer_info[0]
            path_count = layer_info[1]
            feature_map_count = layer_info[2]
            if block_id not in block_parameters_dict:
                block_parameters_dict[block_id] = []
            block_options = {}
            if layer_id == 0:
                block_options["in_dimension"] = configs.first_conv_output_dim
                block_options["input_path_count"] = 1
            else:
                path_count_prev = block_list[layer_id - 1][1]
                feature_map_count_prev = block_list[layer_id - 1][2]
                block_options["in_dimension"] = feature_map_count_prev
                block_options["input_path_count"] = path_count_prev
            block_options["layer_id"] = layer_id
            block_options["block_id"] = block_id
            block_options["out_dimension"] = feature_map_count
            block_options["output_path_count"] = path_count
            if layer_id in configs.double_stride_layers:
                block_options["stride"] = 2
            else:
                block_options["stride"] = 1
            block_parameters_dict[block_id].append(block_options)
        block_parameters_list = sorted([(k, v) for k, v in block_parameters_dict.items()], key=lambda tpl: tpl[0])
        return block_parameters_list

    @staticmethod
    def get_explanation(model, explanation, kv_rows):
        block_params = [(block_id, block_config_list) for block_id, block_config_list in model.blockParametersList]
        block_params = sorted(block_params, key=lambda t__: t__[0])

        layer_id = 0
        for t_ in block_params:
            block_id = t_[0]
            block_config_list = t_[1]
            for block_config_dict in block_config_list:
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} in_dimension".format(layer_id),
                                                    value=block_config_dict["in_dimension"],
                                                    explanation=explanation, kv_rows=kv_rows)
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} input_path_count".format(layer_id),
                                                    value=block_config_dict["input_path_count"],
                                                    explanation=explanation, kv_rows=kv_rows)
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} layer_id".format(layer_id),
                                                    value=layer_id,
                                                    explanation=explanation, kv_rows=kv_rows)
                assert block_id == block_config_dict["block_id"]
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} block_id".format(layer_id),
                                                    value=block_config_dict["block_id"],
                                                    explanation=explanation, kv_rows=kv_rows)
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} out_dimension".format(layer_id),
                                                    value=block_config_dict["out_dimension"],
                                                    explanation=explanation, kv_rows=kv_rows)
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} output_path_count".format(layer_id),
                                                    value=block_config_dict["output_path_count"],
                                                    explanation=explanation, kv_rows=kv_rows)
                explanation = model.add_explanation(name_of_param="BasicBlock_{0} stride".format(layer_id),
                                                    value=block_config_dict["stride"],
                                                    explanation=explanation, kv_rows=kv_rows)
                layer_id += 1

        return kv_rows, explanation

    @staticmethod
    def get_loss_layer(model, final_layer_dimension_multiplier=1):
        model.lossLayers = nn.ModuleList()
        final_feature_dimension = final_layer_dimension_multiplier * \
                                  model.blockParametersList[-1][-1][-1]["out_dimension"]
        if model.lossCalculationKind == "SingleLogitSingleLoss":
            end_module = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=8)),
                ('flatten', nn.Flatten()),
                ('logits', nn.Linear(in_features=final_feature_dimension, out_features=model.numClasses))
            ]))
            model.lossLayers.append(end_module)
        elif model.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                or model.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
            for block_id in range(model.pathCounts[-1]):
                end_module = nn.Sequential(OrderedDict([
                    ('avg_pool_{0}'.format(block_id), nn.AvgPool2d(kernel_size=8)),
                    ('flatten_{0}'.format(block_id), nn.Flatten()),
                    ('logits_{0}'.format(block_id), nn.Linear(
                        in_features=final_feature_dimension, out_features=model.numClasses))
                ]))
                model.lossLayers.append(end_module)

    @staticmethod
    def create_cigt_blocks(model):
        curr_input_shape = (model.batchSize, *model.inputDims)
        feature_edge_size = curr_input_shape[-1]
        for cigt_layer_id, cigt_layer_info in model.blockParametersList:
            path_count_in_layer = model.pathCounts[cigt_layer_id]
            cigt_layer_blocks = nn.ModuleList()
            for path_id in range(path_count_in_layer):
                layers = []
                for inner_block_info in cigt_layer_info:
                    block = BasicBlock(in_planes=inner_block_info["in_dimension"],
                                       planes=inner_block_info["out_dimension"],
                                       stride=inner_block_info["stride"],
                                       norm_type=model.batchNormType)
                    layers.append(block)
                block_obj = Sequential_ext(*layers)
                if model.useDataParallelism:
                    block_obj = nn.DataParallel(block_obj)
                # block_obj.name = "block_{0}_{1}".format(cigt_layer_id, path_id)
                cigt_layer_blocks.append(block_obj)
            model.cigtLayers.append(cigt_layer_blocks)
            # Block end layers: Routing layers for inner layers, loss layer for the last one.
            if cigt_layer_id < len(model.blockParametersList) - 1:
                for inner_block_info in cigt_layer_info:
                    feature_edge_size = int(feature_edge_size / inner_block_info["stride"])
                routing_layer = model.get_routing_layer(cigt_layer_id=cigt_layer_id,
                                                        input_feature_map_size=feature_edge_size,
                                                        input_feature_map_count=cigt_layer_info[-1]["out_dimension"])
                if model.useDataParallelism:
                    routing_layer = nn.DataParallel(routing_layer)
                model.blockEndLayers.append(routing_layer)
        # if cigt_layer_id == len(self.blockParametersList) - 1:
        ResnetConfigInterpreter.get_loss_layer(model=model)
