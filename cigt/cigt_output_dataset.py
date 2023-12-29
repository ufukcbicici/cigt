import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation


class CigtOutputDataset(Dataset):
    def __init__(self, input_reduction_factor=0):
        self.listOfFields = []
        self.dataArrays = {}
        self.dataLength = None
        self.inputReductionFactor = input_reduction_factor
        # self.inputReductionFactor = configs.policy_networks_cbam_layer_input_reduction_ratio

    def load_from_model(self, model, data_loader, repeat_count, list_of_fields):
        self.listOfFields = list_of_fields
        for field_name in self.listOfFields:
            self.dataArrays[field_name] = {}

        # valid_route_combinations = set()
        # all_route_combinations = []
        # for layer_id in range(len(model.pathCounts)):
        #     paths = model.pathCounts[:(layer_id + 1)]
        #     combinations = Utilities.create_route_combinations(shape_=paths)
        #     all_route_combinations.extend(combinations)
        #
        # for comb in all_route_combinations:
        #     if len(comb) < len(model.pathCounts):
        #         valid_route_combinations.add(comb)

        model.eval()
        assert isinstance(model, CigtIgGatherScatterImplementation)

        if self.inputReductionFactor > 1:
            conv_block_reduction_layer = nn.MaxPool2d(kernel_size=self.inputReductionFactor,
                                                      stride=self.inputReductionFactor)
        else:
            conv_block_reduction_layer = nn.Identity()

        for epoch_id in range(repeat_count):
            print("Epoch {0} is starting.".format(epoch_id))
            for i, (input_, target) in tqdm(enumerate(data_loader)):
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input_).to(model.device)
                    target_var = torch.autograd.Variable(target).to(model.device)
                    batch_size = input_var.size(0)
                    cigt_outputs = model.forward_v2(x=input_var, labels=target_var, temperature=1.0)
                    for field_name in self.listOfFields:
                        outputs_dict = cigt_outputs[field_name]
                        for k, v in outputs_dict.items():
                            # Skip block outputs for the last layer, these are not used.
                            if field_name == "block_outputs_dict" and (len(k) == len(model.pathCounts) or k == ()):
                                continue
                            if k not in self.dataArrays[field_name]:
                                self.dataArrays[field_name][k] = []

                            if field_name == "block_outputs_dict":
                                v_reduced = conv_block_reduction_layer(v)
                                self.dataArrays[field_name][k].append(v_reduced.detach().cpu().numpy())
                            else:
                                self.dataArrays[field_name][k].append(v.detach().cpu().numpy())
        # Merge all outputs from all iterations into large arrays
        merged_dict = {}
        for field_name in self.listOfFields:
            merged_dict[field_name] = {}
            for k, v in self.dataArrays[field_name].items():
                merged_dict[field_name][k] = np.concatenate(v, axis=0)
        self.dataArrays = merged_dict

        # Delete redundant label arrays
        for k in self.dataArrays["labels_dict"]:
            assert np.array_equal(self.dataArrays["labels_dict"][()], self.dataArrays["labels_dict"][k])
        self.dataArrays["labels_dict"] = {(): self.dataArrays["labels_dict"][()]}

        data_lengths = set()
        for field_name in self.listOfFields:
            for k, v in self.dataArrays[field_name].items():
                data_lengths.add(v.shape[0])

        assert len(data_lengths) == 1
        self.dataLength = list(data_lengths)[0]

    def save(self, file_path):
        Utilities.pickle_save_to_file(path=file_path, file_content=self)

    def load_from_file(self, file_path):
        obj = Utilities.pickle_load_from_file(path=file_path)
        self.listOfFields = obj.listOfFields
        self.dataArrays = obj.dataArrays
        self.dataLength = obj.dataLength

    def __len__(self):
        return self.dataLength

    def __getitem__(self, idx):
        sampled_dict = {}
        for field_name in self.listOfFields:
            sampled_dict[field_name] = {}
            for k, v in self.dataArrays[field_name].items():
                sampled_dict[field_name][k] = v[idx]
        return sampled_dict
