import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

from auxillary.rump_dataset import RumpDataset
from auxillary.utilities import Utilities


class SimilarDatasetDivisionAlgorithm:

    @staticmethod
    def run(model, parent_loader, max_accuracy_dif, division_ratio, save_path, max_trials=1000):
        X_ = []
        y_ = []
        for i, (input_, target) in enumerate(parent_loader):
            X_.append(input_)
            y_.append(target)

        X_ = torch.concat(X_, dim=0).numpy()
        y_ = torch.concat(y_, dim=0).numpy()

        # Original performance of the complete dataset
        kwargs = {'num_workers': 2, 'pin_memory': True}
        # full_dataset = RumpDataset(X_=torch.from_numpy(X_), y_=torch.from_numpy(y_))
        # full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1024, shuffle=False, **kwargs)
        # model.validate(loader=full_loader, temperature=0.1, epoch=0, data_kind="test")

        for trial_id in tqdm(range(max_trials)):
            print("Split trial:{0}".format(trial_id))
            X_0, X_1, y_0, y_1, indices_0, indices_1 = train_test_split(X_, y_, np.arange(X_.shape[0]),
                                                                        test_size=division_ratio, shuffle=True,
                                                                        stratify=y_)
            assert np.array_equal(X_0, X_[indices_0])
            assert np.array_equal(X_1, X_[indices_1])
            assert np.array_equal(y_0, y_[indices_0])
            assert np.array_equal(y_1, y_[indices_1])
            print("Split 0:{0}".format(Counter(y_0)))
            print("Split 1:{0}".format(Counter(y_1)))
            dataset_0 = RumpDataset(X_=torch.from_numpy(X_0), y_=torch.from_numpy(y_0),
                                    indices_in_original_dataset=indices_0)
            data_loader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=1024, shuffle=False, **kwargs)
            dataset_1 = RumpDataset(X_=torch.from_numpy(X_1), y_=torch.from_numpy(y_1),
                                    indices_in_original_dataset=indices_1)
            data_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=1024, shuffle=False, **kwargs)
            acc_0 = model.validate(loader=data_loader_0, temperature=0.1, epoch=0, data_kind="test")
            acc_1 = model.validate(loader=data_loader_1, temperature=0.1, epoch=0, data_kind="test")

            if np.abs(acc_1 - acc_0) <= max_accuracy_dif:
                Utilities.pickle_save_to_file(path=save_path,
                                              file_content={"dataset_0": dataset_0, "dataset_1": dataset_1})
                loaded_datasets = Utilities.pickle_load_from_file(path=save_path)

                # Validate that the datasets have been correctly stored.
                assert np.array_equal(dataset_0.X_, loaded_datasets["dataset_0"].X_)
                assert np.array_equal(dataset_0.y_, loaded_datasets["dataset_0"].y_)
                assert np.array_equal(dataset_0.indicesInOriginalDataset,
                                      loaded_datasets["dataset_0"].indicesInOriginalDataset)
                assert np.array_equal(dataset_1.X_, loaded_datasets["dataset_1"].X_)
                assert np.array_equal(dataset_1.y_, loaded_datasets["dataset_1"].y_)
                assert np.array_equal(dataset_1.indicesInOriginalDataset,
                                      loaded_datasets["dataset_1"].indicesInOriginalDataset)
                data_loader_0_loaded = torch.utils.data.DataLoader(loaded_datasets["dataset_0"],
                                                                   batch_size=1024, shuffle=False, **kwargs)
                data_loader_1_loaded = torch.utils.data.DataLoader(loaded_datasets["dataset_1"],
                                                                   batch_size=1024, shuffle=False, **kwargs)
                loaded_acc_0 = model.validate(loader=data_loader_0_loaded, temperature=0.1, epoch=0, data_kind="test")
                loaded_acc_1 = model.validate(loader=data_loader_1_loaded, temperature=0.1, epoch=0, data_kind="test")
                assert acc_0 == loaded_acc_0
                assert acc_1 == loaded_acc_1
                return dataset_0, dataset_1

        return None
