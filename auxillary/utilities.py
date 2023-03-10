import numpy as np
import pickle
import itertools


class Utilities:
    def __init__(self):
        pass

    INFO_GAIN_LOG_EPSILON = 1e-30

    @staticmethod
    def calculate_mac_of_computation(num_of_input_channels,
                                     height_of_input_map,
                                     width_of_input_map,
                                     height_of_filter,
                                     width_of_filter,
                                     num_of_output_channels,
                                     convolution_stride,
                                     type="conv"):
        if type == "conv":
            C = num_of_input_channels
            H = height_of_input_map
            W = width_of_input_map
            R = height_of_filter
            S = width_of_filter
            M = num_of_output_channels
            # E = height_of_output_map
            # F = width_of_output_map
            U = convolution_stride
            E = (H - R + U) / U
            F = (W - S + U) / U
            cost = M * F * E * R * S * C
            # for m in range(M):
            #     for x in range(F):
            #         for y in range(E):
            #             for i in range(R):
            #                 for j in range(S):
            #                     for k in range(C):
            #                         cost += 1
            return cost
        elif type == "depth_seperable":
            C = num_of_input_channels
            H = height_of_input_map
            W = width_of_input_map
            R = height_of_filter
            S = width_of_filter
            M = num_of_output_channels
            # E = height_of_output_map
            # F = width_of_output_map
            U = convolution_stride
            E = (H - R + U) / U
            F = (W - S + U) / U
            cost = E * F * C * (R * S + M)
            return cost
        elif type == "fc":
            C = num_of_input_channels
            H = 1
            W = 1
            R = 1
            S = 1
            M = num_of_output_channels
            E = 1
            F = 1
        else:
            raise NotImplementedError()
        cost = M * F * E * R * S * C
        return cost

    @staticmethod
    def get_variable_name(name, node, prefix=""):
        return "{0}Node{1}_{2}".format(prefix, node.index, name)

    # @staticmethod
    # def compare_model_list_outputs(l_1, l_2):
    #     assert len(l_1) == len(l_2)
    #     for arr_1, arr_2 in zip(l_1, l_1):
    #         assert isinstance(arr_1, tf.Tensor)
    #         assert isinstance(arr_2, tf.Tensor)
    #         is_equal = np.array_equal(arr_1.numpy(), arr_2.numpy())
    #         if not is_equal:
    #             return False
    #     return True

    # @staticmethod
    # def compare_model_dictionary_outputs(d_1, d_2):
    #     for k_1 in d_1.keys():
    #         assert k_1 in d_2
    #         v_1 = d_1[k_1]
    #         v_2 = d_2[k_1]
    #         if isinstance(v_1, dict):
    #             assert isinstance(v_2, dict)
    #             is_equal = Utilities.compare_model_dictionary_outputs(d_1=v_1, d_2=v_2)
    #             if not is_equal:
    #                 return False
    #         elif isinstance(v_1, list):
    #             assert isinstance(v_2, list)
    #             is_equal = Utilities.compare_model_list_outputs(l_1=v_1, l_2=v_2)
    #             if not is_equal:
    #                 return False
    #         elif isinstance(v_1, tf.Tensor):
    #             assert isinstance(v_2, tf.Tensor)
    #             is_equal = np.array_equal(v_1.numpy(), v_2.numpy())
    #             if not is_equal:
    #                 return False
    #         else:
    #             raise NotImplementedError()
    #     return True

    @staticmethod
    def compare_model_outputs(output_1, output_2):
        assert isinstance(output_1, dict)
        assert isinstance(output_2, dict)
        assert len(output_1) == len(output_2)

        for _key in output_1.keys():
            assert _key in output_2
            v_1 = output_1[_key]
            v_2 = output_2[_key]
            assert (isinstance(v_1, dict) and isinstance(v_2, dict)) or \
                   (isinstance(v_1, list) and isinstance(v_2, list))
            if isinstance(v_1, dict):
                is_equal = Utilities.compare_model_dictionary_outputs(d_1=v_1, d_2=v_2)
                if not is_equal:
                    return False
            elif isinstance(v_2, list):
                is_equal = Utilities.compare_model_list_outputs(l_1=v_1, l_2=v_2)
                if not is_equal:
                    return False
            else:
                raise NotImplementedError()
        return True

    @staticmethod
    def pickle_save_to_file(path, file_content):
        f = open(path, "wb")
        pickle.dump(file_content, f)
        f.close()

    @staticmethod
    def pickle_load_from_file(path):
        f = open(path, "rb")
        content = pickle.load(f)
        f.close()
        return content

    @staticmethod
    def concat_to_np_array_dict(dct, key, array):
        if key not in dct:
            if not np.isscalar(array):
                dct[key] = array
            else:
                scalar = array
                dct[key] = np.array(scalar)
        else:
            if not np.isscalar(array):
                dct[key] = np.concatenate((dct[key], array))
            else:
                scalar = array
                dct[key] = np.append(dct[key], scalar)

    @staticmethod
    def concat_to_np_array_dict_v2(dct, key, array):
        if key not in dct:
            if not np.isscalar(array):
                dct[key] = [array]
            else:
                scalar = array
                dct[key] = np.array(scalar)
        else:
            if not np.isscalar(array):
                dct[key].append(array)
            else:
                scalar = array
                dct[key] = np.append(dct[key], scalar)

    @staticmethod
    def merge_dict_of_ndarrays(dict_target, dict_to_append):
        # key_set1 = set(dict_target.keys())
        # key_set2 = set(dict_to_append.keys())
        # assert key_set1 == key_set2
        for k in dict_to_append.keys():
            Utilities.concat_to_np_array_dict(dct=dict_target, array=dict_to_append[k], key=k)

    @staticmethod
    def get_cartesian_product(list_of_lists):
        cartesian_product = list(itertools.product(*list_of_lists))
        return cartesian_product

    @staticmethod
    def append_dict_to_dict(dict_destination, dict_source, convert_to_numpy=True):
        for k, v in dict_source.items():
            arr = v.numpy() if convert_to_numpy else v
            if k not in dict_destination:
                dict_destination[k] = [arr]
            else:
                dict_destination[k].append(arr)

    @staticmethod
    def discretize_value(sampled_value, interval_start, interval_end, discrete_values):
        sorted_values = sorted(discrete_values)
        interval_length = interval_end - interval_start
        step_length = interval_length / len(discrete_values)
        interval_id = min(int((sampled_value - interval_start) / step_length), len(sorted_values) - 1)
        discretized_value = sorted_values[interval_id]
        return discretized_value

    @staticmethod
    def concatenate_dict_of_arrays(dict_, axis):
        for k in dict_.keys():
            dict_[k] = np.concatenate(dict_[k], axis=axis)

    @staticmethod
    def calculate_entropies(prob_distributions):
        log_prob = np.log(prob_distributions + Utilities.INFO_GAIN_LOG_EPSILON)
        # is_inf = tf.is_inf(log_prob)
        # zero_tensor = tf.zeros_like(log_prob)
        # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
        prob_log_prob = prob_distributions * log_prob
        entropies = -1.0 * np.sum(prob_log_prob, axis=1)
        return entropies

    @staticmethod
    def divide_array_into_chunks(arr, count):
        chunk_size = len(arr) // count
        curr_index = 0
        list_of_chunks = []
        for idx in range(count):
            if idx != count - 1:
                list_of_chunks.append(arr[curr_index:curr_index + chunk_size])
            else:
                list_of_chunks.append(arr[curr_index:])
            curr_index += chunk_size
        return list_of_chunks

    @staticmethod
    def one_hot_numpy(arr):
        assert len(arr.shape) == 2
        arg_max_indices = np.argmax(arr, axis=1)
        one_hot_matrix = np.zeros(shape=arr.shape, dtype=np.int32)
        one_hot_matrix[np.arange(arr.shape[0]), arg_max_indices] = 1
        return one_hot_matrix

    @staticmethod
    def convert_trajectory_array_to_indices(trajectory_array):
        path_indices = tuple([trajectory_array[:, idx] for idx in range(trajectory_array.shape[1])])
        return path_indices

