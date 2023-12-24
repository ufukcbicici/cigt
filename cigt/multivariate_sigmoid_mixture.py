import numpy as np
from sklearn.mixture import GaussianMixture

from cigt.sigmoid_gaussian_mixture import SigmoidGaussianMixture


class MultivariateSigmoidMixture(SigmoidGaussianMixture):

    # covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    def __init__(self, num_of_components, dimension_intervals, covariance_type):
        super().__init__(num_of_components)
        self.covarianceType = covariance_type
        self.dimensionIntervals = dimension_intervals
        self.dimension = self.dimensionIntervals.shape[0]
        self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
                                               tol=1e-6,
                                               max_iter=10000, covariance_type=covariance_type)

    def sample(self, num_of_samples):
        # Sample from Gaussian Mixture first
        if self.isFitted:
            a = self.dimensionIntervals[:, 0]
            b = self.dimensionIntervals[:, 1]
            s = self.gaussianMixture.sample(n_samples=num_of_samples)[0]
            # Apply sigmoid
            s2 = 1.0 + np.exp(-s)
            y = np.reciprocal(s2)
            # Convert into target interval
            # a = self.lowEnd
            # b = self.highEnd
            y_hat = np.expand_dims(b - a, axis=0) * y + np.expand_dims(a, axis=0)
        else:
            y_hat = []
            for dim_id in range(self.dimension):
                low_end = self.dimensionIntervals[dim_id, 0]
                high_end = self.dimensionIntervals[dim_id, 1]
                y_dim_hat = np.random.uniform(low=low_end, high=high_end, size=(num_of_samples,))
                y_hat.append(y_dim_hat)
            y_hat = np.stack(y_hat, axis=1)
        return y_hat

    def apply_g_inverse(self, y):
        a = self.dimensionIntervals[:, 0]
        b = self.dimensionIntervals[:, 1]
        y_minus_a = y - np.expand_dims(a, axis=0)
        b_minus_y = np.expand_dims(b, axis=0) - y
        y_hat = y_minus_a * np.reciprocal(b_minus_y)
        x = np.log(y_hat)
        return x

    def apply_g_inverse_test(self, y):
        x_1 = self.apply_g_inverse(y=y)
        x_2 = []
        for idx in range(self.dimension):
            self.lowEnd = self.dimensionIntervals[idx, 0]
            self.highEnd = self.dimensionIntervals[idx, 1]
            x_2_vec = super().apply_g_inverse(y=y[:, idx])
            x_2.append(x_2_vec)
        x_2 = np.stack(x_2, axis=-1)
        assert np.allclose(x_1, x_2)

    def sample_test(self, num_of_samples):
        a = self.dimensionIntervals[:, 0]
        b = self.dimensionIntervals[:, 1]
        sample_matrix = self.gaussianMixture.sample(n_samples=num_of_samples)[0]
        s2 = 1.0 + np.exp(-sample_matrix)
        y = np.reciprocal(s2)
        y_hat1 = np.expand_dims(b - a, axis=0) * y + np.expand_dims(a, axis=0)

        y_list = []
        for idx in range(self.dimension):
            s = sample_matrix[:, idx]
            s2 = 1.0 + np.exp(-s)
            y = np.reciprocal(s2)
            # Convert into target interval
            a = self.dimensionIntervals[idx, 0]
            b = self.dimensionIntervals[idx, 1]
            y_hat = (b - a) * y + a
            y_list.append(y_hat)
        y_hat2 = np.stack(y_list, axis=1)
        assert np.allclose(y_hat1, y_hat2)

    def fit(self, data):
        # Data is in y=g(x). We need to calculate g^{-1}(y)=x.
        x = self.apply_g_inverse(y=data)
        self.gaussianMixture.fit(X=x)
        self.isFitted = True
