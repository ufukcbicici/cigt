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
            # s = self.gaussianMixture.sample(n_samples=num_of_samples)[0]
            # s = s[:, 0]
            # s2 = 1.0 + np.exp(-s)
            # y = np.reciprocal(s2)
            # # Convert into target interval
            # a = self.lowEnd
            # b = self.highEnd
            # y_hat = (b - a) * y + a
            pass
        else:
            y_hat = []
            for dim_id in range(self.dimension):
                low_end = self.dimensionIntervals[dim_id, 0]
                high_end = self.dimensionIntervals[dim_id, 1]
                y_dim_hat = np.random.uniform(low=low_end, high=high_end, size=(num_of_samples,))
                y_hat.append(y_dim_hat)
            y_hat = np.stack(y_hat, axis=1)
        return y_hat
