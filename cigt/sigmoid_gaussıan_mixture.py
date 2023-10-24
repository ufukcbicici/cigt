import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


class SigmoidGaussianMixture(object):
    def __init__(self, num_of_components, low_end=0.0, high_end=1.0, name=""):
        self.name = name
        self.numOfComponents = num_of_components
        self.lowEnd = low_end
        self.highEnd = high_end
        self.isFitted = False
        self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
                                               tol=1e-6, max_iter=10000)

    def init_gmm(self, means, variances, weights, num_of_random_samples=100000):
        assert len(means) == len(variances) and len(means) == len(weights)
        self.numOfComponents = len(means)
        self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
                                               tol=1e-6, max_iter=10000)

        # Generate synthetic samples from three normals.
        data = []
        component_probs = np.array(weights) / np.sum(weights)
        component_selections = np.random.choice(
            a=self.numOfComponents,
            p=component_probs,
            size=(num_of_random_samples,))
        component_samples = []
        for comp_id in range(self.numOfComponents):
            mean = means[comp_id]
            variance = variances[comp_id]
            scale = np.sqrt(variance)
            s = np.random.normal(loc=mean,
                                 scale=scale,
                                 size=(num_of_random_samples,))
            component_samples.append(s)
        component_samples = np.stack(component_samples, axis=1)

        samples_selected = component_samples[np.arange(num_of_random_samples), component_selections]
        self.gaussianMixture.fit(X=np.expand_dims(samples_selected, axis=-1))
        self.isFitted = True

    # OK
    def sample(self, num_of_samples):
        # Sample from Gaussian Mixture first
        if self.isFitted:
            s = self.gaussianMixture.sample(n_samples=num_of_samples)[0]
            s = s[:, 0]
            s2 = 1.0 + np.exp(-s)
            y = np.reciprocal(s2)
            # Convert into target interval
            a = self.lowEnd
            b = self.highEnd
            y_hat = (b - a) * y + a
        else:
            y_hat = np.random.uniform(low=self.lowEnd, high=self.highEnd, size=(num_of_samples,))
        return y_hat

    def apply_g_inverse(self, y):
        a = self.lowEnd
        b = self.highEnd
        y_minus_a = y - a
        b_minus_y = b - y
        y_hat = y_minus_a * np.reciprocal(b_minus_y)
        x = np.log(y_hat)
        return x

    def fit(self, data):
        # Data is in y=g(x). We need to calculate g^{-1}(y)=x.
        x = self.apply_g_inverse(y=data)
        self.gaussianMixture.fit(X=np.expand_dims(x, axis=-1))
        self.isFitted = True

    def pdf(self, y):
        assert np.all(np.greater_equal(y, 0.0))
        assert np.all(np.less_equal(y, 1.0))
        x = self.apply_g_inverse(y=y)
        # Calculate GMM's densities for every sample.
        component_densities = []
        for comp_idx in range(self.numOfComponents):
            component_mean = self.gaussianMixture.means_[comp_idx][0]
            component_variance = self.gaussianMixture.covariances_[comp_idx, 0, 0]
            component_scale = np.sqrt(component_variance)
            pdf_x = norm.pdf(x=x, loc=component_mean, scale=component_scale)
            weighted_pdf_x = pdf_x * self.gaussianMixture.weights_[comp_idx]
            component_densities.append(weighted_pdf_x)
        component_densities = np.stack(component_densities, axis=1)
        mixture_pdf = np.sum(component_densities, axis=1)
        # Calculate the coefficient |d(g^{-1}(y))/dy|
        a = self.lowEnd
        b = self.highEnd
        a_minus_b = a - b
        A_ = (y - a) * (y - b)
        coeff = np.abs(a_minus_b * np.reciprocal(A_))
        final_densities = mixture_pdf * coeff
        return final_densities

    def draw_pdf(self):
        y = np.linspace(self.lowEnd + (self.highEnd - self.lowEnd) * 0.001,
                        self.highEnd - (self.highEnd - self.lowEnd) * 0.001,
                        10000)
        for comp_idx in range(self.numOfComponents):
            fig, ax = plt.subplots(1, 1)
            component_mean = self.gaussianMixture.means_[comp_idx][0]
            component_variance = self.gaussianMixture.covariances_[comp_idx, 0, 0]
            component_scale = np.sqrt(component_variance)
            pi_idx = self.gaussianMixture.weights_[comp_idx]
            min_x_component = -10.0
            max_x_component = 10.0
            x_domain = np.linspace(min_x_component, max_x_component, 10000)
            pdf_x = norm.pdf(x=x_domain, loc=component_mean, scale=component_scale)
            ax.set_title("Gaussian Component {0} mean{0}={1} scale{0}={2} pi{0}={3}".format(
                comp_idx, "{:.3f}".format(component_mean), "{:.3f}".format(component_scale), "{:.3f}".format(pi_idx)))
            ax.plot(x_domain, pdf_x, 'r-', lw=1, alpha=1.0, label='Gaussian Component Pdf')
            fig.tight_layout()
            plt.show()

        fig, ax = plt.subplots(1, 1)
        mixture_pdf = self.pdf(y=y)
        ax.set_title("Mixture")
        ax.plot(y, mixture_pdf, 'r-', lw=1, alpha=1.0, label='Gaussian Mixture Pdf')

        # Sample from the current distribution
        samples = self.sample(num_of_samples=100000)
        ax.hist(samples, density=True, histtype="stepfilled", alpha=0.2, bins=100)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        plt.show()

    def check_mle():
        low_end = 0.2
        high_end = 0.21
        sigmoid_gmm = SigmoidGaussianMixture(num_of_components=3,
                                             name="First Distribution",
                                             low_end=low_end,
                                             high_end=high_end)
        sigmoid_gmm.init_gmm(means=[-1.0, 1.0, 3.0], variances=[0.01, 0.02, 0.1],
                             weights=[1.0, 1.0, 1.0])
        sigmoid_gmm.draw_pdf()
        sigmoid_gmm_2 = SigmoidGaussianMixture(num_of_components=3,
                                               name="Second Distribution",
                                               low_end=low_end,
                                               high_end=high_end)

        # Sample data from the first distribution for the ML estimation in the other.
        samples = sigmoid_gmm.sample(num_of_samples=100000)
        sigmoid_gmm_2.fit(data=samples)

        for component_id in range(sigmoid_gmm_2.numOfComponents):
            print("mu_{0}={1} sigma_{0}={2} weight_{0}={3}".format(
                component_id,
                sigmoid_gmm_2.gaussianMixture.means_[component_id, 0],
                sigmoid_gmm_2.gaussianMixture.covariances_[component_id, 0, 0],
                sigmoid_gmm_2.gaussianMixture.weights_[component_id]))

        sigmoid_gmm_2.draw_pdf()


if __name__ == "__main__":
    sigmoid_gaussian_mixture = SigmoidGaussianMixture(num_of_components=3, low_end=0.2, high_end=0.5)
    sigmoid_gaussian_mixture.init_gmm(means=[-1.0, 1.0, 4.0],
                                      variances=[0.01, 0.2, 1.25],
                                      weights=[1.0, 2.0, 1.5])
    sigmoid_gaussian_mixture.draw_pdf()
    print("X")

    # beta_mixture.draw_pdf()
    #
    # beta_mixture_trained = BetaMixture(component_count=3, learning_rate=1e-3)
    # beta_mixture_trained.mixtureParameters = torch.nn.Parameter(torch.from_numpy(np.array([1.0, 2.0, 2.5])))
    # beta_mixture_trained.componentParameters = torch.nn.Parameter(
    #     torch.from_numpy(
    #         np.array([[2.0, 3.0], [0.5, 1.5], [4.5, 0.75]])))
    #
    # beta_mixture_trained.draw_pdf()
    # samples = beta_mixture_trained.sample(num_of_samples=10000)
    #
    # beta_mixture.fit(data=samples, iteration_count=100000)
    #
    # print("X")

#
#
# import numpy as np
# import os
# from scipy.stats import norm
# import tensorflow as tf
# from sklearn.mixture import GaussianMixture
# from tqdm import tqdm
# from scipy import integrate
# import matplotlib.pyplot as plt
#
#
# class SigmoidMixtureOfGaussians:
#     def __init__(self, num_of_components, low_end=0.0, high_end=1.0, name=""):
#         self.name = name
#         self.numOfComponents = num_of_components
#         self.lowEnd = low_end
#         self.highEnd = high_end
#         self.isFitted = False
#         self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
#                                                tol=1e-6, max_iter=10000)
#
#     # OK
#     def sample(self, num_of_samples):
#         # Sample from Gaussian Mixture first
#         if self.isFitted:
#             s = self.gaussianMixture.sample(n_samples=num_of_samples)[0]
#             s = s[:, 0]
#             s2 = 1.0 + np.exp(-s)
#             y = np.reciprocal(s2)
#             # Convert into target interval
#             a = self.lowEnd
#             b = self.highEnd
#             y_hat = (b - a) * y + a
#         else:
#             y_hat = np.random.uniform(low=self.lowEnd, high=self.highEnd, size=(num_of_samples,))
#         return y_hat
#
#     # OK
#     def pdf(self, y):
#         assert np.all(np.greater_equal(y, 0.0))
#         assert np.all(np.less_equal(y, 1.0))
#         y_hat = y / (1.0 - y)
#         x = np.log(y_hat)
#         a = y - np.square(y)
#         a = np.reciprocal(a)
#         # Calculate GMM's densities for every sample.
#         component_densities = []
#         for comp_idx in range(self.numOfComponents):
#             component_mean = self.gaussianMixture.means_[comp_idx][0]
#             component_variance = self.gaussianMixture.covariances_[comp_idx, 0, 0]
#             component_scale = np.sqrt(component_variance)
#             pdf_x = norm.pdf(x=x, loc=component_mean, scale=component_scale)
#             weighted_pdf_x = pdf_x * self.gaussianMixture.weights_[comp_idx]
#             component_densities.append(weighted_pdf_x)
#         component_densities = np.stack(component_densities, axis=1)
#         mixture_pdf = np.sum(component_densities, axis=1)
#         mixture_pdf = mixture_pdf * a
#         return mixture_pdf
#
#     def maximum_likelihood_estimate(self, data, alpha):
#         # Convert data to the original scale
#         a = self.lowEnd
#         b = self.highEnd
#         data = (data - a) * (1.0 / (b - a))
#         data = np.clip(a=data, a_min=0.0 + 1.0e-10, a_max=1.0 - 1.0e-10)
#         y_hat = data / (1.0 - data)
#         x = np.log(y_hat)
#         self.gaussianMixture.fit(X=np.expand_dims(x, axis=-1))
#         self.isFitted = True
#
#     def plot_distribution(self, show_plot=True, root_path=None):
#         samples = self.sample(num_of_samples=1000000)
#         samples = (samples - self.lowEnd) / (self.highEnd - self.lowEnd)
#
#         y = np.arange(0.0, 1.0, 0.0001)
#         pdf_y = self.pdf(y=y)
#
#         fig, ax = plt.subplots(1, 1)
#         ax.set_title("{0}".format(self.name))
#         ax.plot(y, pdf_y, 'k-', lw=2, label='pdf')
#         ax.hist(samples, density=True, histtype='stepfilled', alpha=0.2, bins=100)
#         ax.legend(loc='best', frameon=False)
#         plt.tight_layout()
#
#         if show_plot:
#             plt.show()
#
#         if root_path is not None:
#             plt.savefig(os.path.join(root_path, "{0}.png".format(self.name)), dpi=1000)
#         plt.close()
#
#     # OK
#     def init_gmm(self, means, variances, weights,
#                  num_of_random_samples=100000):
#         assert len(means) == len(variances) and len(means) == len(weights)
#         self.numOfComponents = len(means)
#         self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
#                                                tol=1e-6, max_iter=10000)
#
#         # Generate synthetic samples from three normals.
#         data = []
#         component_probs = np.array(weights) / np.sum(weights)
#         component_selections = np.random.choice(
#             a=self.numOfComponents,
#             p=component_probs,
#             size=(num_of_random_samples,))
#         component_samples = []
#         for comp_id in range(self.numOfComponents):
#             mean = means[comp_id]
#             variance = variances[comp_id]
#             scale = np.sqrt(variance)
#             s = np.random.normal(loc=mean,
#                                  scale=scale,
#                                  size=(num_of_random_samples,))
#             component_samples.append(s)
#         component_samples = np.stack(component_samples, axis=1)
#
#         samples_selected = component_samples[np.arange(num_of_random_samples), component_selections]
#         self.gaussianMixture.fit(X=np.expand_dims(samples_selected, axis=-1))
#
#
# def plot_and_integrate():
#     sigmoid_gmm = SigmoidMixtureOfGaussians(num_of_components=3)
#     sigmoid_gmm.init_gmm(means=[-1.0, 1.0, 3.0], variances=[0.01, 0.02, 0.1],
#                          weights=[1.0, 1.0, 1.0])
#     # y = np.arange(1.e-30, 1.0 - 1.e-6, 0.0001)
#     x = np.arange(-100, 100, 0.0001)
#
#     # First draw Gaussian components and the GMM itself, then the Sigmoid-GMM
#     fig, ax = plt.subplots(sigmoid_gmm.numOfComponents + 2, 1)
#
#     component_densities = []
#     for comp_idx in range(sigmoid_gmm.numOfComponents):
#         component_mean = sigmoid_gmm.gaussianMixture.means_[comp_idx][0]
#         component_variance = sigmoid_gmm.gaussianMixture.covariances_[comp_idx, 0, 0]
#         component_scale = np.sqrt(component_variance)
#         pdf_x = norm.pdf(x=x, loc=component_mean, scale=component_scale)
#         ax[comp_idx].plot(x, pdf_x)
#         weighted_pdf_x = pdf_x * sigmoid_gmm.gaussianMixture.weights_[comp_idx]
#         component_densities.append(weighted_pdf_x)
#     component_densities = np.stack(component_densities, axis=1)
#     mixture_pdf = np.sum(component_densities, axis=1)
#     ax[-2].plot(x, mixture_pdf)
#
#     # Draw Sigmoid-GMM
#     y = np.arange(1.e-30, 1.0 - 1.e-6, 0.0001)
#     sigmoid_gmm_pdf = sigmoid_gmm.pdf(y=y)
#     ax[-1].plot(y, sigmoid_gmm_pdf)
#
#     plt.tight_layout()
#     plt.show()
#
#     sigmoid_gmm.plot_distribution()
#
#     res = integrate.quadrature(sigmoid_gmm.pdf, 0.0, 1.0, tol=0.0, maxiter=1000000)
#     assert np.isclose(res[0], 1.0)
#
#
# def check_mle():
#     low_end = 0.2
#     high_end = 0.21
#     sigmoid_gmm = SigmoidMixtureOfGaussians(num_of_components=3,
#                                             name="First Distribution",
#                                             low_end=low_end,
#                                             high_end=high_end)
#     sigmoid_gmm.init_gmm(means=[-1.0, 1.0, 3.0], variances=[0.01, 0.02, 0.1],
#                          weights=[1.0, 1.0, 1.0])
#     sigmoid_gmm.plot_distribution()
#     sigmoid_gmm_2 = SigmoidMixtureOfGaussians(num_of_components=3,
#                                               name="Second Distribution",
#                                               low_end=low_end,
#                                               high_end=high_end)
#
#     # Sample data from the first distribution for the ML estimation in the other.
#     samples = sigmoid_gmm.sample(num_of_samples=100000)
#     sigmoid_gmm_2.maximum_likelihood_estimate(data=samples, alpha=None)
#
#     for component_id in range(sigmoid_gmm_2.numOfComponents):
#         print("mu_{0}={1} sigma_{0}={2} weight_{0}={3}".format(
#             component_id,
#             sigmoid_gmm_2.gaussianMixture.means_[component_id, 0],
#             sigmoid_gmm_2.gaussianMixture.covariances_[component_id, 0, 0],
#             sigmoid_gmm_2.gaussianMixture.weights_[component_id]))
#
#     sigmoid_gmm_2.plot_distribution()
#
#
# if __name__ == "__main__":
#     gpus = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#
#     # plot_samples()
#     # plot_and_integrate()
#     check_mle()
