import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


class SigmoidMixtureOfGaussians:
    def __init__(self, num_of_components, low_end=0.0, high_end=1.0, name=""):
        self.name = name
        self.numOfComponents = num_of_components
        self.lowEnd = low_end
        self.highEnd = high_end
        self.isFitted = False
        self.gaussianMixture = GaussianMixture(n_components=self.numOfComponents,
                                               tol=1e-6, max_iter=10000)
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


#     def forward(self):
#         # ensure correct domain for params
#         beta_params_norm = torch.nn.functional.softplus(self.componentParameters)
#         mixture_prob_norm = torch.nn.functional.softmax(self.mixtureParameters, dim=0)
#
#         mix = torch.distributions.Categorical(mixture_prob_norm)
#         comp = torch.distributions.Beta(beta_params_norm[:, 0], beta_params_norm[:, 1])
#         mixture_dist = torch.distributions.MixtureSameFamily(mix, comp)
#
#         return mixture_dist
#
#     def sample(self, num_of_samples):
#         mixture_dist = self()
#         samples = mixture_dist.sample(sample_shape=(num_of_samples,))
#         return samples
#
#     def draw_pdf(self):
#         beta_params_norm = torch.nn.functional.softplus(self.componentParameters)
#         mixture_prob_norm = torch.nn.functional.softmax(self.mixtureParameters, dim=0)
#
#         min_x_list = []
#         max_x_list = []
#         min_x = 0.0001
#         max_x = 0.9999
#         x = np.linspace(min_x, max_x, 1000)
#         pdf_x = []
#         for k in range(self.componentCount):
#             fig, ax = plt.subplots(1, 1)
#             a_k = beta_params_norm.detach().numpy()[k, 0]
#             b_k = beta_params_norm.detach().numpy()[k, 1]
#             pi_k = mixture_prob_norm.detach().numpy()[k]
#             pdf_k = beta.pdf(x, a_k, b_k)
#             ax.set_title("Beta Component {0} a{0}={1} b{0}={2} pi{0}={3}".format(
#                 k, "{:.3f}".format(a_k), "{:.3f}".format(b_k), "{:.3f}".format(pi_k)))
#             ax.plot(x, pdf_k, 'r-', lw=1, alpha=1.0, label='beta mixture pdf')
#             pdf_x.append(pi_k * pdf_k)
#             fig.tight_layout()
#             plt.show()
#         fig, ax = plt.subplots(1, 1)
#         mixture_pdf = np.stack(pdf_x, axis=0)
#         mixture_pdf = np.sum(mixture_pdf, axis=0)
#         ax.set_title("Mixture")
#         ax.plot(x, mixture_pdf, 'r-', lw=1, alpha=1.0, label='beta mixture pdf')
#
#         # Sample from the current distribution
#         samples = self.sample(num_of_samples=100000).numpy()
#         ax.hist(samples, density=True, histtype="stepfilled", alpha=0.2, bins=1000)
#         ax.legend(loc="best", frameon=False)
#         fig.tight_layout()
#         plt.show()
#
#     # MLE Fit
#     def fit(self, data, iteration_count):
#         optimizer = torch.optim.Adam(
#             self.parameters(),
#             lr=self.learningRate)
#
#         for idx in range(iteration_count):
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 mixture_dist = self.forward()
#                 negloglik = -mixture_dist.log_prob(data)
#                 loss = torch.mean(negloglik)
#                 loss.backward()
#                 optimizer.step()
#                 self.trainLogError.append(loss.detach().numpy())
#                 if idx % 100 == 0:
#                     print("Iteration:{0} Loss:{1}".format(idx,
#                                                           np.mean(self.trainLogError[-25:])))
#
#         print("X")
#
#
# if __name__ == "__main__":
#     beta_mixture = BetaMixture(component_count=3, learning_rate=1e-3)
#     beta_mixture.draw_pdf()
#
#     beta_mixture_trained = BetaMixture(component_count=3, learning_rate=1e-3)
#     beta_mixture_trained.mixtureParameters = torch.nn.Parameter(torch.from_numpy(np.array([1.0, 2.0, 2.5])))
#     beta_mixture_trained.componentParameters = torch.nn.Parameter(
#         torch.from_numpy(
#             np.array([[2.0, 3.0], [0.5, 1.5], [4.5, 0.75]])))
#
#     beta_mixture_trained.draw_pdf()
#     samples = beta_mixture_trained.sample(num_of_samples=10000)
#
#     beta_mixture.fit(data=samples, iteration_count=100000)
#
#     print("X")
