import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


class BetaMixture(nn.Module):
    def __init__(self, component_count, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.componentCount = component_count
        self.mixtureParameters = torch.nn.Parameter(torch.randn((self.componentCount,)))
        self.componentParameters = torch.nn.Parameter(torch.randn((self.componentCount, 2)))
        self.trainLogError = []
        self.valLogError = []
        self.learningRate = learning_rate

    def forward(self):
        # ensure correct domain for params
        beta_params_norm = torch.nn.functional.softplus(self.componentParameters)
        mixture_prob_norm = torch.nn.functional.softmax(self.mixtureParameters, dim=0)

        mix = torch.distributions.Categorical(mixture_prob_norm)
        comp = torch.distributions.Beta(beta_params_norm[:, 0], beta_params_norm[:, 1])
        mixture_dist = torch.distributions.MixtureSameFamily(mix, comp)

        return mixture_dist

    # def eval_pdf_manually(self):

    def draw_pdf(self):
        beta_params_norm = torch.nn.functional.softplus(self.componentParameters)
        mixture_prob_norm = torch.nn.functional.softmax(self.mixtureParameters, dim=0)

        min_x_list = []
        max_x_list = []
        min_x = 0.001
        max_x = 0.999
        x = np.linspace(min_x, max_x, 1000)
        pdf_x = []
        for k in range(self.componentCount):
            fig, ax = plt.subplots(1, 1)
            a_k = beta_params_norm.detach().numpy()[k, 0]
            b_k = beta_params_norm.detach().numpy()[k, 1]
            pi_k = mixture_prob_norm.detach().numpy()[k]
            pdf_k = beta.pdf(x, a_k, b_k)
            ax.set_title("Beta Component {0} a{0}={1} b{0}={2} pi{0}={3}".format(
                k, "{:.3f}".format(a_k), "{:.3f}".format(b_k), "{:.3f}".format(pi_k)))
            ax.plot(x, pdf_k, 'r-', lw=1, alpha=1.0, label='beta mixture pdf')
            pdf_x.append(pi_k * pdf_k)
            fig.tight_layout()
            plt.show()
        fig, ax = plt.subplots(1, 1)
        mixture_pdf = np.stack(pdf_x, axis=0)
        mixture_pdf = np.sum(mixture_pdf, axis=0)
        ax.set_title("Mixture")
        ax.plot(x, mixture_pdf, 'r-', lw=1, alpha=1.0, label='beta mixture pdf')
        fig.tight_layout()
        plt.show()

        # for k in range(self.componentCount):
        #     a = beta_params_norm.numpy()[k, 0]
        #     b = beta_params_norm.numpy()[k, 1]
        #     min_x = beta.ppf(0.001, a, b)
        #     max_x = beta.ppf(0.999, a, b)
        #     min_x_list.append(min_x)
        #     max_x_list.append(max_x)
        #
        # min_x_whole = max(min_x_list)
        # max_x_whole = min(max_x_list)


if __name__ == "__main__":
    beta_mixture = BetaMixture(component_count=3, learning_rate=1e-3)
    beta_mixture.draw_pdf()
    beta_mixture()
    print("X")
