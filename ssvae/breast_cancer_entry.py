from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Normal

from utils import Utils
from variational_autoencoder import VariationalAutoencoder
# import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import norm, multivariate_normal

from vector_dataset import VectorDataset
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    normalize_data = True
    input_dim = 30
    embedding_dim = 8
    hidden_layers_encoder = [512, 256, 128]
    hidden_layers_decoder = [128, 256, 512]

    bc_data = load_breast_cancer()

    # Normalize data
    if normalize_data:
        min_max_scaler = MinMaxScaler()
        X_ = min_max_scaler.fit_transform(X=bc_data.data)
    else:
        X_ = bc_data.data
    y_ = bc_data.target

    data_loader = torch.utils.data.DataLoader(VectorDataset(X_=X_, y_=y_),
                                              batch_size=X_.shape[0], shuffle=True,
                                              num_workers=2, pin_memory=True)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X=X_)
    # Visualize the 2D data
    # plt.plot(X_2d[:, 0], X_2d[:, 1], 'x')
    # plt.axis('equal')
    # plt.show()

    vae = VariationalAutoencoder(input_dim=input_dim,
                                 embedding_dim=embedding_dim,
                                 hidden_layers_encoder=hidden_layers_encoder,
                                 hidden_layers_decoder=hidden_layers_decoder,
                                 z_sample_count=1)
    vae.fit(dataset=data_loader, epoch_count=1000000, weight_decay=0.00005, checkpoint_period=5000)

    print("X")

