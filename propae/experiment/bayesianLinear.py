import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mean=0, prior_std=1):
        super(BayesianLinear, self).__init__()

        # Define the prior distribution for the weights and biases
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        # Initialize the weight and bias parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))

        # Initialize the KL divergence loss
        self.kl_divergence_loss = 0

    def forward(self, x):
        # Sample the weight and bias parameters from the posterior distribution using the reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std

        # Compute the output of the layer using the sampled weight and bias parameters
        output = F.linear(x, weight, bias)

        return output

    def sort_bias(self):
        # Sort the bias vector in descending order
        sorted_bias, sorted_indices = torch.sort(self.bias_mu, descending=True)
        self.bias_mu.data = sorted_bias

        # Update the weight matrix to match the sorted bias vector 
        self.bias_logvar.data = self.bias_logvar.data[sorted_indices]
        self.weight_mu.data = self.weight_mu.data[sorted_indices, :]
        self.weight_logvar.data = self.weight_logvar.data[sorted_indices, :]
        return sorted_indices
    

    def compute_kl_divergence_loss(self):
        # Compute the log likelihood of the weight and bias samples under the posterior distribution
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight_log_likelihood = torch.distributions.Normal(self.weight_mu, weight_std).log_prob(self.weight_mu).sum()
        bias_log_likelihood = torch.distributions.Normal(self.bias_mu, bias_std).log_prob(self.bias_mu).sum()

        # Compute the log likelihood of the weight and bias samples under the prior distribution
        weight_prior_log_likelihood = torch.distributions.Normal(self.prior_mean, self.prior_std).log_prob(self.weight_mu).sum()
        bias_prior_log_likelihood = torch.distributions.Normal(self.prior_mean, self.prior_std).log_prob(self.bias_mu).sum()

        # Compute the KL divergence between the posterior and prior distributions for the weight and bias parameters
        weight_kl_divergence = weight_log_likelihood - weight_prior_log_likelihood
        bias_kl_divergence = bias_log_likelihood - bias_prior_log_likelihood
        kl_divergence = weight_kl_divergence + bias_kl_divergence

        return kl_divergence
