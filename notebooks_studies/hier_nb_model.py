"""
Hierarchical Gamma-Poisson (negative binomial) model for a cell x gene count matrix.

Generative hierarchy
--------------------
  hyper:   a0, a1 > 0           dispersion trend  phi_trend(q) = a0 + a1 / q   (DESeq2-style)
           sigma_phi > 0        spread of gene dispersions around the trend
           m_q, sigma_q         population distribution of gene mean expression
  gene j:  log q_j   ~ Normal(m_q, sigma_q)
           log phi_j ~ Normal(log phi_trend(q_j), sigma_phi)
  cell i:  s_i       size factor (treated as known, e.g. library size / median)
  count:   lambda_ij ~ Gamma(shape = 1/phi_j, scale = phi_j * s_i * q_j)    # biological rate
           y_ij      ~ Poisson(lambda_ij)
           => marginally y_ij ~ NB(mean = s_i q_j, dispersion phi_j)       # lambda integrated out

The fitted model uses the marginal NB likelihood (lambda integrated analytically),
which is much easier for NUTS than sampling GENES*CELLS latent rates.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import numpy as np
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS
from scipy import special # optimize
# import arviz as az

# ------------------------------------------------------------------ no pooling: per-gene MLE
def nb_negll(theta, y, m):
    log_q, log_phi = theta

    mu_ = m * np.exp(log_q)
    r = np.exp(-log_phi)

    ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
          + r * np.log(r / (r + mu_)) + y * np.log(mu_ / (r + mu_)))
    return -ll.sum()

# ------------------------------------------------------------------ hierarchical model
def model(s, rand_poi=None, GENES=None):
    a0        = numpyro.sample("a0",        dist.HalfNormal(1.0))
    a1        = numpyro.sample("a1",        dist.HalfNormal(1.0))
    sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(1.0))
    m_q       = numpyro.sample("m_q",       dist.Normal(0.0, 2.0))
    sigma_q   = numpyro.sample("sigma_q",   dist.HalfNormal(2.0))

    with numpyro.plate("genes", GENES):
        # gene means are strongly identified by the data -> centred parameterisation
        log_q = numpyro.sample("log_q", dist.Normal(m_q, sigma_q))
        # dispersions are weakly identified (few cells, low counts) -> non-centred
        z_phi = numpyro.sample("z_phi", dist.Normal(0, 1))
        q = jnp.exp(log_q)
        log_phi = numpyro.deterministic("log_phi", jnp.log(a0 + a1 / q) + sigma_phi * z_phi)

    mu = s[:, None] * q[None, :]

    numpyro.sample("rand_poi", dist.NegativeBinomial2(mean=mu, concentration=jnp.exp(-log_phi)), obs=rand_poi)
