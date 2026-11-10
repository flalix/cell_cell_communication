# %% [markdown]
# # NumPyro: one worked example per key feature
#
# 1. JAX-powered speed          -> 2,000 per-gene NB fits with jit + vmap + grad
# 2. NUTS & HMC                 -> hierarchical tumour-vs-normal differential expression
# 3. Variational inference      -> bulk deconvolution, mini-batched over genes
# 4. Composable effect handlers -> trace / condition / replay / block on one model
# 5. Rich distribution library  -> mixture, transformed and a custom distribution
# 6. Seamless ecosystem         -> Pandas in, Flax + Optax inside, ArviZ out
#
# All data are SYNTHETIC (PDAC-flavoured). Each section marks where to plug in real data.
#
# pip install numpyro optax flax arviz pandas

# %%
import time

import arviz as az
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
from jax import grad, jit, lax, random, vmap
from jax.scipy.special import gammaln
from numpyro import handlers
from numpyro.contrib.module import flax_module
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

numpyro.set_host_device_count(2)  # lets MCMC run 2 chains in parallel on CPU
rng = np.random.default_rng(0)
print("numpyro", numpyro.__version__, "| jax", jax.__version__, "| devices", jax.devices())


def simulate_nb(mu, phi):
    """NB counts with mean mu and inverse-dispersion phi (var = mu + mu^2/phi)."""
    return rng.negative_binomial(n=phi, p=phi / (phi + mu))


# %% [markdown]
# ## 1. JAX-powered speed
# Fit a negative-binomial (mean, dispersion) to every gene by maximum likelihood.
# `grad` gives the gradient, `vmap` turns the one-gene fit into an all-genes fit,
# `jit` compiles the whole optimisation loop with XLA.
#
# Real data: replace `counts_gn` with a (genes x samples) raw count matrix.

# %%
G1, N1 = 2000, 60
mu_true = rng.lognormal(4.0, 1.2, G1)
phi_true = rng.gamma(4.0, 0.5, G1)
counts_gn = jnp.asarray(simulate_nb(np.tile(mu_true[:, None], (1, N1)),
                                    np.tile(phi_true[:, None], (1, N1))))


def nb_nll(params, y):
    log_mu, log_phi = params
    return -dist.NegativeBinomial2(jnp.exp(log_mu), jnp.exp(log_phi)).log_prob(y).sum()


def fit_one_gene(y, steps=300):
    params = jnp.array([jnp.log(y.mean() + 1.0), 0.0])
    opt = optax.adam(0.05)

    def step(carry, _):
        p, state = carry
        updates, state = opt.update(grad(nb_nll)(p, y), state)
        return (optax.apply_updates(p, updates), state), None

    (params, _), _ = lax.scan(step, (params, opt.init(params)), None, length=steps)
    return params


fit_all_genes = jit(vmap(fit_one_gene))  # one gene -> all genes, compiled

t0 = time.perf_counter()
est = fit_all_genes(counts_gn).block_until_ready()
t_compile = time.perf_counter() - t0
t0 = time.perf_counter()
est = fit_all_genes(counts_gn).block_until_ready()
t_run = time.perf_counter() - t0

print(f"\n[1] {G1} genes x 300 Adam steps: {t_compile:.2f}s incl. compile, {t_run:.2f}s compiled")
print(f"    corr(log mu)  = {np.corrcoef(est[:, 0], np.log(mu_true))[0, 1]:.3f}")
print(f"    corr(log phi) = {np.corrcoef(est[:, 1], np.log(phi_true))[0, 1]:.3f}")

# %% [markdown]
# ## 2. NUTS & HMC
# Hierarchical differential expression: every gene has its own log fold change,
# but they share a prior scale `tau` (partial pooling), so noisy genes are shrunk.
#
# Real data: `counts` = (samples x genes) raw counts, `tumor` = 0/1 per sample,
# `log_sf` = log size factors.

# %%
G2, n_per_group = 40, 8
tumor = jnp.array([0] * n_per_group + [1] * n_per_group)
log_sf = rng.normal(0.0, 0.3, 2 * n_per_group)
base_true = rng.normal(5.0, 1.0, G2)
lfc_true = np.where(rng.random(G2) < 0.3, rng.normal(0.0, 1.5, G2), rng.normal(0.0, 0.1, G2))
mu2 = np.exp(log_sf[:, None] + base_true[None, :] + np.asarray(tumor)[:, None] * lfc_true[None, :])
counts_de = jnp.asarray(simulate_nb(mu2, 8.0))


def de_model(tumor, log_sf, counts=None, n_genes=None):
    n_genes = counts.shape[1] if counts is not None else n_genes
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))
    with numpyro.plate("genes", n_genes, dim=-1):
        base = numpyro.sample("base", dist.Normal(5.0, 3.0))
        lfc = numpyro.sample("lfc", dist.Normal(0.0, tau))
        phi = numpyro.sample("phi", dist.LogNormal(2.0, 1.0))
        log_mu = log_sf[:, None] + base + tumor[:, None] * lfc
        with numpyro.plate("samples", tumor.shape[0], dim=-2):
            numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_mu), phi), obs=counts)


mcmc_de = MCMC(NUTS(de_model, target_accept_prob=0.9), num_warmup=500, num_samples=500,
               num_chains=2, progress_bar=False)
t0 = time.perf_counter()
mcmc_de.run(random.PRNGKey(1), tumor, jnp.asarray(log_sf), counts_de)
t_nuts = time.perf_counter() - t0

post = mcmc_de.get_samples()
lfc_mean = post["lfc"].mean(0)
lo, hi = np.percentile(post["lfc"], [5, 95], axis=0)
print(f"\n[2] NUTS, 2 chains x (500 warmup + 500 draws), {3 * G2 + 1} parameters: {t_nuts:.1f}s")
print(f"    divergences = {int(mcmc_de.get_extra_fields()['diverging'].sum())}")
print(f"    corr(posterior mean LFC, true LFC) = {np.corrcoef(lfc_mean, lfc_true)[0, 1]:.3f}")
print(f"    90% interval coverage of true LFC  = {np.mean((lfc_true >= lo) & (lfc_true <= hi)):.2f}")
print(f"    tau (shared LFC scale) = {post['tau'].mean():.2f}")

# %% [markdown]
# ## 3. Variational inference with mini-batching
# Bulk deconvolution: each bulk sample is a mixture of cell-type reference profiles.
# `plate(..., subsample_size=...)` draws a random gene mini-batch at every step and
# rescales the likelihood, so the cost per step is independent of the number of genes.
#
# Real data: `bulk` = (samples x genes) counts, `ref` = (cell types x genes)
# reference profiles with rows summing to 1 (e.g. from an scRNA-seq atlas).

# %%
cell_types = ["malignant", "CAF", "T cell", "myeloid", "endothelial", "acinar"]
S3, G3, K3 = 24, 5000, len(cell_types)
ref = rng.lognormal(0.0, 1.5, (K3, G3))
ref /= ref.sum(1, keepdims=True)
theta_true = rng.dirichlet(np.ones(K3) * 1.5, S3)
depth = rng.uniform(1e7, 3e7, S3)
bulk = jnp.asarray(simulate_nb(depth[:, None] * (theta_true @ ref), 20.0))
ref_j, depth_j = jnp.asarray(ref), jnp.asarray(depth)


def deconv_model(bulk, ref, depth, batch_size=512):
    n_samples, n_genes = bulk.shape
    n_types = ref.shape[0]
    theta = numpyro.sample("theta", dist.Dirichlet(jnp.ones(n_types)).expand([n_samples]).to_event(1))
    phi = numpyro.sample("phi", dist.LogNormal(3.0, 1.0))
    with numpyro.plate("genes", n_genes, subsample_size=batch_size, dim=-1) as idx:
        mu = depth[:, None] * (theta @ ref[:, idx])
        with numpyro.plate("samples", n_samples, dim=-2):
            numpyro.sample("y", dist.NegativeBinomial2(mu, phi), obs=bulk[:, idx])


guide = AutoNormal(deconv_model)
svi = SVI(deconv_model, guide, optax.adam(optax.exponential_decay(0.05, 1000, 0.3)), Trace_ELBO())
t0 = time.perf_counter()
svi_result = svi.run(random.PRNGKey(2), 3000, bulk, ref_j, depth_j, progress_bar=False)
t_svi = time.perf_counter() - t0

theta_hat = guide.median(svi_result.params)["theta"]
print(f"\n[3] SVI, 3000 steps, 512 of {G3} genes per step: {t_svi:.1f}s")
print(f"    ELBO loss: {svi_result.losses[:50].mean():.3e} -> {svi_result.losses[-50:].mean():.3e}")
print(f"    mean |theta_hat - theta_true| = {np.abs(theta_hat - theta_true).mean():.4f}")
print(pd.DataFrame({"true": theta_true[0], "estimated": np.asarray(theta_hat[0])},
                   index=cell_types).round(3).to_string())

# %% [markdown]
# ## 4. Composable effect handlers
# A model is a plain function; handlers wrap it to change what `numpyro.sample` does.
# The same model becomes a simulator, a density evaluator, a counterfactual
# generator or a constrained model, without being rewritten.

# %%
def expr_model(tumor, y=None):
    base = numpyro.sample("base", dist.Normal(5.0, 2.0))
    lfc = numpyro.sample("lfc", dist.Normal(0.0, 1.0))
    phi = numpyro.sample("phi", dist.LogNormal(2.0, 1.0))
    with numpyro.plate("samples", tumor.shape[0]):
        return numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(base + lfc * tumor), phi), obs=y)


design = jnp.array([0.0] * 6 + [1.0] * 6)
print("\n[4] effect handlers")

# trace: record every sample site of one execution (seed supplies the PRNG key)
tr = handlers.trace(handlers.seed(expr_model, random.PRNGKey(3))).get_trace(design)
for name, site in tr.items():
    print(f"    trace   {name:8s} {site['type']:6s} shape={jnp.shape(site['value'])}")
log_joint = sum(site["fn"].log_prob(site["value"]).sum() for site in tr.values() if site["type"] == "sample")
print(f"    log joint of that execution = {log_joint:.2f}")

# condition: clamp latents -> the model becomes a simulator for a known scenario
simulator = handlers.condition(expr_model, {"base": 5.0, "lfc": 2.0, "phi": 10.0})
y_sim = handlers.seed(simulator, random.PRNGKey(4))(design)
print(f"    condition  lfc=2 -> mean normal {y_sim[:6].mean():.0f}, mean tumour {y_sim[6:].mean():.0f}")

# replay: rerun with the SAME latent draws but a different design (counterfactual)
latents = {k: v for k, v in tr.items() if k != "y"}
y_cf = handlers.seed(handlers.replay(expr_model, latents), random.PRNGKey(5))(jnp.ones(12))
print(f"    replay     same latents (lfc={tr['lfc']['value']:.2f}), all samples tumour -> mean {y_cf.mean():.0f}")

# block: hide sites from whatever handler sits outside
visible = handlers.trace(handlers.block(handlers.seed(expr_model, random.PRNGKey(3)), hide=["phi"])).get_trace(design)
print(f"    block      sites visible outside: {list(visible)}")

# handlers compose with inference: fix the dispersion, infer the rest
mcmc_h = MCMC(NUTS(handlers.condition(expr_model, {"phi": 10.0})), num_warmup=300, num_samples=300,
              progress_bar=False)
mcmc_h.run(random.PRNGKey(6), design, y=y_sim)
print(f"    condition+NUTS  lfc = {mcmc_h.get_samples()['lfc'].mean():.2f} (simulated with 2.0)")

# %% [markdown]
# ## 5. Rich distribution library
# (a) Mixture + Transformed: a gene that is "off" in some cells and "on" in others.
#     The cell label is marginalised by `MixtureSameFamily`; `OrderedTransform`
#     forces loc[0] < loc[1], which removes label switching.
# (b) Custom: a zero-truncated Poisson for UMI counts in cells where a gene was detected.

# %%
is_on = rng.random(400) < 0.35
log_expr = jnp.asarray(np.where(is_on, rng.normal(3.0, 0.6, 400), rng.normal(0.5, 0.4, 400)))


def bimodal_model(x):
    w_on = numpyro.sample("w_on", dist.Beta(2.0, 2.0))
    loc = numpyro.sample("loc", dist.TransformedDistribution(
        dist.Normal(0.0, 5.0).expand([2]), dist.transforms.OrderedTransform()))
    scale = numpyro.sample("scale", dist.HalfNormal(2.0).expand([2]).to_event(1))
    mixture = dist.MixtureSameFamily(dist.Categorical(probs=jnp.stack([1 - w_on, w_on])),
                                     dist.Normal(loc, scale))
    with numpyro.plate("cells", x.shape[0]):
        numpyro.sample("x", mixture, obs=x)


mcmc_mix = MCMC(NUTS(bimodal_model), num_warmup=400, num_samples=400, progress_bar=False)
mcmc_mix.run(random.PRNGKey(7), log_expr)
s = mcmc_mix.get_samples()
print("\n[5] distributions")
print(f"    mixture  w_on = {s['w_on'].mean():.2f} (true 0.35), loc = {np.round(s['loc'].mean(0), 2)} (true [0.5, 3.0])")


class ZeroTruncatedPoisson(dist.Distribution):
    """Poisson conditioned on value >= 1."""

    arg_constraints = {"rate": constraints.positive}
    support = constraints.positive_integer

    def __init__(self, rate, *, validate_args=None):
        self.rate = rate
        super().__init__(batch_shape=jnp.shape(rate), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        key_u, key_p = random.split(key)
        # time of the first event, truncated to fall inside the interval, then the rest
        t = -jnp.log1p(random.uniform(key_u, shape) * jnp.expm1(-self.rate))
        return 1 + random.poisson(key_p, self.rate - t, shape)

    def log_prob(self, value):
        log_pois = value * jnp.log(self.rate) - self.rate - gammaln(value + 1.0)
        return log_pois - jnp.log(-jnp.expm1(-self.rate))


umis = ZeroTruncatedPoisson(jnp.array(1.8)).sample(random.PRNGKey(8), (500,))


def ztp_model(y):
    rate = numpyro.sample("rate", dist.Gamma(2.0, 1.0))
    with numpyro.plate("cells", y.shape[0]):
        numpyro.sample("y", ZeroTruncatedPoisson(rate), obs=y)


mcmc_ztp = MCMC(NUTS(ztp_model), num_warmup=300, num_samples=300, progress_bar=False)
mcmc_ztp.run(random.PRNGKey(9), umis)
print(f"    custom   min count = {int(umis.min())}, naive mean = {umis.mean():.2f}, "
      f"posterior rate = {mcmc_ztp.get_samples()['rate'].mean():.2f} (true 1.8)")

# %% [markdown]
# ## 6. Seamless ecosystem
# Pandas holds the data, a Flax MLP is the mean function inside a NumPyro model,
# Optax trains it, ArviZ diagnoses the NUTS run from section 2.
#
# Real data: a long-format table with one row per well (dose, subtype, viability).

# %%
n_wells = 600
df = pd.DataFrame({
    "log_dose": rng.uniform(-3, 3, n_wells),
    "subtype": rng.choice(["classical", "basal-like"], n_wells),
})
ec50 = np.where(df["subtype"] == "basal-like", 1.0, -0.5)  # basal-like is more resistant


def hill(log_dose, ec50):
    return 1.0 / (1.0 + np.exp(1.8 * (log_dose - ec50)))


df["viability"] = hill(df["log_dose"], ec50) + rng.normal(0, 0.05, n_wells)

X = jnp.asarray(np.column_stack([df["log_dose"], df["subtype"] == "basal-like"]), dtype=jnp.float32)
y6 = jnp.asarray(df["viability"].to_numpy(), dtype=jnp.float32)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(16)(x))
        x = nn.tanh(nn.Dense(16)(x))
        return nn.Dense(1)(x).squeeze(-1)


def dose_response_model(X, y=None):
    net = flax_module("mlp", MLP(), input_shape=(1, X.shape[1]))  # weights = numpyro.param
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))          # noise stays Bayesian
    with numpyro.plate("wells", X.shape[0]):
        numpyro.sample("y", dist.Normal(net(X), sigma), obs=y)


guide6 = AutoNormal(dose_response_model)
svi6 = SVI(dose_response_model, guide6, optax.adamw(3e-3, weight_decay=1e-4), Trace_ELBO())
res6 = svi6.run(random.PRNGKey(10), 4000, X, y6, progress_bar=False)

grid = np.linspace(-3, 3, 50)
X_new = jnp.asarray(np.column_stack([np.tile(grid, 2), np.repeat([0, 1], 50)]), dtype=jnp.float32)
pred = Predictive(dose_response_model, guide=guide6, params=res6.params, num_samples=200)(
    random.PRNGKey(11), X_new)["y"].mean(0)
truth = hill(np.tile(grid, 2), np.repeat([-0.5, 1.0], 50))
print("\n[6] ecosystem")
print(f"    Flax+Optax  RMSE of learned dose-response vs true Hill curves = "
      f"{np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)):.3f}")
print(f"    noise sigma = {guide6.median(res6.params)['sigma']:.3f} (true 0.05)")

idata = az.from_numpyro(mcmc_de)                       # NUTS run from section 2 -> ArviZ
summary = az.summary(idata, var_names=["tau", "lfc"])  # a pandas DataFrame
print("    ArviZ summary of the section-2 model (first rows):")
print(summary.head(6).to_string())
