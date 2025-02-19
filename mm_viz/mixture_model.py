import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp
from functools import partial


def gibbs_weights(ALPHA_W: float, key: Array, assignments: Bool[Array, 'N l']) -> Float[Array, 'l']:
    counts = jnp.sum(assignments, axis=0)
    alpha = counts + ALPHA_W
    return log_dirichlet(key, alpha) # this is numerically unstable

def gibbs_params(ALPHA_P: float, key: Array, x: Bool[Array, 'N K c'], assignments: Bool[Array, 'N l']) -> Float[Array, 'l K c']:
    counts = jnp.einsum('NKc,Nl-> lKc', x, assignments)
    alpha = counts + ALPHA_P
    return log_dirichlet(key, alpha)

def gibbs_assignments(key: Array, weights: Float[Array, 'l'], x: Bool[Array, 'N K c'], params: Float[Array, 'l K c']) -> Bool[Array, 'N l']:
    l, k, _ = params.shape
    likelihood = jnp.einsum('NKc,lKc->Nl', x, params)
    posterior = likelihood + weights[None, :]
    logZ = jax.nn.logsumexp(posterior, axis=-1)
    posterior = posterior - logZ[:, None]
    c = jax.nn.one_hot(jax.random.categorical(key, posterior, axis=-1), l)
    return c

def log_dirichlet(key, alpha):
    y = jax.random.dirichlet(key, alpha)
    return jnp.log(y)

def gibbs(ALPHA_W: float, ALPHA_P: float, l: int, key: Array, x: Bool[Array, 'N K c'], num_steps=20):
    def gibbs_step(c: Bool[Array, 'N l'], key_i: Array) -> tuple[Bool[Array, 'N l'], tuple[Float[Array, 'l'], Float[Array, 'l K c'], Bool[Array, 'N l']]]:
        key1, key2, key3 = jax.random.split(key_i, 3)
        weights_step = gibbs_weights(ALPHA_W, key1, c)
        params_step = gibbs_params(ALPHA_P, key2, x, c)
        assignments_step = gibbs_assignments(key3, weights_step, x, params_step)
        return assignments_step, (weights_step, params_step, assignments_step, score(ALPHA_W, ALPHA_P, weights_step, params_step, assignments_step, x))
    
    N, K, _ = x.shape
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_steps)
    assignments_init = jax.nn.one_hot(jax.random.categorical(key, jnp.log(0.5) * jnp.ones((N, l)), axis=-1), l)
    _, (weights, params, assignments, s) = jax.lax.scan(gibbs_step, assignments_init, keys)
    return weights, params, assignments, s

def score_weights(ALPHA_W: float, weights: Float[Array, 'l']):
    return jax.scipy.stats.dirichlet.logpdf(jnp.exp(weights), ALPHA_W * jnp.ones_like(weights))

def score_params(ALPHA_P: float, params: Float[Array, 'l K c']):
    c = params.shape[-1]
    likelihoods = jax.vmap(jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(0, None)), in_axes=(0, None))(jnp.exp(params), ALPHA_P * jnp.ones(c))
    return jnp.sum(likelihoods)

def score_assignments(assignments: Bool[Array, 'N l'], weights: Float[Array, 'l']):
    return jnp.einsum('Nl,l->', assignments, weights)

def score_x(x: Bool[Array, 'N K c'], assignments: Bool[Array, 'N l'], params: Float[Array, 'l K c']):
    return jnp.einsum('NKc,Nl,lKC->', x, assignments, params)

def score(ALPHA_W: float, ALPHA_P: float, weights, params, assignments, x):
    return score_weights(ALPHA_W, weights) + score_params(ALPHA_P, params) + score_assignments(assignments, weights) + score_x(x, assignments, params)

def forward_sample(key, ALPHA_W, ALPHA_P, l, K, categories, N):
    key, subkey = jax.random.split(key)
    weights = jnp.log(jax.random.dirichlet(subkey, ALPHA_W * jnp.ones(l)))

    key, subkey = jax.random.split(key)
    params = jnp.log(jax.random.dirichlet(subkey, ALPHA_P * jnp.ones((l, K, categories))))

    key, subkey = jax.random.split(key)
    assignments = jax.random.gumbel(subkey, shape=(N, l)) + weights[None, :]
    assignments = jnp.max(assignments, axis=-1)[..., None] == assignments

    key, subkey = jax.random.split(key)
    x_logits = jnp.einsum('Nl,lKC->NKC', assignments, params)
    x = jax.random.gumbel(subkey, shape=(N, K, categories)) + x_logits
    x = jnp.max(x, axis=-1)[..., None] == x

    return weights, params, assignments, x

def sbc(l, K, categories, N, ALPHA_W, ALPHA_P, key):
    weights_gt, params_gt, assignments_gt, x = forward_sample(key, ALPHA_W, ALPHA_P, l, K, categories, N)
    weights, params, assignments, s = gibbs(ALPHA_W, ALPHA_P, l, key, x, num_steps=100)
    return weights_gt, params_gt, assignments_gt, x, weights, params, assignments, s