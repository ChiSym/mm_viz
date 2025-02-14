import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp


def gibbs_pi(ALPHA_PI: float, key: Array, c: Bool[Array, 'N l']) -> Float[Array, 'l']:
    counts = jnp.sum(c, axis=0)
    alpha = counts + ALPHA_PI
    return log_dirichlet(key, alpha) # this is numerically unstable

def gibbs_c(key: Array, pi: Float[Array, 'l'], x: Bool[Array, 'N K c'], w: Float[Array, 'l K c']) -> Bool[Array, 'N l']:
    l, k, _ = w.shape
    likelihood = jnp.einsum('NKc,lKc->Nl', x, w)
    posterior = likelihood + pi[None, :]
    logZ = jax.nn.logsumexp(posterior, axis=-1)
    posterior = posterior - logZ[:, None]
    c = jax.nn.one_hot(jax.random.categorical(key, posterior, axis=-1), l)
    return c

def gibbs_w(ALPHA_W: float, key: Array, x: Bool[Array, 'N K c'], c: Bool[Array, 'N l']) -> Float[Array, 'l K c']:
    counts = jnp.einsum('NKc,Nl-> lKc', x, c)
    alpha = counts + ALPHA_W
    return log_dirichlet(key, alpha)

def log_dirichlet(key, alpha):
    y = jax.random.dirichlet(key, alpha)
    return jnp.log(y)

def gibbs(ALPHA_PI: float, ALPHA_W: float, l: int, key: Array, x: Bool[Array, 'N K c'], num_steps=20):
    def gibbs_step(c: Bool[Array, 'N l'], key_i: Array) -> tuple[Bool[Array, 'N l'], tuple[Float[Array, 'l'], Float[Array, 'l K c'], Bool[Array, 'N l']]]:
        key1, key2, key3 = jax.random.split(key_i, 3)
        pi_step = gibbs_pi(ALPHA_PI, key1, c)
        w_step = gibbs_w(ALPHA_W, key2, x, c)
        c_step = gibbs_c(key3, pi_step, x, w_step)
        return c_step, (pi_step, w_step, c_step, score(ALPHA_PI, ALPHA_W, pi_step, w_step, c_step, x))
    
    N, K, _ = x.shape
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_steps)
    c_init = jax.nn.one_hot(jax.random.categorical(key, jnp.log(0.5) * jnp.ones((N, l)), axis=-1), l)
    _, (pi, w, c, s) = jax.lax.scan(gibbs_step, c_init, keys)
    return pi, w, c, s

def score_pi(ALPHA_PI: float, pi):
    return jax.scipy.stats.dirichlet.logpdf(jnp.exp(pi), ALPHA_PI * jnp.ones_like(pi))

def score_w(ALPHA_W: float, w: Float[Array, 'l K c']):
    c = w.shape[-1]
    likelihoods = jax.vmap(jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(0, None)), in_axes=(0, None))(jnp.exp(w), ALPHA_W * jnp.ones(c))
    return jnp.sum(likelihoods)

def score_c(c: Bool[Array, 'N l'], pi: Float[Array, 'l']):
    return jnp.einsum('Nl,l->', c, pi)

def score_x(x: Bool[Array, 'N K c'], c: Bool[Array, 'N l'], w: Float[Array, 'l K c']):
    return jnp.einsum('NKc,Nl,lKC->', x, c, w)

def score(ALPHA_PI: float, ALPHA_W: float, pi, w, c, x):
    return score_pi(ALPHA_PI, pi) + score_w(ALPHA_W, w) + score_c(c, pi) + score_x(x, c, w)