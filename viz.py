# %%
%load_ext autoreload
%autoreload 2


# %%
import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp

ALPHA_PI = 1.0
ALPHA_W = 1.0
l = 2 # num clusters
categories = 2 # num categories

def gibbs_pi(key: Array, c: Bool[Array, 'N l']) -> Float[Array, 'l']:
    counts = jnp.sum(c, axis=0)
    alpha = counts + ALPHA_PI
    return log_dirichlet(key, alpha) # this is numerically unstable

def gibbs_c(key: Array, pi: Float[Array, 'l'], x: Bool[Array, 'N K c'], w: Float[Array, 'l K c']) -> Bool[Array, 'N l']:
    k = x.shape[1] # number of columns
    likelihood = jnp.einsum('NKc,lKc->Nl', x, w)
    posterior = likelihood + pi[None, :]
    logZ = jax.nn.logsumexp(posterior, axis=-1)
    posterior = posterior - logZ[:, None]
    c = jax.nn.one_hot(jax.random.categorical(key, posterior, axis=-1), l)
    return c

def gibbs_w(key: Array, x: Bool[Array, 'N K c'], c: Bool[Array, 'N l']) -> Float[Array, 'l K c']:
    counts = jnp.einsum('NKc,Nl-> lKc', x, c)
    alpha = counts + ALPHA_W
    return log_dirichlet(key, alpha)

def log_dirichlet(key, alpha):
    y = jax.random.dirichlet(key, alpha)
    return jnp.log(y)

def gibbs(key, x, num_steps=20):
    def gibbs_step(c: Bool[Array, 'N l'], key_i: Array) -> tuple[Bool[Array, 'N l'], tuple[Float[Array, 'l'], Float[Array, 'l K c'], Bool[Array, 'N l']]]:
        key1, key2, key3 = jax.random.split(key_i, 3)
        pi = gibbs_pi(key1, c)
        w = gibbs_w(key2, x, c)
        c = gibbs_c(key3, pi, x, w)
        return c, (pi, w, c, score(pi, w, c, x))
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_steps)
    c_init = jax.nn.one_hot(jax.random.categorical(key, jnp.log(0.5) * jnp.ones((N, l)), axis=-1), l)
    _, (pi, w, c, s) = jax.lax.scan(gibbs_step, c_init, keys)
    return pi, w, c, s

N = 4 
K = 5
k = jax.random.key(123)
x = jax.nn.one_hot(jax.random.bernoulli(k, p=.5, shape=(N, K)).astype(jnp.int32), categories, axis=-1) 

# %%
k, key = jax.random.split(k)
pi, w, c = gibbs(key, x)

# %%
jnp.exp(pi)

# %%
jnp.exp(w)

# %%
import jax
import jax.numpy as jnp
key = jax.random.key(1234)
x = jax.random.bernoulli(key, p=.5, shape=(4, 5))
# %%
jnp.einsum('nclusters naaa -> nclusters', x)
# %%

def score_pi(pi):
    return jax.scipy.stats.dirichlet.logpdf(jnp.exp(pi), ALPHA_PI * jnp.ones_like(pi))

def score_w(w: Float[Array, 'l K c']):
    c = w.shape[-1]
    likelihoods = jax.vmap(jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(0, None)), in_axes=(0, None))(jnp.exp(w), ALPHA_W * jnp.ones(c))
    return jnp.sum(likelihoods)

def score_c(c: Bool[Array, 'N l'], pi: Float[Array, 'l']):
    return jnp.einsum('Nl,l->', c, pi)

def score_x(x: Bool[Array, 'N K c'], c: Bool[Array, 'N l'], w: Float[Array, 'l K c']):
    return jnp.einsum('NKc,Nl,lKC->', x, c, w)

def score(pi, w, c, x):
    return score_pi(pi) + score_w(w) + score_c(c, pi) + score_x(x, c, w)


# %%
score(pi[-1], w[-1], c[-1], x)
# %%
pi[-1].shape
# %%
ALPHA_PI
# %%
pi
# %%
