# %%
%load_ext autoreload
%autoreload 2

# %%
import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp

ALPHA_PI = 1
ALPHA_W = 1
l = 2 # num clusters
categories = 2 # num categories
N = 100 
K = 5
key = jax.random.key(1234)

# %%
from mm_viz.mixture_model import forward_sample
pi_gt, w_gt, c_gt, x = forward_sample(key, ALPHA_PI, ALPHA_W, l, K, categories, N)

# %%
from mm_viz.mixture_model import gibbs
key, subkey = jax.random.split(key)
pi, w, c, s = gibbs(ALPHA_PI, ALPHA_W, l, subkey, x, num_steps=1000)

# %%
from mm_viz.mixture_model import sbc
keys = jax.random.split(key, 1000)
pi_gt, w_gt, c_gt, x, pi, w, c, s = jax.vmap(sbc, in_axes=(None, None, None, None, None, None, 0))(l, K, categories, N, ALPHA_PI, ALPHA_W, keys)

# %%
jnp.exp(pi[:, -1, 0]).mean(axis=0)

# %%
jnp.exp(pi_gt[:, 0]).mean(axis=0)

# %%
jnp.exp(w[:, -1, 0, :, 0]).mean()

# %%
jnp.exp(w_gt[:, 0, :, 0]).mean()

# %%
from plotnine import ggplot, aes, geom_line, labs
import polars as pl
import numpy as np

# Create a DataFrame for the scores
df = pl.DataFrame({'step': range(len(s)), 'score': np.array(s)})

# Create the line plot
plot = (ggplot(df, aes(x='step', y='score'))
        + geom_line()
        + labs(title='Score over Gibbs Sampling Steps', x='Step', y='Score'))

plot

# %%
from mm_viz.mixture_model import score_pi, score_w, score_c, score_x, score
score_pi(ALPHA_PI, pi[-1])
# %%
