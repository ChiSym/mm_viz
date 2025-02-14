# %%
%load_ext autoreload
%autoreload 2

# %%
import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp

ALPHA_PI = 1e-2
ALPHA_W = 1e-2
l = 2 # num clusters
categories = 2 # num categories
N = 100 
K = 1
k = jax.random.key(1234)
x = jax.nn.one_hot(jax.random.bernoulli(k, p=.5, shape=(N, K)).astype(jnp.int32), categories, axis=-1) 

# %%
from mm_viz.mixture_model import gibbs
k, key = jax.random.split(k)
pi, w, c, s = gibbs(ALPHA_PI, ALPHA_W, l, key, x, num_steps=100)

# %%
jnp.exp(pi)

# %%
jnp.exp(w)

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
s