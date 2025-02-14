# %%
import jax
import jax.numpy as jnp

# %%
import numpy as np
n = 100
n_groups = 2
n_repeats = 2

def make_var_group(repeats):
    x = np.random.binomial(1, 0.5, n)
    return np.vstack([x for _ in range(repeats)]).T

np.random.seed(1234)
group_arrs = [make_var_group(2) for _ in range(n_groups)]
# %%
x = np.concatenate(group_arrs, axis=1)
x.shape

# %%
x = jax.nn.one_hot(x, 2)
x.shape

# %%
ALPHA_PI = 1
ALPHA_W = 1
l = 2 # num clusters
categories = 2 # num categories
N = 100 
K = n_groups * n_repeats
key = jax.random.key(1234)

# %%
from mm_viz.mixture_model import gibbs
subkey, key = jax.random.split(key)
pi, w, c, s = gibbs(ALPHA_PI, ALPHA_W, l, subkey, x, num_steps=1000)

# %%
from mm_viz.viz import plot_w
ani = plot_w(w)

# %%
ani.save('local_minimum.gif')