import numpy as np
import polars as pl
from plotnine import ggplot, aes, geom_bar, ylim, position_dodge
from plotnine.animation import PlotnineAnimation


def plot_w(w):
    # plots a single category of w, animated over time
    w_subset = np.exp(np.array(w[:, :, :, 0]))
    T, l, K = w_subset.shape

    category = np.zeros_like(w_subset)
    for i in range(1, K):
        category[..., i] = i

    cluster = np.zeros_like(w_subset)
    for i in range(1, l):
        cluster[:, i] = i

    time = np.arange(T)
    time = np.repeat(time, np.prod(w_subset.shape[1:]))
    
    df = pl.DataFrame({
        'time': time,
        'column': category.ravel(),
        'cluster': cluster.ravel(),
        'cluster_parameters': w_subset.ravel()
    })
    df = df.with_columns(pl.col('cluster').cast(pl.String))

    def plot_fn(t):
        plot = (
            ggplot(df.filter(pl.col('time') == t), aes(x='column', y='cluster_parameters', fill='cluster')) 
            + geom_bar(stat='identity', position=position_dodge(width=0.9))
            + ylim(0, 1)
        )
        return plot

    plots = (plot_fn(k) for k in range(T))

    ani = PlotnineAnimation(plots, interval=10, repeat_delay=1000)

    return ani

