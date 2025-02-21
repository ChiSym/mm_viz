import jax
from jaxtyping import Array, Float, Bool, Integer
import jax.numpy as jnp
from functools import partial

# development procedure:
# 1. write a basic version of algorithm with for-loops
# 2. convert basic version to numpy/array-centric version 
# 3. JAX-ify it (np -> jnp and vmap/jitting)
# 4. refactor into GenJAX

# following is very high level psuedocode!
def smc(ALPHA_W, ALPHA_P, key, data, nparticles, num_clusters, num_categories):
    # initialize particles 
    N, K, _ = data.shape
    particles = initialize_particles(ALPHA_W, ALPHA_P, key, num_clusters, K, num_categories, nparticles)
    def smc_step(key, particles, data_index):
        new_particles = []
        weights = []
        key1, key2 = jax.random.split(key, 2)
        for particle in particles:
            # incorporate new datapoint into each particle: assign to cluster using likelihood
            # based on existing cluster weights + params, and then continue with gibbs updates
            # to weights and params
            d  = data[:data_index + 1]
            new_particle = gibbs_proposal(ALPHA_W, ALPHA_P, key1, particle, d)
            new_particles.append(new_particle)

            # compute updated particle weight 
            weight = compute_weight(ALPHA_W, ALPHA_P, new_particle, d)
            weights.append(weight)
        
        # resample particles using updated weights; TODO: don't do this at every iteration!
        particles = resample(key2, new_particles, jnp.array(weights))
        return particles
    
    keys = jax.random.split(key, N)
    for data_index in range(N):
        particles = smc_step(keys[data_index], particles, data_index)

    return particles

def initialize_particles(ALPHA_W, ALPHA_P, key, l, K, categories, nparticles):
    particles = []
    for i in range(nparticles):
        key, subkey = jax.random.split(key)
        weights = jnp.log(jax.random.dirichlet(subkey, ALPHA_W * jnp.ones(l)))

        key, subkey = jax.random.split(key)
        params = jnp.log(jax.random.dirichlet(subkey, ALPHA_P * jnp.ones((l, K, categories))))
        particle = (weights, params, None)
        particles.append(particle)

    return particles

def gibbs_proposal(ALPHA_W: float, ALPHA_P: float, key, particle, data):
    weights, params, assignments = particle 
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    # create new assignments by assigning new datapoint to cluster
    new_datapoint = data[-1]
    c = gibbs_assignments(key1, weights, new_datapoint[None, :], params)
    if assignments is None:
        new_assignments = c
    else:          
        new_assignments = jnp.vstack((assignments, c))
    
    # update weights and params according to new assignment
    new_weights = gibbs_weights(ALPHA_W, key2, new_assignments)
    new_params = gibbs_params(ALPHA_P, key3, data, new_assignments)

    # maybe: update assignments again, but globally? might make sense as a rejuvenation step instead
    new_assignments = gibbs_assignments(key4, new_weights, data, new_params)

    return (new_weights, new_params, new_assignments) 

def gibbs_assignments(key: Array, weights: Float[Array, 'l'], x: Bool[Array, 'N K c'], params: Float[Array, 'l K c']) -> Bool[Array, 'N l']:
    l, k, _ = params.shape
    likelihood = jnp.einsum('NKc,lKc->Nl', x, params)
    posterior = likelihood + weights[None, :]
    logZ = jax.nn.logsumexp(posterior, axis=-1)
    posterior = posterior - logZ[:, None]
    c = jax.nn.one_hot(jax.random.categorical(key, posterior, axis=-1), l)
    return c

def gibbs_weights(ALPHA_W: float, key: Array, assignments: Bool[Array, 'N l']) -> Float[Array, 'l']:
    counts = jnp.sum(assignments, axis=0)
    alpha = counts + ALPHA_W
    return log_dirichlet(key, alpha) # this is numerically unstable

def gibbs_params(ALPHA_P: float, key: Array, x: Bool[Array, 'N K c'], assignments: Bool[Array, 'N l']) -> Float[Array, 'l K c']:
    counts = jnp.einsum('NKc,Nl-> lKc', x, assignments)
    alpha = counts + ALPHA_P
    return log_dirichlet(key, alpha)

def log_dirichlet(key, alpha):
    y = jax.random.dirichlet(key, alpha)
    return jnp.log(y)

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

def compute_weight(ALPHA_W: float, ALPHA_P: float, particle, data):
    weights, params, assignments = particle 
    return score(ALPHA_W, ALPHA_P, weights, params, assignments, data)

def resample(key, particles, weights):
    # normalize weights
    logZ = jax.nn.logsumexp(jnp.array(weights))
    normalized_weights = weights - logZ
    indices = jax.random.categorical(key, normalized_weights * jnp.ones_like(weights[:, None]))
    new_particles = []
    for i in indices:
        new_particles.append(particles[i])
    return new_particles

# questions: SMCP3?