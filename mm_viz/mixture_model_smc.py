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
def smc(data, nparticles):
    # initialize particles 
    particles = initialize_particles()
    def smc_step(particles, data_index):
        new_particles = []
        weights = []
        for particle in particles:
            # incorporate new datapoint into each particle: assign to cluster using likelihood
            # based on existing cluster weights + params, and then continue with gibbs updates
            # to weights and params
            new_particle = gibbs_proposal(particle, data[data_index])
            new_particles.append(new_particle)

            # compute updated particle weight 
            weight = compute_weight(new_particle, data)
            weights.append(weight)
        
        # resample particles using updated weights
        particles = resample(new_particles, weights)
        return particles
    
    for data_index in range(data.shape[0]):
        particles = smc_step(particles, data_index)

    return particles

def initialize_particles():
    pass

def gibbs_proposal():
    pass 

def compute_weight():
    pass

def resample():
    pass

# questions: SMCP3?