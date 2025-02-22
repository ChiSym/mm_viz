import jax
import jax.numpy as jnp
from genjax import beta, flip, gen, Target, ChoiceMap, dirichlet, gumbel
from genjax.inference.smc import ImportanceK

def model(ALPHA_W, ALPHA_P, l, K, categories, N):
    weights = jnp.log(dirichlet(ALPHA_W * jnp.ones(l))) @ "weights"

    params = jnp.log(dirichlet(ALPHA_P * jnp.ones((l, K, categories)))) @ "params"

    assignments = jnp.max(gumbel(loc=jnp.zeros((N, l), scale=jnp.ones(N, l))) + weights[None, :], axis=-1)[..., None] == assignments @ "assignments"

    x_logits = jnp.einsum('Nl,lKC->NKC', assignments, params)
    x = jnp.max(jax.random.gumbel(loc=jnp.zeros((N, K, categories)), scale=jnp.ones((N, K, categories))) + x_logits, axis=-1)[..., None] == x @ "x"
    return x

def run_inference(ALPHA_W, ALPHA_P, l, k, categories, N, obs):
    # Create an inference query - a posterior target - by specifying
    # the model, arguments to the model, and constraints.
    posterior_target = Target(model, # the model
                              (ALPHA_W, ALPHA_P, l, k, categories, N), # arguments to the model
                              ChoiceMap.d({"x": obs}), # constraints
                            )

    # Use a library algorithm, or design your own - more on that in the docs!
    alg = ImportanceK(posterior_target, k_particles=50)

    # Everything is JAX compatible by default.
    # JIT, vmap, to your heart's content.
    key = jax.random.key(314159)
    sub_keys = jax.random.split(key, 100)
    _, posterior_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, posterior_target
    )

    # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
    return (posterior_chm["weights"], posterior_chm["params"], posterior_chm["assignments"])

# # Create a generative model.
# @gen
# def beta_bernoulli(α, β):
#     p = beta(α, β) @ "p"
#     v = flip(p) @ "v"
#     return v

# @jax.jit
# def run_inference(obs: bool):
#     # Create an inference query - a posterior target - by specifying
#     # the model, arguments to the model, and constraints.
#     posterior_target = Target(beta_bernoulli, # the model
#                               (2.0, 2.0), # arguments to the model
#                               ChoiceMap.d({"v": obs}), # constraints
#                             )

#     # Use a library algorithm, or design your own - more on that in the docs!
#     alg = ImportanceK(posterior_target, k_particles=50)

#     # Everything is JAX compatible by default.
#     # JIT, vmap, to your heart's content.
#     key = jax.random.key(314159)
#     sub_keys = jax.random.split(key, 50)
#     _, p_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
#         sub_keys, posterior_target
#     )

#     # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
#     return jnp.mean(p_chm["p"])

# (run_inference(True), run_inference(False))