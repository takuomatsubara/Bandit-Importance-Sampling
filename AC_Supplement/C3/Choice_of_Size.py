#===========================================
# Import Library
#===========================================

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
jax.config.update("jax_enable_x64", True) # Enable Float64 for more stable matrix inversions.
import blackjax

import gpjax as gpx
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.mcmc import sample_halton_sequence

import sys
sys.path.append("../..")
from src import GP, GPBanditSampler, HaltonSequence, MMD, MCMC_Bounded

import argparse



#===========================================
# GP Training
#===========================================

def fit_gp(_key, X, y):
    gp_mean = lambda x: 0.0
    gp_kernel = gpx.kernels.RBF(n_dims=2, lengthscale=jnp.array([1.0, 1.0]), variance=jnp.array([5.0]))
    gp = GP(gp_mean, gp_kernel)
    gp.tune_kernel_parameter(X, y, rng_key=_key)
    gp.fit(X, y)
    return gp



#===========================================
# Experiment BIS
#===========================================

def one_experiment_exp(key, log_density, bounds, n_eval, sample_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    bandit = GPBanditSampler(n_init=10, n_eval=n_eval, sequence=halton)
    sample_loss = []
    loss_index = jnp.array([12, 15, 19, 24, 31, 39, 49, 62, 78, 99])
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    sample_loss += [ sample_error.compute(X, w) ]
    #====================
    
    for ith in range(10, 100):
        #====== fit GP ======
        key, _key = jax.random.split(key)
        gp = fit_gp(_key, X, y)
        objective = jax.vmap(gp.jensen_exp, in_axes=0, out_axes=0)
        #====================

        #====== bandit ======
        X_new = bandit.find(objective)
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================
        
        #=== compute loss ===
        if ith in loss_index:
            sample_loss += [ sample_error.compute(X, w) ]
        #====================
    
    return jnp.array(sample_loss), key



#===========================================
# Execute
#===========================================

if __name__ == "__main__":
    #============= Parse Argument ==============
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="Banana", required=True)
    args = parser.parse_args()
    #===========================================
    
    #============= Target Density ==============
    if args.target == "Gaussian":
        Srho = jnp.array([[1.0, 0.25], [0.25, 1.0]])
        Srho_inv = jnp.linalg.inv(Srho)
        
        @jax.jit
        def log_density(x):
            return - ((x @ Srho_inv) @ x) / 2.0
        
        bounds = ( jnp.array([-16, -16]), jnp.array([16, 16]) )
        
    elif args.target == "Bimodal":
        Srho = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        Srho_inv = jnp.linalg.inv(Srho)
        
        @jax.jit
        def log_density(x):
            z = jnp.array([x[0], x[1]**2 - 2.0]).T
            return - ((z @ Srho_inv) @ z) / 2.0
        
        bounds = ( jnp.array([-6.0, -6.0]), jnp.array([6.0, 6.0]) )
        
    elif args.target == "Banana":
        Srho = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        Srho_inv = jnp.linalg.inv(Srho)
        
        @jax.jit
        def log_density(x):
            z = jnp.array([x[0], x[1] + x[0]**2 + 1]).T
            return - ((z @ Srho_inv) @ z) / 2.0
        
        bounds = ( jnp.array([-6, -20]), jnp.array([6, 2]) )
        
    else:
        assert False, "No Valid Target Selected"
    #===========================================

    #================ Baseline =================
    key = jrandom.key(0)
    key, _key = jax.random.split(key)
    halton_class = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    target_samples = halton_class.generate(jnp.arange(10000))
    target_logpdfs = jnp.array([ log_density(x) for x in target_samples ])
    target_weights = jnp.exp( target_logpdfs - jsp.special.logsumexp( target_logpdfs ) )
    sample_error = MMD(gpx.kernels.RBF(n_dims=2, lengthscale=0.1, variance=1.0), target_samples, target_weights)
    #===========================================

    #================= Errors ==================
    #sizes = jnp.array([2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15])
    sizes = jnp.array([2**13, 2**15])
    
    for ith in range(sizes.shape[0]):
        sample_losses = jnp.empty((0,10+1))
    
        for kth in range(10):
            sample_loss, key = one_experiment_exp(key, log_density, bounds, sizes[ith], sample_error)
            sample_losses = jnp.vstack((sample_losses, sample_loss))
    
        with open('Data/Size_{:s}_{:04d}_MMD.npy'.format(args.target, sizes[ith]), 'wb') as f:
            jnp.save(f, sample_losses)
    #===========================================


