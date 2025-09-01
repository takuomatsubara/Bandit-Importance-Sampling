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
sys.path.append("..")
from src import HaltonSequence, MMD, MCMC

import argparse



#===========================================
# Experiment QMC
#===========================================

def one_experiment_qmc(key, reference_loss, log_density, bounds, sample_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    sample_loss = 0
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================

    check_point = 0
    for ith in range(10, 3000):
        #====== halton ======
        X_new = halton.generate(ith).flatten()
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================
        
        #=== compute loss ===
        if ith % 100 == 0:
            sample_loss = sample_error.compute(X, w)
            if sample_loss <= reference_loss:
                check_point = ith
                break
        #====================
    
    #====== init X ======
    sample_loss = 0
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================

    for ith in range(10, 3000):
        #====== halton ======
        X_new = halton.generate(ith).flatten()
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================
        
        #=== compute loss ===
        if ith >= check_point - 100:
            sample_loss = sample_error.compute(X, w)
            if sample_loss <= reference_loss:
                break
        #====================
    
    return ith+1, sample_loss, key



#===========================================
# Execute
#===========================================

if __name__ == "__main__":
    #============= Parse Argument ==============
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="Banana", required=True)
    parser.add_argument('--experiment', type=str, default="QMC", required=True)
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
        
        with open('Data/Sample_Gaussian_BIS_MMD.npy', 'rb') as f:
            losses_bis = jnp.load(f)[:,-1]
        
    elif args.target == "Bimodal":
        Srho = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        Srho_inv = jnp.linalg.inv(Srho)
        
        @jax.jit
        def log_density(x):
            z = jnp.array([x[0], x[1]**2 - 2.0]).T
            return - ((z @ Srho_inv) @ z) / 2.0
        
        bounds = ( jnp.array([-6.0, -6.0]), jnp.array([6.0, 6.0]) )
        
        with open('Data/Sample_Bimodal_BIS_MMD.npy', 'rb') as f:
            losses_bis = jnp.load(f)[:,-1]
            
    elif args.target == "Banana":
        Srho = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        Srho_inv = jnp.linalg.inv(Srho)
        
        @jax.jit
        def log_density(x):
            z = jnp.array([x[0], x[1] + x[0]**2 + 1]).T
            return - ((z @ Srho_inv) @ z) / 2.0
        
        bounds = ( jnp.array([-6, -20]), jnp.array([6, 2]) )
        
        with open('Data/Sample_Banana_BIS_MMD.npy', 'rb') as f:
            losses_bis = jnp.load(f)[:,-1]
        
    else:
        assert False, "No Valid Target Selected"
    #===========================================

    #================== MCMC ===================
    key = jrandom.key(0)
    key, _key = jax.random.split(key)
    halton_class = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    target_samples = halton_class.generate(jnp.arange(10000))
    target_logpdfs = jnp.array([ log_density(x) for x in target_samples ])
    target_weights = jnp.exp( target_logpdfs - jsp.special.logsumexp( target_logpdfs ) )
    sample_error = MMD(gpx.kernels.RBF(n_dims=2, lengthscale=0.1, variance=1.0), target_samples, target_weights)
    #===========================================

    #================= Errors ==================
    if args.experiment == "QMC":
        one_experiment = one_experiment_qmc
    else:
        assert False, "No Valid Experiment Selected"
    
    sample_numbers = jnp.zeros(10)
    sample_losses = jnp.zeros(10)
    
    for kth in range(10):
        print("========================================")
        print(" {:s} - {:s}: {:02d}-th iteration".format(args.target, args.experiment, kth))
        print("========================================")
        sample_number, sample_loss, key = one_experiment(key, losses_bis[kth], log_density, bounds, sample_error)
        sample_numbers = sample_numbers.at[kth].set(sample_number)
        sample_losses = sample_losses.at[kth].set(sample_loss)
    
    print("Sample Number : {:f} ({:f})".format(jnp.mean(sample_numbers), jnp.std(sample_numbers)))
    print("Sample Loss : {:f} ({:f})".format(jnp.mean(sample_losses), jnp.std(sample_losses)))
    #===========================================


