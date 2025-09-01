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
from src import GP, GPBanditSampler, HaltonSequence, GPUCB, GPEIV, TVD, MCMC

import argparse



#===========================================
# GP Training
#===========================================

def fit_gp(key, X, y):
    gp_mean = lambda x: 0.0
    gp_kernel = gpx.kernels.RBF(n_dims=2, lengthscale=jnp.array([1.0, 1.0]), variance=jnp.array([5.0]))
    gp = GP(gp_mean, gp_kernel)
    gp.tune_kernel_parameter(X, y, rng_key=key)
    gp.fit(X, y)
    return gp



#===========================================
# Experiment BIS
#===========================================

def one_experiment_bis(key, log_density, bounds, tvd_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    tvd_loss = []
    loss_index = jnp.array([12, 15, 19, 24, 31, 39, 49, 62, 78, 99])
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================

    #====== fit GP ======
    key, _key = jax.random.split(key)
    gp = fit_gp(_key, X, y)
    objective = jax.vmap(gp.jensen_exp, in_axes=0, out_axes=0)
    gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
    tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
    #====================

    #==== init GPUJB ====
    bandit = GPBanditSampler(n_init=10, n_eval=2048, sequence=halton)
    #====================
    
    for ith in range(10, 100):
        #====== bandit ======
        X_new = bandit.find(objective)
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================
    
        #====== fit GP ======
        key, _key = jax.random.split(key)
        gp = fit_gp(_key, X, y)
        objective = jax.vmap(gp.jensen_exp, in_axes=0, out_axes=0)
        #====================    

        #== MCMC GP Sample ==
        if ith in loss_index:
            gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
            tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
        #====================
    
    return jnp.array(tvd_loss), key



#===========================================
# Experiment QMC
#===========================================

def one_experiment_qmc(key, log_density, bounds, tvd_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    tvd_loss = []
    loss_index = jnp.array([12, 15, 19, 24, 31, 39, 49, 62, 78, 99])
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================
    
    #====== fit GP ======
    key, _key = jax.random.split(key)
    gp = fit_gp(_key, X, y)
    gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
    tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
    #====================
    
    for ith in range(10, 100):
        #====== halton ======
        X_new = halton.generate(ith).flatten()
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================

        #====== fit GP ======
        key, _key = jax.random.split(key)
        gp = fit_gp(_key, X, y)
        #====================    

        #== MCMC GP Sample ==
        if ith in loss_index:
            gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
            tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
        #====================

    return jnp.array(tvd_loss), key



#===========================================
# Experiment RBO
#===========================================

def one_experiment_rbo(key, log_density, bounds, tvd_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    tvd_loss = []
    loss_index = jnp.array([12, 15, 19, 24, 31, 39, 49, 62, 78, 99])
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================

    #====== fit GP ======
    key, _key = jax.random.split(key)
    gp = fit_gp(_key, X, y)
    gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
    tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
    #====================

    #==== init GPUCB ====
    key, _key = jax.random.split(key)
    gpucb = GPUCB(_key, 2, 2048, bounds)
    #====================

    for ith in range(10, 100):
        #====== GP RBO ======
        X_new = gpucb.find(gp) if ith % 2 == 0 else halton.generate(ith).flatten()
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================

        #====== fit GP ======
        key, _key = jax.random.split(key)
        gp = fit_gp(_key, X, y)
        #====================    

        #== MCMC GP Sample ==
        if ith in loss_index:
            gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
            tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
        #====================

    return jnp.array(tvd_loss), key



#===========================================
# Experiment EIV
#===========================================

def one_experiment_eiv(key, log_density, bounds, tvd_error):
    #====== set up ======
    key, _key = jax.random.split(key)
    halton = HaltonSequence(dim=2, bounds=bounds, rngkey=_key)
    tvd_loss = []
    loss_index = jnp.array([12, 15, 19, 24, 31, 39, 49, 62, 78, 99])
    #====================

    #====== init X ======
    X = halton.generate(jnp.arange(10))
    y = jnp.array([log_density(x) for x in X])
    w = jnp.exp( y - jsp.special.logsumexp( y ) )
    #====================

    #====== fit GP ======
    key, _key = jax.random.split(key)
    gp = fit_gp(_key, X, y)
    gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
    tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
    #====================
    
    #==== init GPEIV ====
    key, _key = jax.random.split(key)
    gpeiv = GPEIV(_key, 2, 2048, bounds)
    #====================

    for ith in range(10, 100):
        #====== GP EIV ======
        X_new = gpeiv.find(gp)
        y_new = log_density(X_new)
        #====================
        
        #=== stack point ====
        X = jnp.vstack((X, X_new))
        y = jnp.hstack((y, y_new))
        w = jnp.exp( y - jsp.special.logsumexp( y ) )
        #====================

        #====== fit GP ======
        key, _key = jax.random.split(key)
        gp = fit_gp(_key, X, y)
        #====================    

        #== MCMC GP Sample ==
        if ith in loss_index:
            gp_posterior_mean = jax.vmap(gp.posterior_mean, in_axes=0, out_axes=0)
            tvd_loss += [ tvd_error.compute(gp_posterior_mean) ]
        #====================

    return jnp.array(tvd_loss), key



#===========================================
# Execute
#===========================================

if __name__ == "__main__":
    #============= Parse Argument ==============
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="Banana", required=True)
    parser.add_argument('--experiment', type=str, default="BIS", required=True)
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

    #================ Baseline==================
    key = jrandom.key(0)
    key, _key = jax.random.split(key)
    tvd_error = TVD(jax.vmap(log_density), num=10000, dim=2, bounds=bounds, rngkey=_key)
    #===========================================
    
    #================= Errors ==================
    if args.experiment == "BIS":
        one_experiment = one_experiment_bis
    elif args.experiment == "QMC":
        one_experiment = one_experiment_qmc
    elif args.experiment == "RBO":
        one_experiment = one_experiment_rbo
    elif args.experiment == "EIV":
        one_experiment = one_experiment_eiv
    else:
        assert False, "No Valid Experiment Selected"
    
    tvd_losses = jnp.empty((0,10+1))
    
    for kth in range(10):
        print("========================================")
        print(" {:s} - {:s}: {:02d}-th iteration".format(args.target, args.experiment, kth+1))
        print("========================================")
        tvd_loss, key = one_experiment(key, log_density, bounds, tvd_error)
        tvd_losses = jnp.vstack((tvd_losses, tvd_loss))
    
    with open('Data/GP_{:s}_{:s}_TVD.npy'.format(args.target, args.experiment), 'wb') as f:
        jnp.save(f, tvd_losses)
    #===========================================


