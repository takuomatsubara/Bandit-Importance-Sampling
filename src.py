#===========================================
# Import Library
#===========================================

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as lax
import jax.random as jrandom
jax.config.update("jax_enable_x64", True) # Enable Float64 for more stable matrix inversions.

import optax
import gpjax as gpx
from jaxopt import ScipyBoundedMinimize
import blackjax

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as dist
from tensorflow_probability.substrates.jax.mcmc import sample_halton_sequence

from sklearn.base import BaseEstimator
from copy import deepcopy



#===========================================
# GP 
#===========================================

class GP(BaseEstimator):
    '''=================
    init:
    ----------
        mean : mean function m(x) of GP prior 
               input x [D vector or 1 x D matrix], output m(x) [0 i.e. scalar]
        kernel : covariance function k(x1, x2) of GP prior
                 input (x1, x2) ([D vector or 1 x D matrix], [D vector or 1 x D matrix]), output k(x1, x2) [0 i.e. scalar]
    ================='''
    def __init__(self, mean, kernel, noise=0.0):
        self.mean = mean #scalar-output for x1[D or 1 x D]
        self.mean_vmap = jax.jit(jax.vmap(self.mean, in_axes=0, out_axes=0)) # N vector for x[N x D]
        
        self.kernel = kernel #scalar-output for x1[D or 1 x D] and x2[D or 1 x D]
        self.kernel_vmap_x2 = jax.jit(jax.vmap(self.kernel, in_axes=(None, 0), out_axes=0)) # M vector for x1[D or 1 x D] and X2[M x D]
        self.kernel_vmap_x1x2 = jax.jit(jax.vmap(self.kernel_vmap_x2, in_axes=(0, None), out_axes=0)) # N x M vector for X1[N x D] and X2[M x D]

        self.noise = noise #Gaussian noise
        self.jitter = 1e-6 #jitter for matrix inversion
        
        
    '''=================
    fit: store quantities used for GP-posterior computation given dataset (X, y)
    ----------
        X : input dataset [N x D matrix]
        y : output dataset [D vector]
    ================='''
    def fit(self, X, y):
        M_X = y - self.mean_vmap(X)
        K_XX = self.kernel_vmap_x1x2(X, X) + ( self.jitter + self.noise**2 ) * jnp.eye(X.shape[0])
        L_XX = lax.linalg.cholesky( K_XX ) #https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.linalg.cholesky.html
        
        self.X = X
        self.L_XX_inv = jnp.linalg.inv( L_XX )
        self.K_XX_inv = self.L_XX_inv.T @ self.L_XX_inv
        self.Sigma = self.K_XX_inv @ M_X
        
        
    '''=================
    posterior_mean: value of mean function of GP posterior at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def posterior_mean(self, x):
        return self.mean(x) + self.kernel_vmap_x2(x, self.X) @ self.Sigma
    
    
    '''=================
    posterior_cov: value of covariance function of GP posterior at input vector (x1, x2)
    ----------
        x1 : first input vector [D vector or 1 x D matrix]
        x2 : second input vector [D vector or 1 x D matrix]
    ================='''
    def posterior_cov(self, x1, x2):
        return self.kernel(x1, x2) - ( self.kernel_vmap_x2(x1, self.X) @ self.K_XX_inv ) @ self.kernel_vmap_x2(x2, self.X)

    
    '''=================
    posterior_scale: value of scale (squared standard deviation) of GP posterior at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def posterior_scale(self, x):
        kernel_X2 = self.kernel_vmap_x2(x, self.X)
        return self.kernel(x, x) - ( kernel_X2 @ self.K_XX_inv ) @ kernel_X2
    
    
    '''=================
    jensen_exp: value of proportional term of exp-based jensen-gp surrogate density at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def jensen_exp(self, x):
        return jnp.exp( self.posterior_mean(x) + 0.5 * self.posterior_scale(x) )
    
    
    '''=================
    log_jensen_exp: log value of proportional term of exp-based jensen-gp surrogate density at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def log_jensen_exp(self, x):
        return self.posterior_mean(x) + 0.5 * self.posterior_scale(x)
    
    
    '''=================
    jensen_relu: value of proportional term of relu-based jensen-gp surrogate density at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def jensen_relu(self, x):
        m = self.posterior_mean(x)
        s = self.posterior_scale(x)
        return m * jsp.stats.norm.cdf( m / jnp.sqrt(s) ) + s * jsp.stats.norm.pdf( m / jnp.sqrt(s) )
    
    
    '''=================
    jensen_forth: value of proportional term of 4th-order-relu-based jensen-gp surrogate density at input vector x
    ----------
        x : input vector [D vector or 1 x D matrix]
    ================='''
    def jensen_square(self, x):
        return self.posterior_mean(x)**2 + self.posterior_scale(x)**2
    
    
    '''=================
    tune_kernel_parameter: tune kernel hyperparameter of GP posterior
    ----------
        X : input dataset [N x D matrix]
        y : output dataset [D vector]
        rng_key: JAX random key
        lr: learning rate of ADAM optimiser
        num: number of iteration of ADAM optimiser
    ================='''
    def tune_kernel_parameter(self, X, y, rng_key=jax.random.key(0), lr=0.01, num=1000):
        # create GPJAX wrapper of GP mean, prior, and likelihood
        copy_mean_vmap = deepcopy(self.mean_vmap)
        copy_kernel = deepcopy(self.kernel)
        class GP_Mean(gpx.mean_functions.AbstractMeanFunction):
            def __call__(self, x):
                return copy_mean_vmap(x).reshape(-1,1)
        prior = gpx.gps.Prior(mean_function=GP_Mean(), kernel=copy_kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=X.shape[0], obs_stddev=self.noise)
        
        # optimise hyperparameter
        opt_posterior, _ = gpx.fit(model = prior * likelihood,
            objective = lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data = gpx.Dataset(X=X, y=y.reshape(-1,1)),
            optim = optax.adam(learning_rate=lr),
            num_iters = num,
            safe = True,
            key = rng_key,
            verbose = True)
        
        # copy optimised hyperparameter
        lengthscale = deepcopy(opt_posterior.prior.kernel.lengthscale)
        variance = deepcopy(opt_posterior.prior.kernel.variance)
        
        # update kernel hyperparameter and reset its vmap
        self.kernel.lengthscale = lengthscale
        self.kernel.variance = variance
        self.kernel_vmap_x2 = jax.jit(jax.vmap(self.kernel, in_axes=(None, 0), out_axes=0))
        self.kernel_vmap_x1x2 = jax.jit(jax.vmap(self.kernel_vmap_x2, in_axes=(0, None), out_axes=0))
        
        return (lengthscale.value, variance.value)



#===========================================
# GP Bandit Sampler
#===========================================

class GPBanditSampler():
    '''=================
    init:
    ----------
        n_init : number of initial evaluation
        n_eval : number of candidate pool M
        sequence : class to generate proposal sequence
    ================='''
    def __init__(self, n_init=10, n_eval=1000, sequence=None):
        self.n_step = n_init + n_eval
        self.sequence = sequence
        self.points_candidate = self.sequence.generate(jnp.arange(n_init, self.n_step))
    
    '''=================
    find:
    ----------
        objective : criterion for point selection
                    function of input x [D vector]
    ================='''
    def find(self, objective):
        values_gp = objective( self.points_candidate )
        index_max = jnp.argmax( values_gp )
        point_max = self.points_candidate[ index_max ]

        point_new = self.sequence.generate(self.n_step)
        self.n_step = self.n_step + 1
        
        self.points_candidate = jnp.delete(self.points_candidate, index_max, axis=0)
        self.points_candidate = jnp.append(self.points_candidate, point_new, axis=0)
        
        return point_max



#===========================================
# Other Criterion
#===========================================

class GPUCB():
    def __init__(self, rngkey, dim, num, bounds):
        self.points_candidate = sample_halton_sequence(dim, num_results=num, 
            dtype=jnp.float64, seed=rngkey) * (bounds[1] - bounds[0]) + bounds[0]  
        self.bounds = bounds
        self.volume = jnp.prod(bounds[1] - bounds[0])
        
        
    def find(self, gp):
        objective = lambda x: gp.posterior_mean(x) + jnp.sqrt(2) * gp.posterior_scale(x)
        
        func_vmap = jax.vmap( objective, in_axes=0, out_axes=0 )
        values_gp = func_vmap( self.points_candidate )
        index_max = jnp.argmax( values_gp )
        point_max = self.points_candidate[ index_max ]
 
        solver = ScipyBoundedMinimize(fun=lambda x: - objective(x))
        result = solver.run(point_max, bounds=self.bounds)
        
        return result.params



class GPEIV():
    def __init__(self, rngkey, dim, num, bounds):
        self.points_candidate = sample_halton_sequence(dim, num_results=num, 
            dtype=jnp.float64, seed=rngkey) * (bounds[1] - bounds[0]) + bounds[0]  
        self.bounds = bounds
        self.volume = jnp.prod(bounds[1] - bounds[0])
        
        
    def find(self, gp):
        gp_m2 = lambda x: 2 * gp.posterior_mean(x)
        gp_s2 = lambda x: gp.posterior_scale(x)**2
        gp_t2 = lambda x, y: ( gp.posterior_cov(x, y)**2 ) / ( gp.posterior_scale(y)**2 + gp.jitter**2 )
        
        integrand = lambda x, y: jnp.exp( gp_m2(x) + gp_s2(x) ) * ( jnp.exp( gp_s2(x) ) - jnp.exp( gp_t2(x, y) ) ) 
        integrand_vmap = jax.vmap(integrand, in_axes=(0, None), out_axes=0)
        
        func = lambda y: jnp.mean( integrand_vmap(self.points_candidate, y) / ( self.volume ) )
        func_vmap = jax.vmap(func, in_axes=0, out_axes=0)
        
        index_min = jnp.argmin( func_vmap( self.points_candidate ) )
        point_min = self.points_candidate[ index_min ]
        
        solver = ScipyBoundedMinimize(fun=lambda x: func(x))
        result = solver.run(point_min, bounds=self.bounds)
        
        return result.params


#===========================================
# Sequence
#===========================================
    
class HaltonSequence():
    def __init__(self, dim=1, bounds=(-10, 10), rngkey=jrandom.key(0)):
        self.dim = dim
        self.bounds = bounds
        self.rngkey = rngkey
        
    def generate(self, indices):
        return sample_halton_sequence(self.dim, sequence_indices=indices, 
            dtype=jnp.float64, seed=self.rngkey) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]  



#===========================================
# Error
#===========================================
  
class MMD():
    def __init__(self, kernel, target_samples, target_weights=None):
        self.kernel = kernel #scalar-output for x1[D or 1 x D] and x2[D or 1 x D]
        self.kernel_vmap_x2 = jax.jit(jax.vmap(self.kernel, in_axes=(None, 0), out_axes=0)) 
        self.kernel_vmap_x1x2 = jax.jit(jax.vmap(self.kernel_vmap_x2, in_axes=(0, None), out_axes=0))
        self.target_samples = target_samples
        self.target_weights = target_weights if target_weights is not None else jnp.ones(target_samples.shape[0]) / target_samples.shape[0]
        self.K_01 = ( self.kernel_vmap_x1x2(self.target_samples, self.target_samples) @ self.target_weights ) @ self.target_weights
        
    def compute(self, samples, weights):
        K_02 = ( self.kernel_vmap_x1x2(self.target_samples, samples) @ weights ) @ self.target_weights
        K_03 = ( self.kernel_vmap_x1x2(samples, samples) @ weights ) @ weights
        return self.K_01 - 2.0 * K_02 + K_03



class TVD():
    def __init__(self, target_density, num=1000, dim=1, bounds=(-10, 10), rngkey=jrandom.key(0)):
        self.halton_points = sample_halton_sequence(dim, num_results=num, 
            dtype=jnp.float64, seed=rngkey) * (bounds[1] - bounds[0]) + bounds[0]  
        self.domain_volume = jnp.prod( bounds[1] - bounds[0] ) / num
        _target_log_values = target_density( self.halton_points )
        _target_log_nconst = jnp.log( self.domain_volume ) + jsp.special.logsumexp( _target_log_values ) - jnp.log( self.halton_points.shape[0] )
        self.target_values = jnp.exp( _target_log_values - _target_log_nconst )
        
    def compute(self, gp_density):
        _gp_log_values = gp_density( self.halton_points )
        _gp_log_nconst = jnp.log( self.domain_volume ) + jsp.special.logsumexp( _gp_log_values ) - jnp.log( self.halton_points.shape[0] )
        gp_values = jnp.exp( _gp_log_values - _gp_log_nconst )
        return 0.5 * self.domain_volume * jnp.mean( jnp.abs( self.target_values - gp_values ) )



#===========================================
# MCMC Sampling Algorithms
#===========================================

class MCMC():
    def __init__(self, dim, logdensity):
        self.dim = dim
        self.logdensity = logdensity
        
    def sample(self, rngkey, num=10000, burnin=5000, thin=10):
        mcmc_rngkey, init_rngkey = jax.random.split(rngkey)
        mh_proposal = blackjax.additive_step_random_walk(self.logdensity, blackjax.mcmc.random_walk.normal(jnp.ones(self.dim)))
        init_state0 = mh_proposal.init(jax.random.multivariate_normal(init_rngkey, jnp.zeros(self.dim), jnp.eye(self.dim)))
        mcmc_kernel = jax.jit(mh_proposal.step)
        
        def inference_loop(key, kernel, init_state, num_samples):
            @jax.jit
            def one_step(state, _rngkey):
                state, _ = kernel(_rngkey, state)
                return state, state
            
            _rngkeys = jax.random.split(key, num_samples)
            _, states = jax.lax.scan(one_step, init_state, _rngkeys)
            return states
        
        states = inference_loop(mcmc_rngkey, mcmc_kernel, init_state0, burnin+num*thin)
        return states.position[burnin::thin]


