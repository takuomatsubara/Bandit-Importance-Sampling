# Bandit Importance Sampling

### About

Bandit importance sampling (BIS) is a new class of importance sampling methods designed for settings where the target density is expensive to evaluate. 
In contrast to adaptive importance sampling, which optimizes a proposal distribution, BIS directly designs the samples through a sequential strategy that combines space-filling designs with multi-armed bandits. 
Our method leverages Gaussian process surrogates to guide sample selection, enabling efficient exploration of the parameter space with minimal target evaluations. 
This repository contains source code for the following paper:

> Matsubara, T., Duncan, A., Cotter, S., Zygalakis, K. (2026), `Sampling as Bandits: Evaluation-Efficient Design for Black-Box Densities', *arXiv*.

The following three experiments were presented in the main text:

1. Sampling for Benchmark Densities
2. Bayesian Inference of Lorenz Weather Model with Synthetic Likelihood
3. Exact Bayesian Inference of G & K Model
4. US Precipitation Anomalies Modeled by Markov Random Fields

The experimental results can be reproduced by running the iPython notebook, or the Python code, in each directory.

### Main Class

The first main class is "GPBanditSampler" in src.py, which performs the point selection in the BIS algorithm, given a proposal sequence and a criterion for the point selection.

    class GPBanditSampler():
    """A Class for Point Selection in BIS
    
    Parameters
    ----------
    n_init : integer
        The number of initial evaluations used in BIS
        
    n_eval : integer
        The size of the candidate pool from which a point is selected at each iteration
        
    sequence : instance
        An instance with a function "generate", which generates points to be used for the candidate pool

    Main Attributes
    ----------
    find : function
        A function to find a point that maximizes a given criterion from the candidate pool
    """

The second main class is "GP" in src.py, which is equipped with functions to initialize a GP instance, fit it to given data, and tune the hyperparameter. 

    class GP(BaseEstimator):
    """A Class for GP Regression

    Parameters
    ----------
    mean : function
        A mean function m(x) of a GP prior 
        input x [D vector or 1 x D matrix], output m(x) [0 i.e. scalar]

    kernel: bivariate function
        A covariance function k(x, x') of a GP prior
        input (x1, x2) ([D vector or 1 x D matrix], [D vector or 1 x D matrix]), output k(x1, x2) [0 i.e. scalar]

    Main Attributes
    ----------
    fit : function
        A function to update the GP prior to the GP posterior given data (X, y)

    tune_kernel_parameter : function
        A function to tune the hyperparameters of the GP given data (X, y)

    jensen_exp : function
        A function to compute the value of GP-UJB at input x using the GP posterior

    posterior_mean : function
        A function to compute the value of the GP posterior mean m_n(x) at input x
        
    posterior_cov : function
        A function to compute the value of the GP posterior covariance k_n(x_1, x_2) at input (x_1, x_2)
    
    posterior_scale : function
        A function to compute the value of the GP posterior covariance scale ( k_n(x, x) )^(1/2) at input x
    """

### Data

The dataset used in Section 6.4 is publicly available at https://www.image.ucar.edu/Data/precip_tapering/. Further details of the dataset can be found in the following reference.

> Kaufman, C. G., Schervish, M. J. & Nychka, D. W. (2008), ‘Covariance tapering for likelihood-based estimation in large spatial data sets’, Journal of the American Statistical Association 103(484), 1545–1555.

All the other datasets used in the experiments are available through simulation from the models specified in the paper.

### Dependency

The source code uses Python 3.11.5 and the following packages:

- jax (version 0.4.27)
- tensorflow-probability (version 0.24.0)
- optax (version 0.2.3)
- jaxopt (version 0.8.2)
- gpjax (version 0.9.1)
- blackjax (version 1.2.3)
- matplotlib (version 3.9.2)
- seaborn (version 0.12.2)
- pandas (version 1.5.3)
- tqdm (version 4.66.5)

### License

The source code is licensed under the MIT License (see LICENSE file).


