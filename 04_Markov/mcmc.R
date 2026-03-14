library(fmcmc)
source("script.R")

chain <- MCMC(
  initial = ( bounds_l + bounds_r ) / 2,
  fun = logposterior,
  nsteps = 10000,
  burnin = 5000,
  thin = 10,
  kernel = kernel_normal_reflective(lb = bounds_l, ub = bounds_r, scale = 0.1),
  conv_checker = convergence_geweke(5000),
  seed = 1
)

saveRDS(as.matrix(chain), "Data/samples.rds")
