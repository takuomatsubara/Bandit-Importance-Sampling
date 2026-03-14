library(INLA)
library(rSPDE)
library(inlabru)
library(geosphere)
library(sp)
library(fmesher)
library(fmcmc)

load("Data/anom1962.RData")
ind <- !is.na(z)
Y <- z[ind]
loc <- loc[ind,]

mesh <- fm_mesh_2d(loc = loc, max.edge = c(0.5, 10), cutoff = 0.35)
A <- fm_basis(mesh, loc = loc)

bounds_l <- c(0.1, 0.1, 0.1)
bounds_r <- c(5.0, 5.0, 5.0)

op <- matern.operators(
  mesh = mesh,
  range = 1.0,
  sigma = 1.0,
  nu = 2.0,
  m = 1,
  parameterization = "matern",
  compute_higher_order = TRUE,
)

loglikelihood <- function(theta){
  range <- theta[1]
  sigma <- theta[2]
  sigma.e <- theta[3]
  
  ll <- rSPDE.matern.loglike(
    op,
    Y = Y,
    A = A,
    sigma.e = sigma.e,
    range = range,
    sigma = sigma,
  )
  
  return(ll)
}

logprior <- function(theta){
  return( dunif(theta[1], min = bounds_l[1], max = bounds_r[1], log = TRUE) + dunif(theta[2], min = bounds_l[2], max = bounds_r[2], log = TRUE) + dunif(theta[3], min = bounds_l[3], max = bounds_r[3], log = TRUE)  )
}

logposterior <- function(theta){
  if ( any( theta < bounds_l ) || any( theta > bounds_r ) ) {
    return(-Inf)
  }
  return( loglikelihood(theta) + logprior(theta) ) 
}
