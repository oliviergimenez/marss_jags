# Bayesian Multivariate Autoregressive State-Space models - MARSS in Jags

## Context

Species interactions are fascinating. To study them we often have to analyse time series of counts. Counts are challenging because of observation errors, lack of independence and spatial heterogeneity. State-space models are often used to deal with these issues. However, state-space models are complex statistical tools and are not easy to manipulate.  

[Eli Holmes](https://eeholmes.github.io/), [Mark Scheuerell](https://faculty.washington.edu/scheuerl/) and [Eric Ward](https://ericward-noaa.github.io/) share electronic books and courses on [the analysis of time series](https://nwfsc-timeseries.github.io/), including material on state-space models. They introduce MARSS models (MARSS = Multivariate Autoregressive State-Space) as a flexible framework to analyse time series of counts, and provide a package called `MARSS` to implement these models. 

Check it out, these resources are awesome!

Holmes, Scheuerell and Ward also have a great [book on the analysis of time series](https://nwfsc-timeseries.github.io/atsa-labs/), in which they illustrate the frequentist approach with their package `MARSS` and the Bayesian approach with `Jags` and `Stan`. Regarding species interactions more specifically, Eli Holmes has a [dedicated course](https://nwfsc-timeseries.github.io/atsa/Lectures/Week%209/lec_17_estimating_interactions.pdf) in which she asks how interactions change over time, and how environmental change affect interactions, and she uses state-space models to answer these questions. 

Below I will use `Jags` to reproduce an example from the package `MARSS` user's guide. 

## Models

Multivariate state-space models can be written as:

![\begin{equation}
\mathbf{x}_t = \mathbf{B} \mathbf{x}_{t-1}+\mathbf{w}_t \text{ where } \mathbf{w}_t \sim \,\text{N}(0,\mathbf{Q}) \\ \nonumber
\mathbf{y}_t = \mathbf{Z}\mathbf{x}_t+\mathbf{a}+\mathbf{v}_t \text{ where } \mathbf{v}_t \sim \,\text{N}(0,\mathbf{R}) \\ \nonumber
\mathbf{x}_0 = \boldsymbol{\mu} \nonumber
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cmathbf%7Bx%7D_t+%3D+%5Cmathbf%7BB%7D+%5Cmathbf%7Bx%7D_%7Bt-1%7D%2B%5Cmathbf%7Bw%7D_t+%5Ctext%7B+where+%7D+%5Cmathbf%7Bw%7D_t+%5Csim+%5C%2C%5Ctext%7BN%7D%280%2C%5Cmathbf%7BQ%7D%29+%5C%5C+%5Cnonumber%0A%5Cmathbf%7By%7D_t+%3D+%5Cmathbf%7BZ%7D%5Cmathbf%7Bx%7D_t%2B%5Cmathbf%7Ba%7D%2B%5Cmathbf%7Bv%7D_t+%5Ctext%7B+where+%7D+%5Cmathbf%7Bv%7D_t+%5Csim+%5C%2C%5Ctext%7BN%7D%280%2C%5Cmathbf%7BR%7D%29+%5C%5C+%5Cnonumber%0A%5Cmathbf%7Bx%7D_0+%3D+%5Cboldsymbol%7B%5Cmu%7D+%5Cnonumber%0A%5Cend%7Bequation%7D%0A)

Briefly speaking, ![](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7By%7D_t) is a vector of observed log counts for each species, ![\mathbf{x}_t](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7Bx%7D_t) is for the true log abundances, ![\mathbf{B}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7BB%7D) is a matrix that captures species interactions, ![\mathbf{w}_t](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7Bw%7D_t) the process error, ![\mathbf{v}_t](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7Bv%7D_t) the observation error, ![\mathbf{a}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cmathbf%7Ba%7D) is a vector of intrinsic growth rate for each species. There are many variations of this model. For example, we may wish to incorporate environmental covariates. More details can be found in the `MARSS` user's guide that you can easily access by typing in the following command.

```r
RShowDoc("UserGuide", package="MARSS")
```

Chapter 14 on the *Estimation of species interaction strengths with and without covariates* will be of much interest to us. You can easily get the code and data that come with this chapter by typing in the following command.

```r
RShowDoc("Chapter_SpeciesInteractions.R", package="MARSS")
```

Now let's get to it. We will use a dataset from a landmark paper on MARSS by Ives et al. (2003) entitled *Estimating community stability and ecological interactions from time-series data* and published in the journal Ecological Monographs.

## Read in and visualize data

Prerequisites.

```r
library(tidyverse)
theme_set(theme_light()) 
library(janitor)
library(lubridate)
library(MARSS)
```

Load in the plankton data. Only use the plankton, daphnia, and non-daphnia.

```r
data(ivesDataByWeek)
dat <- ivesDataByWeek %>%
  as_tibble() %>% 
  clean_names() %>% # clean column names
  select(large_phyto, small_phyto, daphnia, non_daphnia) %>% # select species
  mutate(across(where(is.double), log)) %>% # log transform all columns
  mutate(across(where(is.double), scale, scale = FALSE)) %>% # center all columns
  clean_names() # clean column names
dat
```

```
## # A tibble: 269 x 4
##    large_phyto[,1] small_phyto[,1] daphnia[,1] non_daphnia[,1]
##              <dbl>           <dbl>       <dbl>           <dbl>
##  1           0.152          -0.504      -1.22           -1.63 
##  2           0.383          -0.271      -0.906          -1.68 
##  3           0.491          -0.222      -0.582          -1.04 
##  4          -0.538          -0.820      -1.32           -0.664
##  5          -0.394          -0.649      -0.853          -1.20 
##  6          -1.44           -0.704      -1.71           -0.748
##  7          -1.42           -0.893      -0.847          -1.03 
##  8          -1.17           -0.688      -1.13           -1.18 
##  9          -0.739          -0.611      -2.22           -1.61 
## 10          NA              NA          NA              NA    
## # … with 259 more rows
```

Plot all data.

```r
dat %>%
  mutate(week = row_number()) %>% # add week id
  pivot_longer(large_phyto:non_daphnia, 
               values_to = "log_biomass", 
               names_to = "species") %>%
  ggplot() + 
  aes(x = week, y = log_biomass, color = species) + 
  geom_line() +
  labs(x = "Week", 
       y = "Biomass (log)",
       color = "Species") + 
  geom_hline(yintercept = 0, lty = "dashed") + 
  expand_limits(x = 0)
```

![](unnamed-chunk-5-1.png)<!-- -->

Plot by species.

```r
dat %>%
  mutate(week = row_number()) %>% # add week id
  pivot_longer(large_phyto:non_daphnia, 
               values_to = "log_biomass", 
               names_to = "species") %>%
  ggplot() + 
  aes(x = week, y = log_biomass) + 
  geom_line() +
  labs(x = "Week", 
       y = "Biomass (log)") + 
  geom_hline(yintercept = 0, lty = "dashed", color = "gray70") + 
  expand_limits(x = 0) + 
  facet_wrap(~species)
```

![](unnamed-chunk-6-1.png)<!-- -->

## MARSS model in the frequentist framework

We consider a model with the following assumptions:
* All phytoplankton share the same process variance.
* All zooplankton share the same process variance.
* Phytoplankton and zooplankton have different measurement variances.
* Measurement errors are independent.
* Process errors are independent.

We fit this model with the `MARSS` package. We need to specify the ingredients first. 

```r
Q <- matrix(list(0), 4, 4)
diag(Q) <- c("Phyto", "Phyto", "Zoo", "Zoo")
R <- matrix(list(0), 4, 4)
diag(R) <- c("Phyto", "Phyto", "Zoo", "Zoo")
plank.model.0 <- list(
  B = "unconstrained", U = "zero", Q = Q,
  Z = "identity", A = "zero", R = R,
  x0 = "unequal", tinitx = 1
)
plank.model.0
```

```
## $B
## [1] "unconstrained"
## 
## $U
## [1] "zero"
## 
## $Q
##      [,1]    [,2]    [,3]  [,4] 
## [1,] "Phyto" 0       0     0    
## [2,] 0       "Phyto" 0     0    
## [3,] 0       0       "Zoo" 0    
## [4,] 0       0       0     "Zoo"
## 
## $Z
## [1] "identity"
## 
## $A
## [1] "zero"
## 
## $R
##      [,1]    [,2]    [,3]  [,4] 
## [1,] "Phyto" 0       0     0    
## [2,] 0       "Phyto" 0     0    
## [3,] 0       0       "Zoo" 0    
## [4,] 0       0       0     "Zoo"
## 
## $x0
## [1] "unequal"
## 
## $tinitx
## [1] 1
```
Then we fit the model.

```r
kem.plank.0 <- dat %>%
  t() %>%
  MARSS(model = plank.model.0)
```

```
## Success! abstol and log-log tests passed at 301 iterations.
## Alert: conv.test.slope.tol is 0.5.
## Test with smaller values (<0.1) to ensure convergence.
## 
## MARSS fit is
## Estimation method: kem 
## Convergence test: conv.test.slope.tol = 0.5, abstol = 0.001
## Estimation converged in 301 iterations. 
## Log-likelihood: -410.8999 
## AIC: 869.7998   AICc: 873.3501   
##  
##                  Estimate
## R.Phyto           0.41998
## R.Zoo             0.06767
## B.(1,1)           0.76682
## B.(2,1)           0.18777
## B.(3,1)          -0.42591
## B.(4,1)          -0.32672
## B.(1,2)           0.28961
## B.(2,2)           0.51355
## B.(3,2)           2.28596
## B.(4,2)           1.35405
## B.(1,3)          -0.01823
## B.(2,3)           0.00521
## B.(3,3)           0.49159
## B.(4,3)          -0.21804
## B.(1,4)           0.13090
## B.(2,4)          -0.04494
## B.(3,4)           0.38885
## B.(4,4)           0.83103
## Q.Phyto           0.11955
## Q.Zoo             0.17140
## x0.X.large_phyto  0.20698
## x0.X.small_phyto  0.01609
## x0.X.daphnia     -1.13943
## x0.X.non_daphnia -1.69203
## Initial states (x0) defined at t=1
## 
## Standard errors have not been calculated. 
## Use MARSSparamCIs to compute CIs and bias estimates.
```

We may get the estimates in a more readable format. For example, let's have a look to the interactions. We denote LP for large phytoplankton, SP for small phytoplankton, D for Daphnia and ND for non-Daphnia. These estimates describe the effect of the density of species j on the per capita growth rate of species i.

```r
B.0 <- coef(kem.plank.0, type = "matrix")$B[1:4, 1:4]
rownames(B.0) <- colnames(B.0) <- c("LP", "SP", "D", "ND")
print(B.0, digits = 2)
```

```
##       LP   SP       D     ND
## LP  0.77 0.29 -0.0182  0.131
## SP  0.19 0.51  0.0052 -0.045
## D  -0.43 2.29  0.4916  0.389
## ND -0.33 1.35 -0.2180  0.831
```

The effect of species j on species i is given by the cell at i-th row and j-th column. The B matrix suggests that SP has a possitive effect on D (2.29). 
In the diagonal, we have the strenght of density-dependence: if species i is density-independent, then ![B_{i,i}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+B_%7Bi%2Ci%7D) equals 1; smaller ![B_{i,i}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+B_%7Bi%2Ci%7D) means more density dependence.

## MARSS model in the Bayesian framework

Let's try to reproduce these results in a Bayesian framework with `Jags`. 

```r
library(R2jags)
```

Let's explore the scaled gama prior for precision which gives us the scaled half-t prior for the standard deviation in `Jags`. Check out [this presentation](http://genome.jouy.inra.fr/applibugs/applibugs.18_06_21.mplummer.pdf) by Martyn Plummer for more details.

```r
LaplacesDemon::rhalft(1000, scale = 1, nu = 10) %>%
  as_tibble() %>%
  ggplot(aes(value)) +
  geom_histogram(fill = "white", color = "black")
```

![](unnamed-chunk-11-1.png)<!-- -->

Write the model.

```r
jagsscript <- cat("
model {  

   # Estimate the initial state vector of population abundances
   for(i in 1:nSpecies) {
      X[i,1] ~ dnorm(0,1) # weakly informative normal prior 
      xknot[i] <- X[i,1]
   }

   # B matrix of interactions
   B[1, 1] <- alpha[1]
   B[1, 2] <- alpha[2]
   B[1, 3] <- alpha[3]
   B[1, 4] <- alpha[4]
   B[2, 1] <- alpha[5]
   B[2, 2] <- alpha[6]
   B[2, 3] <- alpha[7]
   B[2, 4] <- alpha[8]
   B[3, 1] <- alpha[9]
   B[3, 2] <- alpha[10]
   B[3, 3] <- alpha[11]
   B[3, 4] <- alpha[12]
   B[4, 1] <- alpha[13]
   B[4, 2] <- alpha[14]
   B[4, 3] <- alpha[15]
   B[4, 4] <- alpha[16]
   for (k in 1:16){
   alpha[k] ~ dunif(-1, 1)
   }

   # Autoregressive process
   for(t in 2:nYears) {
      for(i in 1:nSpecies) {
         predX[i,t] <- inprod(B[i,], X[,t-1])
         X[i,t] ~ dnorm(predX[i,t], tauQ[species[i]])
      }
   }
	 tauQ[1] ~ dscaled.gamma(1, 10)
	 tauQ[2] ~ dscaled.gamma(1, 10)
   Qp <- 1 / tauQ[1]
   Qd <- 1 / tauQ[2]

   # Observation model
   for(t in 1:nYears) {
     for(i in 1:nSpecies) {
       counts[i,t] ~ dnorm(X[i,t], tauR[species[i]])
     }
   }
	 tauR[1] ~ dscaled.gamma(1, 10)
	 tauR[2] ~ dscaled.gamma(1, 10)
   Rp <- 1 / tauR[1]
   Rd <- 1 / tauR[2]

}  

",file="marss-jags.txt")
```

Put the data in a list, specify the parameters to monitor.

```r
tdat <- t(dat)
jags.data <- list(counts = tdat, 
                  nSpecies = nrow(tdat), 
                  nYears = ncol(tdat),
                  species = c(1, 1, 2, 2))
jags.params <- c("Qp", "Qd", "xknot","alpha", "Rp", "Rd", "X") 
model.loc <- "marss-jags.txt" # name of the txt file
```

Now run `Jags`!

```r
mod_1 <- jags(jags.data, 
              parameters.to.save = jags.params, 
              model.file = model.loc, 
              n.chains = 2, 
              n.burnin = 2500, 
              n.thin = 1, 
              n.iter = 5000)  
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 363
##    Unobserved stochastic nodes: 1809
##    Total graph size: 3530
## 
## Initializing model
```

Inspect estimates. Probably needs to run it longer to improve n.eff. For illustration purpose, that'll do. 

```r
mod_1
```

```
## Inference for Bugs model at "marss-jags.txt", fit using jags,
##  2 chains, each with 5000 iterations (first 2500 discarded)
##  n.sims = 5000 iterations saved
##           mu.vect sd.vect    2.5%     25%     50%     75%   97.5%  Rhat n.eff
## Qd          0.243   0.099   0.095   0.169   0.226   0.302   0.475 1.044    42
## Qp          0.272   0.107   0.120   0.200   0.254   0.320   0.579 1.516     6
## Rd          0.163   0.071   0.022   0.116   0.160   0.206   0.315 1.155    41
## Rp          0.344   0.105   0.012   0.296   0.351   0.408   0.521 1.359    14
## X[1,1]      0.212   0.462  -0.717  -0.084   0.205   0.516   1.119 1.005  5000
## X[2,1]     -0.194   0.364  -0.896  -0.446  -0.201   0.048   0.532 1.017    93
## X[3,1]     -0.914   0.369  -1.597  -1.167  -0.935  -0.671  -0.134 1.003  1400
## X[4,1]     -1.378   0.351  -2.018  -1.619  -1.398  -1.161  -0.631 1.001  5000
## X[1,2]      0.114   0.409  -0.717  -0.160   0.129   0.394   0.913 1.012   130
## X[2,2]     -0.007   0.327  -0.632  -0.240  -0.014   0.214   0.629 1.008   210
## X[3,2]     -0.997   0.318  -1.645  -1.200  -0.991  -0.788  -0.379 1.002  5000
## X[4,2]     -1.361   0.321  -1.946  -1.586  -1.372  -1.154  -0.707 1.006   280
## X[1,3]      0.037   0.398  -0.738  -0.232   0.044   0.319   0.806 1.035    49
## ...
## X[1,268]    0.511   0.380  -0.244   0.263   0.515   0.757   1.259 1.008   200
## X[2,268]    0.307   0.317  -0.316   0.098   0.310   0.520   0.924 1.002   950
## X[3,268]    0.257   0.315  -0.387   0.047   0.267   0.470   0.857 1.003   570
## X[4,268]   -0.252   0.291  -0.817  -0.441  -0.256  -0.062   0.321 1.001  5000
## X[1,269]    0.846   0.440  -0.018   0.542   0.846   1.158   1.654 1.057    32
## X[2,269]    0.638   0.420  -0.200   0.360   0.642   0.934   1.443 1.017   100
## X[3,269]   -0.058   0.336  -0.700  -0.282  -0.075   0.161   0.647 1.012   150
## X[4,269]   -0.153   0.327  -0.797  -0.364  -0.152   0.056   0.503 1.002  1400
## alpha[1]    0.599   0.160   0.239   0.501   0.616   0.711   0.870 1.038   170
## alpha[2]    0.150   0.221  -0.248   0.005   0.129   0.284   0.660 1.286     9
## alpha[3]    0.022   0.070  -0.108  -0.027   0.019   0.069   0.163 1.143    15
## alpha[4]    0.143   0.100  -0.049   0.074   0.142   0.209   0.336 1.030    70
## alpha[5]    0.102   0.172  -0.195  -0.021   0.092   0.215   0.446 1.072   170
## alpha[6]    0.566   0.175   0.211   0.450   0.568   0.685   0.894 1.070    40
## alpha[7]    0.037   0.078  -0.096  -0.022   0.031   0.088   0.200 1.021   230
## alpha[8]   -0.097   0.114  -0.314  -0.175  -0.094  -0.021   0.135 1.044    41
## alpha[9]   -0.067   0.218  -0.532  -0.198  -0.068   0.092   0.338 1.079    34
## alpha[10]   0.877   0.140   0.456   0.839   0.923   0.967   0.997 1.180    32
## alpha[11]   0.527   0.106   0.313   0.458   0.532   0.597   0.726 1.155    18
## alpha[12]   0.519   0.167   0.212   0.407   0.503   0.617   0.887 1.024   110
## alpha[13]  -0.141   0.167  -0.487  -0.255  -0.131  -0.034   0.171 1.081    41
## alpha[14]   0.738   0.138   0.451   0.642   0.745   0.839   0.976 1.075    40
## alpha[15]  -0.198   0.079  -0.362  -0.247  -0.198  -0.145  -0.041 1.159    15
## alpha[16]   0.839   0.103   0.619   0.772   0.851   0.922   0.992 1.154    16
## xknot[1]    0.212   0.462  -0.717  -0.084   0.205   0.516   1.119 1.005  5000
## xknot[2]   -0.194   0.364  -0.896  -0.446  -0.201   0.048   0.532 1.017    93
## xknot[3]   -0.914   0.369  -1.597  -1.167  -0.935  -0.671  -0.134 1.003  1400
## xknot[4]   -1.378   0.351  -2.018  -1.619  -1.398  -1.161  -0.631 1.001  5000
## deviance  455.623 157.971 -39.036 427.024 495.739 546.645 624.539 1.413     9
## 
## For each parameter, n.eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor (at convergence, Rhat=1).
## 
## DIC info (using the rule, pD = var(deviance)/2)
## pD = 10854.9 and DIC = 11310.5
## DIC is an estimate of expected predictive error (lower deviance is better).
```

Check convergence.

```r
traceplot(mod_1, 
          ask = FALSE,
          varname = c("Qp", "Qd", "xknot","alpha", "Rp", "Rd"))
```

![](unnamed-chunk-16-1.png)<!-- 
-->![](unnamed-chunk-16-2.png)<!-- 
-->![](unnamed-chunk-16-3.png)<!-- 
-->![](unnamed-chunk-16-4.png)<!-- 
-->![](unnamed-chunk-16-5.png)<!-- 
-->![](unnamed-chunk-16-6.png)<!-- 
-->![](unnamed-chunk-16-7.png)<!-- 
-->![](unnamed-chunk-16-8.png)<!-- 
-->![](unnamed-chunk-16-9.png)<!-- 
-->![](unnamed-chunk-16-10.png)<!-- 
-->![](unnamed-chunk-16-11.png)<!-- 
-->![](unnamed-chunk-16-12.png)<!-- 
-->![](unnamed-chunk-16-13.png)<!-- 
-->![](unnamed-chunk-16-14.png)<!-- 
-->![](unnamed-chunk-16-15.png)<!-- 
-->![](unnamed-chunk-16-16.png)<!-- 
-->![](unnamed-chunk-16-17.png)<!-- 
-->![](unnamed-chunk-16-18.png)<!-- 
-->![](unnamed-chunk-16-19.png)<!-- 
-->![](unnamed-chunk-16-20.png)<!-- 
-->![](unnamed-chunk-16-21.png)<!-- 
-->![](unnamed-chunk-16-22.png)<!-- 
-->![](unnamed-chunk-16-23.png)<!-- 
-->![](unnamed-chunk-16-24.png)<!-- -->

Get Q and R estimates and compare to MLEs. 

```r
mod_1$BUGSoutput$mean$Qd
```

```
## [1] 0.2428883
```

```r
mod_1$BUGSoutput$mean$Qp
```

```
## [1] 0.2716571
```

```r
mod_1$BUGSoutput$mean$Rd
```

```
## [1] 0.1629837
```

```r
mod_1$BUGSoutput$mean$Rp
```

```
## [1] 0.3437218
```


```r
kem.plank.0$coef['Q.Zoo']
```

```
##     Q.Zoo 
## 0.1713966
```

```r
kem.plank.0$coef['Q.Phyto']
```

```
##   Q.Phyto 
## 0.1195536
```

```r
kem.plank.0$coef['R.Zoo']
```

```
##      R.Zoo 
## 0.06767447
```

```r
kem.plank.0$coef['R.Phyto']
```

```
##   R.Phyto 
## 0.4199771
```


Get init X estimates and compare to MLEs. 

```r
mod_1$BUGSoutput$mean$xknot
```

```
## [1]  0.2115938 -0.1937940 -0.9135877 -1.3777157
```


```r
kem.plank.0$coef['x0.X.large_phyto']
```

```
## x0.X.large_phyto 
##        0.2069771
```

```r
kem.plank.0$coef['x0.X.small_phyto']
```

```
## x0.X.small_phyto 
##       0.01608991
```

```r
kem.plank.0$coef['x0.X.daphnia']
```

```
## x0.X.daphnia 
##    -1.139434
```

```r
kem.plank.0$coef['x0.X.non_daphnia']
```

```
## x0.X.non_daphnia 
##        -1.692034
```

Get B estimates and compare to MLEs.

```r
round(matrix(mod_1$BUGSoutput$mean$alpha, byrow = TRUE, ncol = 4) ,2)
```

```
##       [,1] [,2]  [,3]  [,4]
## [1,]  0.60 0.15  0.02  0.14
## [2,]  0.10 0.57  0.04 -0.10
## [3,] -0.07 0.88  0.53  0.52
## [4,] -0.14 0.74 -0.20  0.84
```


```r
print(B.0, digits = 2)
```

```
##       LP   SP       D     ND
## LP  0.77 0.29 -0.0182  0.131
## SP  0.19 0.51  0.0052 -0.045
## D  -0.43 2.29  0.4916  0.389
## ND -0.33 1.35 -0.2180  0.831
```

Compare observed counts and estimated abundances.
```r
pivot_dat <- dat %>%
  mutate(week = row_number()) %>%
  pivot_longer(large_phyto:non_daphnia, 
               values_to = "log_biomass", 
               names_to = "species")

mod_1$BUGSoutput$sims.matrix %>%
  as_tibble() %>%
  pivot_longer(cols = everything(),  values_to = "value", names_to = "parameter") %>%
  filter(str_detect(parameter, "X")) %>%
  mutate(species = rep(rep(unique(pivot_dat$species), 269),5000),
         week = as.numeric(rep(gl(269, 4),5000))) %>%
  group_by(parameter, species, week) %>%
  summarize(medianN = median(value),
            lci = quantile(value, probs = 2.5/100),
            uci = quantile(value, probs = 97.5/100)) %>%
  arrange(week) %>%
  ggplot() + 
  geom_ribbon(aes(x = week, y = medianN, ymin = lci, ymax = uci), fill = "red", alpha = 0.3) + 
  geom_line(aes(x = week, y = medianN), lty = "dashed", color = "red") +
  geom_point(data = pivot_dat, aes(x = week, y = log_biomass)) + 
  labs(x = "Week", 
       y = "Biomass (log)") + 
  geom_hline(yintercept = 0, lty = "dashed", color = "gray70") + 
  expand_limits(x = 0) + 
  facet_wrap(~species, scales = "free_y")
```

![](unnamed-chunk-23-1.png)<!-- -->



## Yet to be done

### Getting closer to Ives' et al (2003) results

We are far from the estimates that Ives and colleagues obtained in their paper. This is because we do not fit the same model. In Chapter 14 on the *Estimation of species interaction strengths with and without covariates* of the MARSS package user's guide, additionnal steps are given to fit models more similar to Ives' model.  

### Tackle a more complex problem

At some stage, I would like to analyse the data from this paper [A multi-decade time ser ies of kelp forest community structure at San Nicolas Island, California (USA)](https://esajournals.onlinelibrary.wiley.com/doi/epdf/10.1890/13-0561R.1).

These data are analysed in Eli Holmes' course on [estimating interactions](https://nwfsc-timeseries.github.io/atsa/Lectures/Week%209/lec_17_estimating_interactions.pdf) starting at slide 31. Below are some preliminary descriptive analyses.

When you go on the [data paper webpage](http://www.esapubs.org/archive/ecol/E094/244/), you have the [raw data](http://www.esapubs.org/archive/ecol/E094/244/#data) and the [metadata](http://www.esapubs.org/archive/ecol/E094/244/metadata.php). 

Broadly speaking, we have:

* 7 sites around the island
* Biannual surveys from 1980-2011 (n= 63)
* Divers collect data on:
* Fish (59 spp)
* Inverts (14 spp)
* Kelps (6 spp)

Get sea otter (Enhydra lutris) counts.

```r
ind_otters <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Table2_independent_sea_otters.csv") %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date)) %>%
  select(-date) %>%
  pivot_longer(cols = west:south, values_to = "counts", names_to = "region") %>%
  add_column(stage = "independent")
  
pup_otters <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Table3_sea_otter_pups.csv") %>% 
  clean_names() %>%
  mutate(date = mdy(year),
         year = year(date)) %>%
  select(-date) %>%
  pivot_longer(cols = west:south, values_to = "counts", names_to = "region") %>%
  add_column(stage = "pup")

otters <- bind_rows(ind_otters, pup_otters)
```

Visualize.

```r
otters %>%
  count(year, region, wt = counts) %>%
  ggplot() + 
  aes(x = year, y = n, fill = region) +
  geom_col() +
  labs(y = "# otters",
       fill = "Region")
```

![](unnamed-chunk-25-1.png)<!-- -->


Now let's get the counts for all species.

Benthic fishes first. 

```r
benthicfish_raw <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Benthic%20fish%20density%20raw%20data.csv")

benthicfish <- benthicfish_raw %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date)) %>%
#  separate(date, c("year", "month", "day"), "-")
  select(station, year, species_code, adult_density, juv_density)

#dat %>% View()

benthicfish %>% count(station) # 7 stations/sites
```

```
## # A tibble: 7 x 2
##   station     n
##     <dbl> <int>
## 1       1  9731
## 2       2  9361
## 3       3  9361
## 4       4  9731
## 5       5  9361
## 6       6  9509
## 7       7  5365
```

```r
benthicfish %>% count(year) # 30 years
```

```
## # A tibble: 30 x 2
##     year     n
##    <dbl> <int>
##  1  1981   888
##  2  1982  1998
##  3  1983  2035
##  4  1984  1295
##  5  1985  2220
##  6  1986  2405
##  7  1987  2590
##  8  1988  2590
##  9  1989  2590
## 10  1990  1295
## # … with 20 more rows
```

```r
benthicfish %>% count(species_code) # 37 species
```

```
## # A tibble: 37 x 2
##    species_code     n
##           <dbl> <int>
##  1         1002  1687
##  2         1003  1687
##  3         1011  1687
##  4         1012  1687
##  5         1014  1687
##  6         1016  1687
##  7         1017  1687
##  8         1018  1687
##  9         1019  1687
## 10         1020  1687
## # … with 27 more rows
```


```r
midwaterfish_raw <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Midwater%20fish%20density%20raw%20data.csv")

midwaterfish <- midwaterfish_raw %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date)) %>%
#  separate(date, c("year", "month", "day"), "-")
  select(station, year, species_code, adult_density, juv_density)

#dat %>% View()

midwaterfish %>% count(station) # 7 stations/sites
```

```
## # A tibble: 7 x 2
##   station     n
##     <dbl> <int>
## 1       1 15635
## 2       2 15045
## 3       3 15045
## 4       4 15576
## 5       5 15281
## 6       6 15340
## 7       7  8555
```

```r
midwaterfish %>% count(year) # 30 years
```

```
## # A tibble: 30 x 2
##     year     n
##    <dbl> <int>
##  1  1981  1770
##  2  1982  3481
##  3  1983  3245
##  4  1984  2360
##  5  1985  3540
##  6  1986  3835
##  7  1987  4130
##  8  1988  4130
##  9  1989  4130
## 10  1990  2065
## # … with 20 more rows
```

```r
midwaterfish %>% count(species_code) # 59 species
```

```
## # A tibble: 59 x 2
##    species_code     n
##           <dbl> <int>
##  1         1001  1703
##  2         1002  1703
##  3         1003  1703
##  4         1004  1703
##  5         1005  1703
##  6         1006  1703
##  7         1008  1703
##  8         1009  1703
##  9         1010  1703
## 10         1011  1703
## # … with 49 more rows
```



```r
benthicover_raw <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Benthic%20cover%20raw%20data.csv")

benthicover <- benthicover_raw %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date)) %>%
#  separate(date, c("year", "month", "day"), "-")
  select(station, year, species_code, cover)

#dat %>% View()

benthicover %>% count(station) # 7 stations/sites
```

```
## # A tibble: 7 x 2
##   station     n
##     <dbl> <int>
## 1       1 84546
## 2       2 81620
## 3       3 81312
## 4       4 87780
## 5       5 85470
## 6       6 83160
## 7       7 45738
```

```r
benthicover %>% count(year) # 31 years
```

```
## # A tibble: 31 x 2
##     year     n
##    <dbl> <int>
##  1  1980  9240
##  2  1981 18480
##  3  1982 18480
##  4  1983 16786
##  5  1984 13706
##  6  1985 18480
##  7  1986 20020
##  8  1987 21406
##  9  1988 21560
## 10  1989 21406
## # … with 21 more rows
```

```r
benthicover %>% count(species_code) # 154 species
```

```
## # A tibble: 154 x 2
##    species_code     n
##    <chr>        <int>
##  1 0000          3569
##  2 0002          3569
##  3 0012          3569
##  4 0025          3569
##  5 0029          3569
##  6 0030          3569
##  7 0045          3569
##  8 0046          3569
##  9 0047          3569
## 10 0048          3569
## # … with 144 more rows
```

```r
benthicover %>% count(cover) # 21 levels
```

```
## # A tibble: 21 x 2
##    cover      n
##    <dbl>  <int>
##  1     0 518132
##  2     5  13063
##  3    10   5393
##  4    15   3089
##  5    20   2083
##  6    25   1555
##  7    30   1167
##  8    35    903
##  9    40    740
## 10    45    612
## # … with 11 more rows
```


```r
benthicdensity_raw <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Benthic%20density%20raw%20data.csv")

benthicdensity <- benthicdensity_raw %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date)) %>%
#  separate(date, c("year", "month", "day"), "-")
  select(station, year, species_code, density)

#dat %>% View()

benthicdensity %>% count(station) # 7 stations/sites
```

```
## # A tibble: 7 x 2
##   station     n
##     <dbl> <int>
## 1       1  5187
## 2       2  5035
## 3       3  5035
## 4       4  5415
## 5       5  5320
## 6       6  5054
## 7       7  2869
```

```r
benthicdensity %>% count(year) # 31 years
```

```
## # A tibble: 31 x 2
##     year     n
##    <dbl> <int>
##  1  1980   570
##  2  1981  1140
##  3  1982  1140
##  4  1983  1007
##  5  1984   798
##  6  1985  1121
##  7  1986  1235
##  8  1987  1330
##  9  1988  1330
## 10  1989  1330
## # … with 21 more rows
```

```r
benthicdensity %>% count(species_code) # 19 species
```

```
## # A tibble: 19 x 2
##    species_code     n
##    <chr>        <int>
##  1 0024          1785
##  2 0029          1785
##  3 0030          1785
##  4 0066          1785
##  5 0067          1785
##  6 0075          1785
##  7 0081          1785
##  8 0082          1785
##  9 0085          1785
## 10 0089          1785
## 11 0092          1785
## 12 0152          1785
## 13 0553          1785
## 14 0556          1785
## 15 0557          1785
## 16 0558          1785
## 17 0574          1785
## 18 0575          1785
## 19 0589          1785
```



```r
giantkelp_raw <- read_csv("http://www.esapubs.org/archive/ecol/E094/244/Giant%20kelp%20size%20frequency.csv")

giantkelp <- giantkelp_raw %>% 
  clean_names() %>%
  mutate(date = mdy(date),
         year = year(date))

giantkelp
```

```
## # A tibble: 14,037 x 8
##    station period date       swath kelp_id hold_fast stipe_count  year
##      <dbl>  <dbl> <date>     <chr> <chr>       <dbl>       <dbl> <dbl>
##  1       1      1 1980-08-27 10R   1_2_111        NA          NA  1980
##  2       1      1 1980-08-27 10R   1_2_113        NA          NA  1980
##  3       1      1 1980-08-27 10R   1_2_125        NA          NA  1980
##  4       1      1 1980-08-27 10R   1_2_134        NA          NA  1980
##  5       1      1 1980-08-27 10R   1_2_141        NA          NA  1980
##  6       1      1 1980-08-27 10R   1_2_150        NA          NA  1980
##  7       1      1 1980-08-27 22R   1_4_112        NA          NA  1980
##  8       1      1 1980-08-27 22R   1_4_146        NA          NA  1980
##  9       1      1 1980-08-27 32L   1_5_77         NA          NA  1980
## 10       1      1 1980-08-27 32L   1_5_96         NA          NA  1980
## # … with 14,027 more rows
```

Modeling is yet to come. 

<!-- We would like the counts for species: -->

<!-- * Giant kelp = Macrocystis pyrifera -->
<!-- * Red sea urchin = Strongylocentrotus franciscanus -->
<!-- * Sheephead = Semicossyphus pulcher -->

<!-- 557	Macrocystis pyrifera <1m	Benthic density -->
<!-- 557	Macrocystis pyrifera <1m	Benthic cover -->
<!-- 589	Macrocystis pyrifera >1m	Benthic density -->
<!-- 589	Macrocystis pyrifera >1m	Benthic cover -->

<!-- 29	Strongylocentrotus franciscanus	Benthic density -->
<!-- 29	Strongylocentrotus franciscanus	Benthic cover -->

<!-- 1006	Semicossyphus pulcher (f)	Midwater fish density -->
<!-- 1008	Semicossyphus pulcher (m)	Midwater fish density -->

