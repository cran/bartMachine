\name{pd_plot}
\alias{pd_plot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Partial Dependence Plot
}
\description{
Creates a partial dependence plot for a BART model for regression or classification.
}
\usage{
pd_plot(bart_machine, j, 
levs = c(0.05, seq(from = 0.1, to = 0.9, by = 0.1), 0.95), 
lower_ci = 0.025, upper_ci = 0.975, prop_data = 1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{j}{
The number or name of the column in the design matrix for which the partial dependence plot is to be created. 
}
  \item{levs}{
Quantiles at which the partial dependence function should be evaluated. Linear extrapolation is performed between these points.
}
  \item{lower_ci}{
Lower limit for credible interval
}
  \item{upper_ci}{
Upper limit for credible interval
}
  \item{prop_data}{
 The proportion of the training data to use. Default is 1. Use a lower proportion for speedier pd_plots. The closer to 1, the more resolution
 the PD plot will have; the closer to 0, the lower but faster.
}
}
\details{
For regression models, the units on the y-axis are the same as the units of the response. For classification models, the units on the y-axis are probits. 
}
\value{
Invisibly, returns a list with the following components:

  \item{x_j_quants}{Quantiles at which the partial dependence function is evaluated}
  \item{bart_avg_predictions_by_quantile_by_gibbs}{All samples of \eqn{\hat{f}(x)}}
  \item{bart_avg_predictions_by_quantile}{Posterior means for \eqn{\hat{f}(x)} at \code{x_j_quants}}
  \item{bart_avg_predictions_lower}{Lower bound of the desired confidence of the credible interval of \eqn{\hat{f}(x)}}
  \item{bart_avg_predictions_upper}{Upper bound of the desired confidence of the credible interval of \eqn{\hat{f}(x)}}
  \item{prop_data}{The proportion of the training data to use as specified when this function was executed}
%% ...
}
\references{
Adam Kapelner, Justin Bleich (2016). bartMachine: Machine Learning
with Bayesian Additive Regression Trees. Journal of Statistical
Software, 70(4), 1-40. doi:10.18637/jss.v070.i04

HA Chipman, EI George, and RE McCulloch. BART: Bayesian Additive Regressive Trees.
The Annals of Applied Statistics, 4(1): 266--298, 2010.
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
This function is parallelized by the number of cores set in \code{\link{set_bart_machine_num_cores}}.
}


\examples{
\dontrun{
#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = bartMachine(X, y)

#partial dependence plot for quadratic term
pd_plot(bart_machine, "X3")


#Classification example

#get data and only use 2 factors
data(iris)
iris2 = iris[51:150,]
iris2$Species = factor(iris2$Species)

#build BART classification model
bart_machine = bartMachine(iris2[ ,1:4], iris2$Species)

#partial dependence plot 
pd_plot(bart_machine, "Petal.Width")
}


}
