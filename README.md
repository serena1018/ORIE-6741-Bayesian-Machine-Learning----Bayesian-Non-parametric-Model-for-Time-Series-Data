# ORIE-6741 Bayesia Machine Learning Project
# Predictive Model for Time-Series Data with Bayesian Non-parametrics
Chawannut Prommin (cp626@cornell.edu), Serena Li (sl2327@cornell.edu), Yutao Han (yh675@cornell.edu)
# Paper Structure
## Abstract
We propose a novel Bayesian non-parametric
framework for time-series data modeling with
pattern discovery and online inference. We experiment
with using the Indian Buffet Process
and the infinite Hidden Markov Model for automatic
pattern or cluster discovery. Our model
then uses a novel framework for online inference
of the time-series data using Gaussian Process regression
with a Spectral Mixture kernel function
and hypothesis testing. We consider the scalability
of the model during online inference due
to evaluation of the clusters rather than the entire
dataset.
## Introduction
## Related Work
## Methodology
* Contribution
* Indian Buffet Process (IBP) Discovery of Number of Clusters
* Infinite Hidden Markov Model (iHMM) Clustering
* Spectral Mixture (SM) Kernel Learning
* Clustering Time-Series Data
* Online Inference
* Fast Inference
## Experimental Results
* Time-Series Clustering
* Kernel Learning and Chi-squared Testing
* Evaluation of Model Accuracy
## Discussion
## References

# Code Structure
## Description of scripts to run:
### The MATLAB files below do not include implementation of KISS-GP fast inference which is done in the Python files with GPyTorch. The MATLAB files essentially include all the algorithms except for KISS-GP implementation.  
* IBP_well_log.m

This script recreates figure 4(c) from the final report. The IBP is first used to discover the number of clusters and then GP with Gibbs Sampling is used to assign the data points to clusters. Some code is borrowed from Ilker Yildirim's implementation of the IBP and Carl Edward Rasmussen's code for GPML. The first 500 points from wellLogData are used.

* ihmm_bitcoin.m

This script recreates figure 5(b) in the final report, code is borrowed from Jurgen Van Gael's iHMM library as discussed in section 4.1

* ihmm_well_log.m

This script recreates figures 4(d), 4(e), 6(b) in the final report, code is borrowed from Jurgen Van Gael's iHMM library as discussed in section 4.1, code is also borrowed from  Carl Edward Rasmussen's code for GPML.
Note that hyperparameters need to be tuned depending on the cut_off

* OnlineClust2.m

This script recreates figures 6(c) and 6(d) in the final report, code is also borrowed from  Carl Edward Rasmussen's code for GPML.
Depeding on the length of the training data, the hyperparameters need to be tuned in ihmm_well_log to achieve good clustering.
This script reads already clustered data from data_final.csv to use as the training data.
Please note the separate sections for implementing the spectral mixture (SM) kernel versus the RBF kernel, use the correct sections for
which results are being replicated.
Note that hyperparameters need to be tuned depending on the cut_off.

### KISS-GP fast inference jupyter notebook, note the the online clustering method used is different from section 3.6, because the framework of  KISS-GP in GPyTorch does not allow extra parameters.
*  online_kissGP+SpectralMixtureKernel.ipynb

Preran jupyter notebook, if you wish to run the code please make sure you have Pytorch and Gpytorch installed on your local machine.

