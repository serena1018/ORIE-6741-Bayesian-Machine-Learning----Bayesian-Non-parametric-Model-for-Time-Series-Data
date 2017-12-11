% Test the iHMM Gibbs sampler with normal output.
clear;
T = 1267;                        % Length of HMM
K = 10;                          % Random guess of number of states 

stream = RandStream('mt19937ar','seed',21);
% RandStream.setDefaultStream(stream);

% Sample states using the iHmm Gibbs sampler.
Y = csvread('wellLogData.csv');
Y = Y(:,2);
Y = Y'

hypers.alpha0_a = 5;
hypers.alpha0_b = 5;
hypers.gamma_a = 10;
hypers.gamma_b = 10;
hypers.sigma2 =  6;
hypers.mu_0 = 1;
hypers.sigma2_0 = 6;

tic
[S, stats] = iHmmNormalSampleGibbs(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 10));
toc

S{1}.S'
seq = 1:size(Y',1)
K2 = [seq' , Y' , S{1}.S']
scatter(K2(:,1) , K2(:,2) , [] , K2(:,3))

figure(1)

subplot(2,2,1)
hist(stats.K)
title('K')

subplot(2,2,2)
plot(stats.jll)
title('Joint Log Likelihood')

subplot(2,2,3)
plot(stats.alpha0)
title('alpha0')

subplot(2,2,4)
plot(stats.gamma)
title('gamma')

%subplot(3,2,5)
%imagesc(SampleTransitionMatrix(S{1}.S, zeros(1,S{1}.K))); colormap('Gray');
%title('Transition Matrix')