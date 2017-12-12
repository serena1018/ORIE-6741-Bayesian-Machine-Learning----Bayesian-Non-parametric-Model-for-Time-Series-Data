%Yutao Han - Cornell University
%12.11.2017
%ORIE 6741
%%
%PLEASE READ

%This script recreates figures 4(d), 4(e), 6(b) in the final report, code is borrowed
%from Jurgen Van Gael's iHMM library as discussed in section 4.1, code is
%also borrowed from  Carl Edward Rasmussen's code for GPML.

%Note that hyperparameters need to be tuned depending on the cut_off

%%
%This section generates figures 4(d), 4(e), and 6(b)

% iHMM Gibbs sampler with normal output.
% For comparing iHMM to IBP
clear all; close all;

% Load data for training
Y = csvread('wellLogDatacsv.csv');
%Y = csvread('BitCoin_train.csv');
%Actual cluster
ACT = csvread('data_final.csv');

cut_off=500; % This determines the number of training points

%Note that hyperparameters need to be tuned depending on the cut_off

% For comparing to K Mean clustering 
Z = Y(1:cut_off,:);
idx = kmeans(Z(:,2),3);
% Note that this is not a really fare comparison since we need number of
% cluster from IBP to use in K mean clustering
[idx,C] = kmeans(Z(:,2),3);
scatter(Z(:,1) , Z(:,2) ,10 , idx)

%Train only on first 500 points
Y = Y(1:cut_off,:);
%plot(Y(:,2))
%plot(Y(1:500,1) , Y(1:500,2) )

%Formatting data set
Y = Y(:,2);
Y = Y';

T = cut_off;                             % Length of HMM (N)
K = 10; % 10                         % Random guess of number of states 

%Initialize hyper parameter
%The following set of hyperparameters work well for 500 data points, which
%is used for figure 4(e)
hypers.alpha0_a = 6; % 5 for well log
hypers.alpha0_b = 5; % 5 for well log
hypers.gamma_a = 10; %10 
hypers.gamma_b = 10; %10
hypers.sigma2 =  std(Y); % 6
hypers.mu_0 = 1; % 1
hypers.sigma2_0 = std(Y); % 6

%The following set of hyperparameters work well for 900 data points, which
%is used for figure 6(b)
% hypers.alpha0_a = 9; % 5 for well log
% hypers.alpha0_b = 9; % 5 for well log
% hypers.gamma_a = 10; %10 
% hypers.gamma_b = 10; %10
% hypers.sigma2 =  std(Y) % 6
% hypers.mu_0 = 1; % 1
% hypers.sigma2_0 = std(Y); % 6


% On average K should be 3 some run might result in different result
tic
[S, stats] = iHmmNormalSampleGibbs(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 3));
toc

%Plot key result from MCMC
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

S{1}.S';
seq = 1:size(Y',1);
K2 = [seq' , Y' , S{1}.S'];
%Target = K2

final = [K2 , ACT(1:cut_off,3) ];
% ACC IHMM
final = [final , idx];

%Plot result from ihmm
subplot(2,1,1)
scatter(final(:,1) , final(:,2) ,10 , final(:,3))
title('4(d): Infinite Hidden Markov Model')

%Plot result from Kmean
subplot(2,1,2)
scatter(final(:,1) , final(:,2) ,10 , final(:,5))
title('4(e) and 6(b): K-Means Clustering')
%Comparing to IBP, though the number of cluster found is the same
%How ihmm and k mean group data point is more intuitive and easier to
%understand
csvwrite('final_result.csv', final)
