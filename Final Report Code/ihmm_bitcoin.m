%Yutao Han - Cornell University
%12.11.2017
%ORIE 6741
%%
%PLEASE READ

%This script recreates figure 5(b) in the final report, code is borrowed
%from Jurgen Van Gael's iHMM library as discussed in section 4.1

%%
% iHMM Gibbs sampler with normal output.
% For comparing iHMM to IBP
close all; clear all;
K = 5; % 10                         % Random guess of number of states 
Y = csvread('BitCoin_train.csv');

%plot data
plot(Y)
%For formatting
Y = Y';

T = 600; % Length of HMM
hypers.alpha0_a = 4; % 5 for well log
hypers.alpha0_b = 4; % 5 for well log
hypers.gamma_a = 5; %10 
hypers.gamma_b = 5; %10
hypers.sigma2 =  std(Y); %6
hypers.mu_0 = 1; % 1
hypers.sigma2_0 = std(Y); % 6

%On average cluster found = 3 
tic
[S, stats] = iHmmNormalSampleGibbs(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 3));
toc

S{1}.S';
seq = 1:size(Y',1);
K2 = [seq' , Y' , S{1}.S'];


%plot results
colors=[1 0 0;0 1 0;0 0 1;1 1 0;1 0 1];
size_K2=size(K2);
n_pts=size_K2(1);
n_clust=numel(unique(K2(:,3)));

%Chart in the paper

% figure
% for i=1:n_pts
%     for j=1:n_clust
%         if K2(i,3)==j
%             plot(K2(i,1),K2(i,2),'.','MarkerSize',7,'color',colors(j,:))
%             hold on
%         end
%     end
% end
%Chart in the paper
scatter(K2(:,1) , K2(:,2) ,10 , K2(:,3))
title('5(b): Infinite Hidden Markov Model')
