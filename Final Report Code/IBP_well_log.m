%Yutao Han - Cornell University
%12.11.2017
%ORIE 6741
%%
%PLEASE READ

%This script recreates figure 4(c) from the final report. The IBP is first
%used to discover the number of clusters and then GP with Gibbs Sampling is
%used to assign the data points to clusters. Some code is borrowed from
%Ilker Yildirim's implementation of the IBP and Carl Edward Rasmussen's
%code for GPML.

%The first 500 points from wellLogData are used.

%%
%This section finds the number of clusters with the IBP

%This section might take a while depending on your machine

% This is an implementation of the algorithm described in the Computational
% cognition cheat sheet titied "The Indian Buffet Process."
% Original Written by Ilker Yildirim, September 2012.

close all;clear all;

randn('seed', 1); 
%Load data set
dat = csvread('wellLogData.csv');
acc = dat(1:500,2);
num_objects = 500; 
object_dim = 1;
%Use empirical data to initiate parameter
X = acc;
sigma_x_orig = var(X);
seq = 1:500;

% Compute Harmonic number for N.
%N = maxNumCompThreads
% For testing on the server use multiple core 
%LASTN = maxNumCompThreads(10)
%LASTN = maxNumCompThreads('automatic')
HN = 0;
for i=1:num_objects HN = HN + 1/i; end
E = 1000;
BURN_IN = 200;
SAMPLE_SIZE = 500;

% Initialize the chain.
sigma_A = 1;
sigma_X = var(X);
alpha = 1;
% Truncation level is necessary as described in the paper (to avoid K go to
% infinity)
K_inf = 5;

%Prior over matrix Z
[Z K_plus] = sampleIBP(alpha, num_objects);
chain.Z = zeros(SAMPLE_SIZE,num_objects,K_inf);
chain.K = zeros(SAMPLE_SIZE, 1);
chain.sigma_X = zeros(SAMPLE_SIZE, 1);
chain.sigma_A = zeros(SAMPLE_SIZE, 1);
chain.alpha = zeros(SAMPLE_SIZE,1);
s_counter = 0;

for e=1:E
    % Store samples after the BURN-IN period.
    if (e > BURN_IN)
        s_counter = s_counter+1;
        chain.Z(s_counter,:,1:K_plus) = Z(:,1:K_plus);
        chain.K(s_counter) = K_plus;
        chain.sigma_X(s_counter) = sigma_X;
        chain.sigma_A(s_counter) = sigma_A;
        chain.alpha(s_counter) = alpha;
    end
    disp(['At iteration ', num2str(e), ': K_plus is ', num2str(K_plus), ', alpha is ', num2str(alpha)]);

    for i=1:num_objects
        % The matrix M will be handy for future likelihood and matrix
        % inverse computations.
        M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
        for k=1:K_plus
            % That can happen, since we may decrease K_plus inside.
            if (k>K_plus)
                break;
            end
            if Z(i,k) > 0
                % Take care of singular features
                if sum(Z(:,k)) - Z(i,k) <= 0
                    Z(i,k) = 0;
                    Z(:,k:K_plus-1) = Z(:,k+1:K_plus);
                    K_plus = K_plus-1;
                    M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + ...
                        (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
                    continue;
                end
            end
            
            % This equations are for computing the inverse efficiently.
            % It is an implementation of the trick from Griffiths and
            % Ghahramani (2005; Equations 51 to 54). 
            M1 = calcInverse(Z(:,1:K_plus), M, i, k, 1);
            M2 = calcInverse(Z(:,1:K_plus), M, i, k, 0);

            % Compute conditional distributions for the current cell in Z.
            Z(i,k) = 1;
            P(1) = likelihood(X, Z(:,1:K_plus), M1, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(sum(Z(:,k))- Z(i,k)) -log(num_objects);

            Z(i,k) = 0;
            P(2) = likelihood(X, Z(:,1:K_plus), M2, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(num_objects - sum(Z(:,k))) - log(num_objects);
            P = exp(P - max(P));
            
            % Sample from the conditional.
            if rand < P(1)/(P(1)+P(2))
                Z(i,k) = 1;
                M = M1;
            else
                Z(i,k) = 0;
                M = M2;
            end
        end
        % Sample the number of new dishes for the current object.
        %if K_plus <= K_inf
        %disp('Samling New Dishes'); 
        trun = zeros(1,2);
        alpha_N = alpha / num_objects;
        %This is basically the liklihood of the new dish
        for k_i=0:1
            
            Z(i,K_plus+1:K_plus+k_i) = 1;
            
            M = (Z(:,1:K_plus+k_i)'*Z(:,1:K_plus+k_i) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus+k_i,1)))^-1;
            
            trun(k_i+1) = k_i*log(alpha_N) - alpha_N - log(factorial(k_i)) + ...
                likelihood(X, Z(:,1:K_plus+k_i), M, sigma_A, sigma_X, K_plus+k_i, num_objects, object_dim);
        end
        
        % This is an odd part of the code -- can modify later 
        % Normally for object recognition , we will look at infinite K
        % In our case, we should have limit on K
        Z(i,K_plus+1:K_plus+2) = 0;
        trun = exp(trun - max(trun));
        trun = trun/sum(trun);
        p = rand;
        t = 0;
        for k_i=0:1
            t = t+trun(k_i+1);
            if p < t
                new_dishes = k_i;
                break;
            end
        end
        if K_plus <= K_inf
            Z(i,K_plus+1:K_plus+new_dishes) = 1;
            %disp([num2str(new_dishes)]);
            K_plus = K_plus + new_dishes;
        end
    end
     % End of odd part
     
    % Metropolis steps for sampling sigma_X and sigma_A
    %Here noise is Uniform (-5,5) --> Can also sample from noise
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    
    l_curr = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);

    if rand < .5
        pr_sigma_X = sigma_X - rand/20;
    else
        pr_sigma_X = sigma_X + rand/20;
    end
    
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (pr_sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_X = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, pr_sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_X = exp(min(0, l_new_X - l_curr));

    if rand < .5
        pr_sigma_A = sigma_A - rand/20;
    else
        pr_sigma_A = sigma_A + rand/20;
    end
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/pr_sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_A = likelihood(X, Z(:,1:K_plus+new_dishes), M, pr_sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_A = exp(min(0, l_new_A - l_curr));

    if rand < acc_X
        sigma_X = pr_sigma_X;
    end
    if rand < acc_A
        sigma_A = pr_sigma_A;
    end

    % Sample alpha from its conditional posterior.
    % Posterior according to Zoubin
    alpha = mygamrnd(1+K_plus, 1/(1+HN),1);
    
    % Save the chain at every 1000th iteration.
    if mod(e,1000) == 0
        s = strcat('chain_ibp_',num2str(e));
        save(s, 'chain');
    end
end

post_Z  = [acc , Z];
%Frequency table 
sum(Z);
sum(Z,2);
% Take the median from MCMC as number of cluster
%Plot latent feature based on cluster
C1 = post_Z(post_Z(:,2) == 1 , 1);
C2 = post_Z(post_Z(:,3) == 1 , 1);
C3 = post_Z(post_Z(:,4) == 1 , 1);
C4 = post_Z(post_Z(:,5) == 1 , 1);
figure
subplot(2,2,1)
plot(seq(1:size(C1,1)) , C1)
subplot(2,2,2)
plot(seq(1:size(C2,1)) , C2)
subplot(2,2,3)
plot(seq(1:size(C3,1)) , C3)
subplot(2,2,4)
plot(seq(1:size(C4,1)) , C4)

%Since IBP assume independent of column (feature) and one data point can
%can belong to multiple cluster , we decide to take the number of cluster
%For assigning point to cluster, we use the GP+Gibbs sampling process in the
%next section

number_of_cluster = median(chain.K);

%%
%This section clusters the data with GP+Gibbs sampling, this method uses
%the RBF kernel for clustering with Gibbs sampling
close all;
%generate and plot data, also randomly initiate
n_clust=number_of_cluster;%number of clusters
n_clust_init=n_clust;

load 'WellLogData.txt'

n_pts=num_objects;

%randomly initialize data points with the number of clusters
data=[randi(n_clust_init,n_pts,1) WellLogData(1:n_pts,:)];
data_init=data;

num_it=5;%number of iterations to run the gibbs sampler

for n=1:num_it
%gibbs sampling
for i=1:n_pts%iterate through each data point
    
    %remove point i from any clusters
    data(i,1)=0;
    %number of clusters
    n_clust=numel(unique(data(:,1)))-1;
    
    cdf_save=[];
    for j=1:n_clust%interate through each cluster
        
        data_clust=[];
        %idx of cluster wrt data
        clust_idx=find(data(:,1)==j);
        for k=1:length(clust_idx)
            data_clust(k,:)=data(clust_idx(k),2:3);
        end
        %optimize hyperparameters here
        x_opt=data_clust(:,1);
        y_opt=data_clust(:,2);
        ell_init=75;
        sf_init=5;
        sigma_n_init=5;
        hyp=[log(ell_init);log(sf_init);log(sigma_n_init)];
        k=@covSEiso;%function handle for which kernel to use
        [nlml, dnlml] = ToyGP_negLogProb(hyp,x_opt,y_opt,k);
        
        [log_hyp_opt nlml_it num_it] = minimize(hyp,'ToyGP_negLogProb',-20000,x_opt,y_opt,k);

        hyp_opt=exp(log_hyp_opt);
        
        %check_params=[log(ell_init);log(sf_init)];
        %checkgrad('ToyGP_negLogProb', check_params, 1e-5,x_opt,y_opt,k)
        
        %now given those hyperparameters, find posterior for training point
        xstar=data(i,2);
        ystar=data(i,3);
        
        [cdf_point]=ToyGP_cdf(hyp_opt(1),hyp_opt(2),hyp_opt(3),x_opt,y_opt,xstar,ystar);

        
        cdf_save(j,:)=[j log(cdf_point)]; 
    end
    
    %assign data point to cluster with highest likelihood
    [max_cdf_val,max_cdf_idx]=max(cdf_save(:,2));
    data(i,1)=cdf_save(max_cdf_idx,1);
    
   
    num=[n i]
end

end
%%
%plot the results
colors=[1 0 0;0 1 0;0 0 1;1 1 0;1 0 1];

subplot(2,1,1)
plot(data(:,2),data(:,3),'b')
title('Raw training data')
subplot(2,1,2)
for i=1:n_pts
    for j=1:n_clust
        if data(i,1)==j
            plot(data(i,2),data(i,3),'marker','.','MarkerSize',5,'color',colors(j,:));
        end
    end
    hold on
end
title('4(c): GP and Gibbs sampling results')

