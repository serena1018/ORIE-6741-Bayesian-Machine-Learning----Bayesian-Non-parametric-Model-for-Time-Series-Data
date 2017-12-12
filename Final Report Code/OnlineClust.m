%Yutao Han - Cornell University
%11.28.2017
%ORIE 6741 Project, Online Clustering with Chi-squared
%Well Log data
%%
close all;clear all;
%load pre-clustered data
data=csvread('data_final.csv');

%plot data
% figure
% plot(data(:,1),data(:,2),'b')

n_clust=max(data(:,3));%number of clusters
size_data=size(data);
n_pts=size_data(1);%number of points

colors=[1 0 0;0 1 0;0 0 1;1 0 1];%colors for each cluster

%determine where data cuts off for clustering purpose
cut_off=976;
% cut_off=n_pts;
figure
for i=1:cut_off
    for j=1:n_clust
        if data(i,3)==j%if the point belongs to jth cluster
        plot(data(i,1),data(i,2),'color',colors(j,:),'Marker','o','MarkerSize',5)
        end
        hold on
    end
end
%%
%optimize hyperparameters for RBF kernel
%divide into training and test data
train_data=data(1:cut_off,:);
test_data=data(cut_off+1:end,:);

%hyperparameter guesses
ell_init=50;
sf_init=1;
sigma_n_init=3;
log_hyp_init=[log(ell_init);log(sf_init);log(sigma_n_init)];
k=@covSEiso;%function handle for which kernel to use

%optimize hyperparameters for training data
for i=1:n_clust%iterate through each cluster
    
    %get x_opt and y_opt corresponding to that cluster
    idx=find(train_data(:,3)==i);
    x_opt=train_data(idx,1);
    y_opt=train_data(idx,2);
    
    %assume empirical mean (y-mux)
    [log_hyp_opt nlml_it num_it] = minimize(log_hyp_init,'ToyGP_negLogProb',-20000,x_opt,y_opt,k);
    hyp_opt=exp(log_hyp_opt);
    test_hyp_opt(:,i)=[i;hyp_opt];
    %check_params=log_hyp_init;
    %checkgrad('ToyGP_negLogProb', check_params, 1e-5,x_opt,y_opt,k)
end
%%
%optimize hyperparameters for SM kernel
%divide into training and test data
train_data=data(1:cut_off,:);
test_data=data(cut_off+1:end,:);

k=@covSMfast;%cov function handle

%optimize hyperparameters for training data
for i=1:n_clust%iterate through each cluster
    
    %get x_opt and y_opt corresponding to that cluster
    idx=find(train_data(:,3)==i);
    x_opt=train_data(idx,1);
    y_opt=train_data(idx,2);
    
    %fit spectral mixture kernel to the data
    [N,D]=size(x_opt);%Dimension of input "x"
    Q=10;%Number of Gaussian Mixture Components
    
    nlml=Inf;
    n_it=10;
    
    for j=1:n_it
        
        log_hyp_init=initSMhypers(Q,x_opt,y_opt);%initialise SM hyperparameters
        sigma_n_init=3.5;%intial guess for noise, need smarter way to initialize sigma_n_init
        %for non-noise models this should be close to zero
        log_hyp_init=[log_hyp_init; log(sigma_n_init)];
        
        shortrun=100;%short run function evaluations

        log_hyp_opt=minimize(log_hyp_init,'NLP_SM',-shortrun,x_opt,y_opt,k,Q,D);
        %log_hyp_opt=minimize(log_hyp_init,'NLP_SM_nonoise',-shortrun,x_opt,y_opt,k,Q,D);
        
        %function [nlml, dnlml] = NLP_SM_nonoise(log_hyp,x,y,k,Q,D)
        %[nlml_new, dnlml] = NLP_SM_nonoise(log_hyp_opt,x_opt,y_opt,k,Q,D);
        %function [nlml, dnlml] = NLP_SM(log_hyp,x,y,k,Q,D)
        [nlml_new, dnlml]=NLP_SM(log_hyp_opt,x_opt,y_opt,k,Q,D);
        
        if (nlml_new < nlml)%replace the initialized hyperparameters, if better nlml achieved
            log_hyp_proc = log_hyp_opt;
            nlml = nlml_new;
        end       
        print_it=[i j]
    end
    
    %after initializing hyperparameters, marginalize over them
    longrun=1000;
    %log_hyp_opt=minimize(log_hyp_proc,'NLP_SM_nonoise',-longrun,x_opt,y_opt,k,Q,D);
    log_hyp_opt=minimize(log_hyp_proc,'NLP_SM',-longrun,x_opt,y_opt,k,Q,D);
    log_hyp_save(:,i)=log_hyp_opt;
end
%%
%given optimized hyperparameters for each cluster
%cluster every 10 new points in test_data
size_test=size(test_data);
n_test=size_test(1);
size_train=size(train_data);
n_train=size_train(1);

%size of the window is 10 pts
size_w=10;
n_w=floor(n_test/size_w);%number of size_w point intervals to be clustered online

mod_test=mod(n_test,size_w);%leftover points

test_data_clust=test_data;%data that is clustered

for i=1:n_w
    %points to be clustered
    xstar=test_data(size_w*i-size_w+1:size_w*i,1);
    ystar=test_data(size_w*i-size_w+1:size_w*i,2);
%     xstar=data(:,1);
%     ystar=data(:,2);
    
    E_save=[];
    SD_save=[];
    for j=1:n_clust%iterate through each cluster
        %get x_opt and y_opt corresponding to that cluster
        idx=find(train_data(:,3)==j);
        x_opt=train_data(idx,1);
        y_opt=train_data(idx,2);
        
        %for RBF
        %hyp_j=test_hyp_opt(2:end,j);%hyperparameters of jth cluster
        %log_hyp_j=log(hyp_j);%log hyperparameters
        
        %for SM
        [N,D]=size(x_opt);%Dimension of input "x"
        Q=10;%Number of Gaussian Mixture Components
        %now given final optimized hyperparameters, do prediction
        %find E, and Cov
        log_hyp_opt=log_hyp_save(:,j);
        %[E,Cov]=SM_pred_nonoise(log_hyp_opt,x_opt,y_opt,x_star,k,Q,D);
        [E,Cov]=SM_pred(log_hyp_opt,x_opt,y_opt,xstar,k,Q,D);
        
        %for RBF
        %expected value and covariance of y(xstar) given GP
        %[E,Cov]=OnlineClust_E_Cov(log_hyp_j,x_opt,y_opt,xstar,ystar,k);
        %noise of f_dstar 
        %sigma_n=hyp_j(3);
        
        %for SM
        sigma_n=exp(log_hyp_opt(end));
        
        %dimension of measurement vector
        dim_star=length(E);
        %covariance of test statistic
        Cov_ts=Cov+sigma_n^2*eye(dim_star);
        Cov_ts=Cov_ts+1e-6*eye(dim_star);%add jitter
        %function X=cholesky_sol(A,B)
        inv_Cov_ts=cholesky_sol(Cov_ts,eye(dim_star));
        %test statistic
        ts=(ystar-E)'*inv_Cov_ts*(ystar-E);
        %p=chi2cdf(x,v,'upper')
        S=chi2cdf(ts,dim_star,'upper');   
        
        %note: S is always positive and represents the probability of
        %observing ystar given the underlying GP distribution, the larger S
        %is the more likely that ystar was generated given E,Cov
        
        %save E and SD for debugging
        SD=sqrt(diag(Cov));
        E_save(:,j)=E;
        SD_save(:,j)=SD;
        
        %save the likelihood
        L(j,:)=[j log(S)];
    end
    
    %assign test_data_clust data to cluster with highest likelihood
    [val_max,idx_max]=max(L(:,2));
    test_data_clust(size_w*i-size_w+1:size_w*i,3)=idx_max;
    
    %predict future points and save for plotting

end
%%
close all

figure
for i=1:n_train
    for j=1:n_clust
        if train_data(i,3)==j%if the point belongs to jth cluster
        plot(train_data(i,1),train_data(i,2),'color',colors(j,:),'Marker','o','MarkerSize',5)
        end
        hold on
    end
end

% for i=1:length(xstar)
% plot(xstar(i),ystar(i),'k','Marker','o','MarkerSize',5)
% hold on
% end
%plot E_save and SD_save
for i=1:n_clust
    plot(xstar,E_save(:,i),'color',colors(i,:),'LineWidth',3)
    
    hold on
end
%%
close all
test_plot=[train_data;test_data_clust];
figure
for i=1:n_pts
    for j=1:n_clust
        if test_plot(i,3)==j%if the point belongs to jth cluster
        plot(test_plot(i,1),test_plot(i,2),'color',colors(j,:),'Marker','o','MarkerSize',5)
        end
        hold on
    end
end
%%
close all
figure
for i=1:n_train
    for j=1:n_clust
        if train_data(i,3)==j%if the point belongs to jth cluster
        plot(train_data(i,1),train_data(i,2),'color',colors(j,:),'Marker','o','MarkerSize',5)
        end
        hold on
    end
end
for i=1:n_w
    for j=1:n_clust
        if test_data_clust(size_w*i,3)==j
        plot(test_data_clust(size_w*i-size_w+1:size_w*i,1),test_data_clust(size_w*i-size_w+1:size_w*i,2),...
            'color',colors(j,:),'Marker','o','MarkerSize',5,'LineStyle','none')
        end
        hold on
    end
    pause(.3)
end

