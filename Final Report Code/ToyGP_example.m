%Yutao Han - Cornell University
%10.26.2017
%ORIE 6741 Project, Toy GP Clustering
%try to cluster simple time-series with GP
%%
close all;clear all;clc;
%generate and plot data, also randomly initiate
n_clust=3;%number of clusters
n_clust_init=n_clust;

% p_1=.33;
% a_1=2;
% x_1=1:38;
% y_1=a_1*sin(p_1*x_1);
% 
% x_2=x_1(end)+1:x_1(end)+19;
% y_2=zeros(1,19);

% figure
% subplot(2,1,1)
% plot(x_1,y_1,'ob')
% hold on
% plot(x_2,y_2,'or')

% x=[x_1';x_2'];
% y=[y_1';y_2'];
% data=[randi(n_clust,length(x),1) x y];

load 'WellLogData.txt'

n_pts=500;

data=[randi(n_clust_init,n_pts,1) WellLogData(1:n_pts,:)];

% subplot(2,1,2)
colors=[1 0 0;0 1 0;0 0 1;1 0 1];
for i=1:n_pts
    for j=1:n_clust_init
        if data(i,1)==j
            plot(data(i,2),data(i,3),'marker','o','color',colors(j,:));
        end
    end
    hold on
end
title('ground truth and randomly initialized clusters')

data_init=data;
%%
%GP clustering
%using RBF for now
%assuming no noise

num_it=1;%number of iterations

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
        
        %include euclidean distance metrix
        %thresh=25;
        %[cdf_scale]=ToyGP_thresh(x_opt,y_opt,xstar,ystar,thresh);
        
        cdf_save(j,:)=[j log(cdf_point)]; 
        %cdf_save(j,:)=[j log(cdf_scale) + log(cdf_point)];   
    end
    
    %assign data point to cluster with highest likelihood
    [max_cdf_val,max_cdf_idx]=max(cdf_save(:,2));
    data(i,1)=cdf_save(max_cdf_idx,1);
    
   
    num=[n i]
end

end
%%
close all;
%plot
figure
subplot(2,1,1)
for i=1:n_pts
    for j=1:n_clust
        if data_init(i,1)==j
            %%plot(data_init(i,2),data_init(i,3),'marker','.','MarkerSize',5,'color',colors(j,:));
            plot(data_init(i,2),data_init(i,3),'.b','MarkerSize',5);
        end
    end
    hold on
end
%%
close all;
subplot(2,1,1)
plot(data(:,2),data(:,3),'b')
subplot(2,1,2)
for i=1:n_pts
    for j=1:n_clust
        if data(i,1)==j
            plot(data(i,2),data(i,3),'marker','.','MarkerSize',5,'color',colors(j,:));
        end
    end
    hold on
end
%%
close all;
subplot(2,1,1)
plot(data(:,2),data(:,3),'b')
%%
subplot(3,1,3)
%check posterior draws for the functions
%for each cluster optimize hyperparameters and get posterior draw
for i=1:500
    for j=1:4
        if data(i,1)==j
            plot(data(i,2),data(i,3),'marker','o','color',colors(j,:));
        end
    end
    hold on
end
for i=1:4
    data_clust=[];
    %idx of cluster wrt data
    clust_idx=find(data(:,1)==i);
    for k=1:length(clust_idx)
        data_clust(k,:)=data(clust_idx(k),2:3);
    end
    %optimize hyperparameters here
    x_opt=data_clust(:,1);
    y_opt=data_clust(:,2);
    ell_init=3;
    sf_init=std(y_opt);
    hyp=[log(ell_init);log(sf_init)];
    k=@covSEiso;%function handle for which kernel to use
    [nlml, dnlml] = ToyGP_negLogProb(hyp,x_opt,y_opt,k);
    
    [log_hyp_opt nlml_it num_it] = minimize(hyp,'ToyGP_negLogProb',-20000,x,y,k);
    
    hyp_opt=exp(log_hyp_opt);
    
    xstar=linspace(data_init(1,2),data_init(end,2),50)';
    [E,sd]=ToyGP_posterior(hyp_opt(1),hyp_opt(2),x_opt,y_opt,xstar);
    
    plot(xstar,E,'color',colors(i,:))
    hold on
    plot(xstar,E+2*sd,'color',colors(i,:))
    hold on
    plot(xstar,E-2*sd,'color',colors(i,:))
    hold on
end
ylim([-2 2])
title('randomly initialized clusters and clustered data')