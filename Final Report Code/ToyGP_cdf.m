%Yutao Han - Cornell University
%10.26.2017
%ORIE 6741 Project, Toy DPGP Clustering
%find likelihood

function [cdf_point]=ToyGP_cdf(l,a,sigma_n,x,y,xstar,ystar)

%using covSEiso

%empirical mean
emp_mean=mean(y);
emp_mean_star=mean(ystar);

%hyperparameters
hyp=[log(l);log(a)];
%Kxx
K_xx = covSEiso(hyp, x, x);
%Kx*x
K_xstarx = covSEiso(hyp, xstar, x);
%Kx*x*
K_xstarxstar = covSEiso(hyp, xstar, xstar);

K_xx=K_xx+sigma_n^2*eye(length(y));
K_xx=K_xx+1e-6*eye(length(y));
R=chol(K_xx);
inv_K_xx=solve_chol(R,eye(length(y)));

E=emp_mean_star+K_xstarx*inv_K_xx*(y-emp_mean);
%E=K_xstarx*inv_K_xx*y;
cov=K_xstarxstar-K_xstarx*inv_K_xx*K_xstarx';
sd=sqrt(diag(cov));

%p = normcdf(x,mu,sigma)
cdf_point=normcdf(ystar,E,sd);
if cdf_point>.5
    cdf_point=1-cdf_point;
end
cdf_point=2*cdf_point;%2 tailed hypothesis test
end