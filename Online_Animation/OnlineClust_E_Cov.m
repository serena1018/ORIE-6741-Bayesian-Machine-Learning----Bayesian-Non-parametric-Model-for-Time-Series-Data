%Yutao Han - Cornell University
%10.26.2017
%ORIE 6741 Project, OnlineClust
%find the expected value, and variance for GP

function [E,Cov]=OnlineClust_E_Cov(log_hyp,x,y,xstar,ystar,k)

%empirical mean
emp_mean=mean(y);
emp_mean_star=mean(ystar);

%hyperparameters
ell=exp(log_hyp(1));
sf=exp(log_hyp(2));
sigma_n=exp(log_hyp(3));

hyp=[log(ell);log(sf)];

%Kxx
%K_xx = covSEiso(hyp, x, x);
K_xx = feval(k, hyp, x, x); 
%Kx*x
%K_xstarx = covSEiso(hyp, xstar, x);
K_xstarx = feval(k, hyp, xstar, x); 
%Kx*x*
%K_xstarxstar = covSEiso(hyp, xstar, xstar);
K_xstarxstar=feval(k, hyp, xstar, xstar);

K_xx=K_xx+sigma_n^2*eye(length(y));
K_xx=K_xx+1e-6*eye(length(y));
R=chol(K_xx);
inv_K_xx=solve_chol(R,eye(length(y)));

E=emp_mean+K_xstarx*inv_K_xx*(y-emp_mean);
%E=K_xstarx*inv_K_xx*y;
Cov=K_xstarxstar-K_xstarx*inv_K_xx*K_xstarx';


end