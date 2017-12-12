%Yutao Han - Cornell University
%11.17.2017
%prediction with SM Kernel

function [E,Cov]=SM_pred(log_hyp,x,y,xstar,k,Q,D)

mux=mean(y);%empirical mean
%unpack hyperparamters
log_hyp_input=log_hyp(1:Q+2*Q*D);
sigma_n=exp(log_hyp(end));%extract noise sigma_n

%function K = covSMfast(Q, hyp, x, z, i)
Kxx = feval(k, Q, log_hyp_input, x, x);  
Kxstarx= feval(k, Q, log_hyp_input, xstar, x);  
Kxstarxstar = feval(k, Q, log_hyp_input, xstar, xstar);  

M=Kxx+sigma_n^2*eye(length(y));%accounting for noise
M=M+1e-6*eye(length(y));%add jitter
R_M=chol(M);
inv_M=solve_chol(R_M,eye(length(y)));%inverse of M

% E=Kxstarx*inv_M*y;
E=mux+Kxstarx*inv_M*(y-mux);
Cov=Kxstarxstar-Kxstarx*inv_M*Kxstarx';
end