%Yutao Han - Cornell University
%11.17.2017
%negLogProb for SM Kernel

function [nlml, dnlml] = NLP_SM(log_hyp,x,y,k,Q,D)

mux=mean(y);%emprical mean
%function K = covSMfast(Q, hyp, x, z, i)
%for function
z=x;

hyp_input=log_hyp(1:Q+2*Q*D);
sigma_n=exp(log_hyp(end));%extract noise sigma_n

K = feval(k, Q, hyp_input, x, z); %evaluate K

M=K+sigma_n^2*eye(length(y));%accounting for noise
M=M+1e-6*eye(length(y));%add jitter
R_M=chol(M);
inv_M=solve_chol(R_M,eye(length(y)));%inverse of M
log_det_M=2*sum(log(diag(R_M)));%log determinant of M

%lml=-y'*inv_M*y-log_det_M;
lml=-(y-mux)'*inv_M*(y-mux)-log_det_M;
nlml=-lml;%negative log likelihood

for i=1:Q+2*Q*D%for non-noise hyperparameters
dK_dlog_hyp = feval(k, Q, hyp_input, x, z, i); 
%positive derivitive of log likelihood wrt log hyperparameters
%dlml=-y'*(-inv_M*dK_dlog_hyp*inv_M)*y-trace(inv_M*dK_dlog_hyp);
dlml=-(y-mux)'*(-inv_M*dK_dlog_hyp*inv_M)*(y-mux)-trace(inv_M*dK_dlog_hyp);
dnlml(i)=-dlml;
end
dnlml=dnlml';

%dlog_sigma_n=-y'*(-inv_M*2*sigma_n^2*inv_M)*y-trace(inv_M*2*sigma_n^2);
dlog_sigma_n=-(y-mux)'*(-inv_M*2*sigma_n^2*inv_M)*(y-mux)-trace(inv_M*2*sigma_n^2);
dlog_sigma_n=-dlog_sigma_n;
dnlml=[dnlml;dlog_sigma_n];

end