%Yutao Han - Cornell University
%10.26.2017
%ORIE 6741 Project, Toy DPGP Clustering
%negLogProb

function [nlml, dnlml] = ToyGP_negLogProb(hyp,x,y,k)

%empirical mean
mux=mean(y);
ell=exp(hyp(1));
sf=exp(hyp(2));
sigma_n=exp(hyp(end));%extract noise sigma_n

%for function
z=x;

%function K = covSEiso(hyp, x, z, i)
hyp_input=[log(ell);log(sf)];

K = feval(k, hyp_input, x, z);  
i=1;
dK_dlog_ell = feval(k, hyp_input, x, z, i); 
i=2;
dK_dlog_sf = feval(k, hyp_input, x, z, i);  

M=K+sigma_n^2*eye(length(y));
M=M+1e-6*eye(length(y));
R_M=chol(M);
inv_M=solve_chol(R_M,eye(length(y)));
log_det_M=2*sum(log(diag(R_M)));%log determinant of M

dnlml_dlog_sigma_n=-(y-mux)'*(-inv_M*2*sigma_n^2*inv_M)*(y-mux)-trace(inv_M*2*sigma_n^2);
%dnlml_dlog_sigma_n=-(y)'*(-inv_M*2*sigma_n^2*inv_M)*(y)-trace(inv_M*2*sigma_n^2);


dnlml_dlog_ell=-(y-mux)'*(-inv_M*dK_dlog_ell*inv_M)*(y-mux)-trace(inv_M*dK_dlog_ell);
dnlml_dlog_sf=-(y-mux)'*(-inv_M*dK_dlog_sf*inv_M)*(y-mux)-trace(inv_M*dK_dlog_sf);
%dnlml_dlog_ell=-y'*(-inv_M*dK_dlog_ell*inv_M)*y-trace(inv_M*dK_dlog_ell);
%dnlml_dlog_sf=-y'*(-inv_M*dK_dlog_sf*inv_M)*y-trace(inv_M*dK_dlog_sf);


%nlml=-y'*inv_M*y-log_det_M;
nlml=-(y-mux)'*inv_M*(y-mux)-log_det_M;
nlml=-nlml;%negative log likelihood
dnlml=[dnlml_dlog_ell; dnlml_dlog_sf; dnlml_dlog_sigma_n];
dnlml=-dnlml;%negative derivatives
end