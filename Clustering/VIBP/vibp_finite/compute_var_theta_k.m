function var_theta_k = compute_var_theta_k( params , tau , k )
% function var_theta_k = compute_var_theta_k( params , tau , k  )
% an intermediate function in computing the nu update	

var_theta_k = psi(tau(1,k)) - psi(tau(2,k));
