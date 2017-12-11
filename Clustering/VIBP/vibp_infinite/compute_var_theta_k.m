function var_theta_k = compute_var_theta_k( params , tau , k )
% function var_theta_k = compute_var_theta_k( params , tau , k )
% an intermediate function in computing the nu update

f = compute_expected_pzk0_qjensen( tau, k );
var_theta_k = sum( psi( tau(1,1:k) ) - psi( sum(tau(:,1:k)) ) ) -  f;

return
