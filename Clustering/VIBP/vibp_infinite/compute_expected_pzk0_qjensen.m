function [lb q] = compute_expected_pzk0_qjensen( tau , k )
% function [lb q] = compute_expected_pzk0_qjensen( tau , k );
%   computes E[ ln( 1 - prod( 1 - v ) ) ] where the prod is from 1..k
%   using q-distribution described by YWT

% select our relevant set of tau's (up to k)
tau = tau( : , 1:k );
digamma_tau = psi( tau );
digamma_sum = psi( sum( tau ) );

% compute the optimal q distribution
digamma_tau1_cumsum = [ 0 cumsum( digamma_tau( 1 , : ) ) ] ;
digamma_sum_cumsum = cumsum( digamma_sum );
tmp = digamma_tau( 2 , : ) + digamma_tau1_cumsum(1:k) - digamma_sum_cumsum;
q = exp( tmp - max(tmp) );
q = q / sum(q);

% compute the lb
lb = sum( q .* ( tmp - log( q ) ) );

return
