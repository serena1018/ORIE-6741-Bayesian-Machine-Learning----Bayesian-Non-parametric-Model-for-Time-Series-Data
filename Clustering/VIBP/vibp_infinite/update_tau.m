function [ tau lower_bounds_log lower_bound_count ] = ...
    update_tau( params, X, alpha, sigma_a, sigma_n, model , ...
                lower_bounds_log, lower_bound_count )
% function [ tau lower_bounds_log lower_bound_count ] = ...
%      update_tau( params, X, alpha, sigma_a, sigma_n, model , ...
%        lower_bounds_log, lower_bound_count )

nu = model.nu;
[ N K ] = size( nu );
sum_n_nu = sum(nu,1);
N_minus_sum_n_nu = N - sum_n_nu;

% Iterate through and update each tau(:,k)
for k = 1:K

  % First we compute q_k for k:K
  tau = model.tau;
  digamma_tau = psi( tau );
  digamma_sum = psi( sum( tau ) );
  digamma_tau1_cumsum = [ 0 cumsum( digamma_tau( 1 , 1:(K-1) ) ) ] ;
  digamma_sum_cumsum = cumsum( digamma_sum );
  exponent = digamma_tau( 2 , : ) + digamma_tau1_cumsum - digamma_sum_cumsum;
  unnormalized = exp(exponent - max(exponent));
  qs = zeros(K,K);
  for m = k:K
    qs(m, 1:m) = unnormalized(1:m) / sum(unnormalized(1:m));
  end

  % Now that we have the q_k, update the tau(:,k)
  tau(1,k) = sum(sum_n_nu(k:K)) + N_minus_sum_n_nu(k+1:K) * ...
      sum(qs(k+1:end, k+1:end),2) + alpha;
  tau(2,k) = N_minus_sum_n_nu(k:K) * qs(k:K, k) + 1;

  % Finally commit the udpated tau and compute the lower bound if desired.
  model.tau(:,k) = tau(:,k);
  if params.compute_intermediate_lb
    lower_bounds_log(1,end+1) = ...
        compute_variational_lower_bound( params, X, alpha, sigma_a, sigma_n, model);
    lower_bounds_log(2,end) = 1;
    lower_bound_count = lower_bound_count + 1;
  end
end

return
