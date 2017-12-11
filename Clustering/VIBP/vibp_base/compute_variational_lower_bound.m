function [lower_bound] = compute_variational_lower_bound(...
    params, X, alpha, sigma_a, sigma_n, model)
% function [lower_bound] = compute_variational_lower_bound(...
%    params, X, alpha, sigma_a, sigma_n, model )
%
% Computes the variational lower bound on the log marginal likelihood
% of the data based on the variational parameters.

% Get parameters and constants.
nu = model.nu;
tau = model.tau;
phi_mean = model.phi_mean;
phi_cov = model.phi_cov;
[N K] = size(nu);
D = size(phi_mean, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The feature probabilities
digamma_sum = sum(psi(tau(1,:)) - psi(sum(tau,1)));
if params.use_finite_model
    lower_bound = K*log(alpha/K) + (alpha/K-1)*digamma_sum;
else
    lower_bound = K* log(alpha) + (alpha-1)*digamma_sum;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The feature stats distribution
feature_stat_bound = 0;
if params.use_finite_model
    feature_stat_bound = sum( nu * psi( tau( 1 , : ) )' ) ...
        + sum( ( 1 - nu )*psi(tau(2,:) )' )...
        - sum( psi( tau( 1 , : ) + tau( 2 , : ) )' )*N;
else
    for k = 1:K
        % This computes the expecation of log(1-prod v_i) by
        % introducing the variational multinomial distribution.
        % See the paper for details.
        f = compute_expected_pzk0_qjensen( tau , k );
        feature_stat_bound = feature_stat_bound ...
            + sum(nu(:,k)) * sum( psi( tau(1,1:k) ) - psi( sum(tau(:,1:k)) ) ) ...
            + (N - sum(nu(:,k))) * f;
    end
end
lower_bound = lower_bound + feature_stat_bound;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The feature distribution
feature_dist_bound =  sum(phi_cov(:)) + sum(dot(phi_mean, phi_mean));
lower_bound = lower_bound - D*K/2*log(2*pi*sigma_a^2) - 1/(2*sigma_a^2)* feature_dist_bound;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The likelihood
X = X .* params.test_mask; % mask out x's that are test data
likelihood_bound = sum( sum( X.^2 ) );
switch params.model_type

  % linear gaussian case
    case 'LG'
        likelihood_bound = likelihood_bound - 2 * sum( sum( nu .* ( X * phi_mean ) ));
        % first term sums up the number of dimensions present in each
        % data point for the phi_cov, since this is different for
        % each; second term computes dot(phi_mean,phi_mean) but again,
        % for each n, sums only the terms in the dot product with
        % dimensions present for that n.
        tmp = params.test_mask * ( phi_cov + phi_mean.^2 );
        likelihood_bound = likelihood_bound + sum(sum( nu .* tmp ));

        % we do sum_n sum_d sum_k sum_k' over all variables and then
        % subtract out the k=k' part that we do not want; first line
        % does sum_k nu_nk phi_kd; next we do mask_nd nuphi_nd
        % nuphi_nd
        tmp = nu * phi_mean';
        tmp2 = nu.^2 * (phi_mean').^2;
        likelihood_bound = likelihood_bound ...
            + sum(sum( params.test_mask .* tmp .* tmp )) ...
            - sum(sum( params.test_mask .* tmp2 ));

end

lower_bound = lower_bound - sum( params.test_mask(:) )/2*log(2*pi*sigma_n^2) - 1/(2*sigma_n^2)* likelihood_bound;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The entropy

% Add in a factor for pi/v as appropriate.
entropy_bound = sum( gammaln(tau(1,:)) + gammaln(tau(2,:)) ...
    - gammaln(sum(tau,1)) - (tau(1,:)-1) .* psi(tau(1,:)) - ...
    (tau(2,:)-1) .* psi(tau(2,:)) + (tau(1,:) + tau(2,:) - 2) .* psi(sum(tau,1)) );

% Add in a factor for the nonzero nu(n,k)
tmpnu = nu + .5 * ( nu == 0 ) - .5 * ( nu == 1 );
tmp = -1 * tmpnu .* log( tmpnu ) - ( 1 - tmpnu ).* log( 1 - tmpnu );
tmp = tmp .* ( nu > 0 ) .* ( nu < 1 );
entropy_bound = entropy_bound + sum( tmp(:) );
entropy_bound = entropy_bound + K*D/2*log(2*pi*exp(1));
entropy_bound = entropy_bound + 1/2*sum(log( phi_cov(:) ));
lower_bound = lower_bound + entropy_bound;

return

