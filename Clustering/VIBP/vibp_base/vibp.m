function [model, lower_bounds_log, num_iters] = ...
    vibp(X, alpha, sigma_a, sigma_n, model, lower_bounds_log, params)
% function [model, lower_bounds_log] = ...
%     vibp(X, alpha, sigma_a, sigma_n, model, lower_bounds_log, params)
%
% Performs the variational updates for the finite model and keeps
% track of the lower bound.

% Get parameters and constants.
nu = model.nu;
tau = model.tau;
phi_mean = model.phi_mean;
phi_cov = model.phi_cov;
[N K] = size(nu);
D = size(phi_mean, 1);
lower_bound_count = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform the variational updates.
iter = 1;
while 1
    % Update all the parameters.

    % Note that the order of updates might affect speed of
    % convergence, so we can play with that here.
    if params.vary_update_order
        param_update_perm = randperm(4);
    else
        param_update_perm = 1:4;
    end

    for param_index = param_update_perm
        switch param_index
            case 1
                %%%%%%%%%%%%%%%%%%%
                % Update all tau
                [ tau lower_bounds_log lower_bound_count ] = ...
                    update_tau( params, X, alpha, sigma_a, sigma_n, model, ...
                    lower_bounds_log, lower_bound_count );
                model.tau = tau;

            case 2
                %%%%%%%%%%%%%%%%%%%
                % Update all phi
                for k = 1:K
                    non_k_indices = [1:(k-1) (k+1):K];
                    % First compute the B and C matrices as outlined in the notes.
                    switch params.model_type

                        case 'LG'
                            B = (1/sigma_a^2 + params.test_mask' * nu(:,k) /sigma_n^2);
                            C = (X .* params.test_mask)' * nu(:,k);
                            for n = 1:N
                                % multiply in mask with the phi_mean is sufficient to zero out the
                                % dot product with A in the expectation
                                C = C - nu(n,k) * ( sum( nu(n*ones(D,1), non_k_indices) ...
                                    .* phi_mean(:, non_k_indices),2) .* params.test_mask(n,:)' );
                            end
                            C = C/ sigma_n^2;

                    end

                    % Then based on B and C, update phi.
                    phi_cov(:,k) = 1./B;
                    phi_mean(:,k) = C ./ B;
                    model.phi_mean = phi_mean;
                    model.phi_cov = phi_cov;

                    % Update our lower bound
                    if params.compute_intermediate_lb
                        lower_bounds_log(1,end+1) = ...
                            compute_variational_lower_bound(params, X, alpha, params.sigma_a, params.sigma_n, model);
                        lower_bounds_log(2,end) = 2;
                        lower_bound_count = lower_bound_count + 1;
                    end
                end

            case 3
                %%%%%%%%%%%%%%%%%%%
                % Update all nu
                for k = 1:K
                    % First compute var_theta as outlined in the notes.  var_theta_k
                    % is the term shared across nu(:,k) so that we don't recompute it.
                    % Minimal value for LG models
                    var_theta_k = compute_var_theta_k( params , tau , k );
                    non_k_indices = [1:(k-1) (k+1):K];
                    for n = randperm(N)
                        switch params.model_type
                            case 'LG'
                                var_theta = var_theta_k ...
                                    - 1/(2*sigma_n^2) * ( params.test_mask(n,:) * phi_cov(:,k) ...
                                    + params.test_mask(n,:) * (phi_mean(:,k).^2))...
                                    + 1/(sigma_n^2) * ( params.test_mask(n,:) .* phi_mean(:,k)' ) * ...
                                    (X(n,:)' - sum(nu(n*ones(D,1), non_k_indices) .* phi_mean(:, non_k_indices),2));
                        end

                        % Once we have var_theta, go from the canonical parameter to the
                        % mean parameter.
                        nu(n,k) = 1 / (1 + exp(-var_theta));
                        model.nu = nu;

                        % Update our lower bound
                        if params.compute_intermediate_lb
                            lower_bounds_log(1,end+1) = ...
                                compute_variational_lower_bound( params, X, alpha, parmas.sigma_a, params.sigma_n, model);
                            lower_bounds_log(2,end) = 3;
                            lower_bound_count = lower_bound_count + 1;
                        end
                    end
                end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lower_bounds_log(1,end+1) = ...
        compute_variational_lower_bound( params, X, alpha, params.sigma_a, params.sigma_n, model);
    lower_bounds_log(2,end) = 5;
    lower_bound_count = lower_bound_count + 1;

    % Determine stopping criteria and stop if we have converged or
    % done enough iterations.
    if (lower_bounds_log(1,end) - lower_bounds_log(1,end-lower_bound_count) ...
            < params.stopping_thresh) ||  (iter >= params.vibp_iter_count)
        break;
    end

    iter = iter + 1;
end

num_iters = iter;

