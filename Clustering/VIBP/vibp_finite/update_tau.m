function [ tau lower_bounds_log lower_bound_count ] = ...
    update_tau( params, X, alpha, sigma_a, sigma_n, model, ...
                lower_bounds_log, lower_bound_count );
% function [ tau lower_bounds_log lower_bound_count ] = ...
%     update_tau( params, X, alpha, sigma_a, sigma_n, model , ...
%                 lower_bounds_log, lower_bound_count );
[ N K ]= size( model.nu );
tau = model.tau;

% loop through and update each tau
for k = 1:K
    sum_nu_k = sum( model.nu(:,k));
    tau(1,k) = alpha/K + sum_nu_k;
    tau(2,k) = 1 + N - sum_nu_k;
    model.tau = tau;

    % Update our lower bound
    if params.compute_intermediate_lb
        lower_bounds_log(1,end+1) = ...
            compute_variational_lower_bound( params, X, alpha, sigma_a, sigma_n, model );
        lower_bounds_log(2,end) = 1;
        lower_bound_count = lower_bound_count + 1;
    end
end
