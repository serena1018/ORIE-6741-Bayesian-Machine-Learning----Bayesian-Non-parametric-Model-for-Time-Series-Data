function [result K_plus] = sampleIBP(alpha, num_objects)
% the parameter ? influences how likely it is that multiple observations will share the same features
result = zeros(num_objects, 10);
%Generate Poisson – alpha is a concentration parameter
t = poissrnd(alpha);

result(1,1:t) = ones(1,t);
K_plus = t;
%Loop through each object (row)
for i=2:num_objects
%Loop through each column (latent feature)
    for j=1:K_plus
% Probability of getting the current dishes
%(Culinary Metaphor)
 
        p(1) = log(sum(result(1:i,j)))-log(i);
        p(2) = log(i - sum(result(1:i,j))) - log(i);
        p = exp(p-max(p));
        if rand < p(1)/sum(p)
            result(i,j) = 1;
        else
            result(i,j) = 0;
        end;
    end;
%New dishes
    t = poissrnd(alpha/i);
    result(i,K_plus+1:K_plus+t) = ones(1,t);
    K_plus = K_plus+t;
end;

result = result(:,1:K_plus);
