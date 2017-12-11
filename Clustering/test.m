% This is an implementation of the algorithm described in the Computational
% cognition cheat sheet titied "The Indian Buffet Process."
% Original Written by Ilker Yildirim, September 2012.

randn('seed', 1); 
% Test with motorcycle data

dat = csvread('temp_data.csv');
acc = dat(:,1)
num_objects = 94; 
object_dim = 1;

sigma_x_orig = .5;
X = acc
seq = 1:94

% Fit RBF Kernel to the whole data set
gprMdl1 = fitrgp(seq',X,'KernelFunction','squaredexponential');
%Plot result
ypred = resubPredict(gprMdl1);
[ypred,ysd,yinit] = resubPredict(gprMdl1)

%Prediction interval
hold on
plot(yinit)
plot(acc , ':b')
hold off
% See that the interval for the first chunk is too wide due to
% non-stationary

%plot(seq,X,'b.');
% hold on;
% plot(seq',ypred,'r','LineWidth',1.0);
% xlabel('x');
% ylabel('y');
% legend('Data','GPR predictions');
% hold off

% Compute Harmonic number for N.
%N = maxNumCompThreads
% For testing on the server
%LASTN = maxNumCompThreads(10)
%LASTN = maxNumCompThreads('automatic')

HN = 0;
for i=1:num_objects HN = HN + 1/i; end;
E = 1000;
BURN_IN = 200;
SAMPLE_SIZE = 94;

% Initialize the chain.
sigma_A = 1;
sigma_X = 1;
alpha = 0.5;
K_inf = 3;

%test = zeros(10:1)
%test culinary metaphor
%for i=1:10 test(i,1) = size(sampleIBP(alpha, num_objects),2); end;

    %Prior over matrix Z
[Z K_plus] = sampleIBP(alpha, num_objects);
chain.Z = zeros(SAMPLE_SIZE,num_objects,K_inf);
chain.K = zeros(SAMPLE_SIZE, 1);
chain.sigma_X = zeros(SAMPLE_SIZE, 1);
chain.sigma_A = zeros(SAMPLE_SIZE, 1);
chain.alpha = zeros(SAMPLE_SIZE,1);
s_counter = 0;

for e=1:E
    % Store samples after the BURN-IN period.
    if (e > BURN_IN)
        s_counter = s_counter+1;
        chain.Z(s_counter,:,1:K_plus) = Z(:,1:K_plus);
        chain.K(s_counter) = K_plus;
        chain.sigma_X(s_counter) = sigma_X;
        chain.sigma_A(s_counter) = sigma_A;
        chain.alpha(s_counter) = alpha;
    end;
    disp(['At iteration ', num2str(e), ': K_plus is ', num2str(K_plus), ', alpha is ', num2str(alpha)]);

    for i=1:num_objects
        % The matrix M will be handy for future likelihood and matrix
        % inverse computations.
        M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
        for k=1:K_plus
            % That can happen, since we may decrease K_plus inside.
            if (k>K_plus)
                break;
            end;
            if Z(i,k) > 0
                % Take care of singular features
                if sum(Z(:,k)) - Z(i,k) <= 0
                    Z(i,k) = 0;
                    Z(:,k:K_plus-1) = Z(:,k+1:K_plus);
                    K_plus = K_plus-1;
                    M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + ...
                        (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
                    continue;
                end;
            end;
            
            % This equations are for computing the inverse efficiently.
            % It is an implementation of the trick from Griffiths and
            % Ghahramani (2005; Equations 51 to 54). 
            M1 = calcInverse(Z(:,1:K_plus), M, i, k, 1);
            M2 = calcInverse(Z(:,1:K_plus), M, i, k, 0);

            % Compute conditional distributions for the current cell in Z.
            Z(i,k) = 1;
            P(1) = likelihood(X, Z(:,1:K_plus), M1, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(sum(Z(:,k))- Z(i,k)) -log(num_objects);

            Z(i,k) = 0;
            P(2) = likelihood(X, Z(:,1:K_plus), M2, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(num_objects - sum(Z(:,k))) - log(num_objects);
            P = exp(P - max(P));
            
            % Sample from the conditional.
            if rand < P(1)/(P(1)+P(2))
                Z(i,k) = 1;
                M = M1;
            else
                Z(i,k) = 0;
                M = M2;
            end;
        end;
        % Sample the number of new dishes for the current object.
        %if K_plus <= K_inf
        %disp('Samling New Dishes'); 
        trun = zeros(1,2);
        alpha_N = alpha / num_objects;
        %This is basically the liklihood of the new dish
        for k_i=0:1
            
            Z(i,K_plus+1:K_plus+k_i) = 1;
            
            M = (Z(:,1:K_plus+k_i)'*Z(:,1:K_plus+k_i) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus+k_i,1)))^-1;
            
            trun(k_i+1) = k_i*log(alpha_N) - alpha_N - log(factorial(k_i)) + ...
                likelihood(X, Z(:,1:K_plus+k_i), M, sigma_A, sigma_X, K_plus+k_i, num_objects, object_dim);
        end;
        
        % This is an odd part of the code -- can modify later 
        % Normally for object recognition , we will look at infinite K
        % In our case, we should have limit on K
        Z(i,K_plus+1:K_plus+2) = 0;
        trun = exp(trun - max(trun));
        trun = trun/sum(trun);
        p = rand;
        t = 0;
        for k_i=0:1
            t = t+trun(k_i+1);
            if p < t
                new_dishes = k_i;
                break;
            end;
        end;
        if K_plus <= K_inf
            Z(i,K_plus+1:K_plus+new_dishes) = 1;
            %disp([num2str(new_dishes)]);
            K_plus = K_plus + new_dishes;
        end;
    end;
     % End of odd part
     
    % Metropolis steps for sampling sigma_X and sigma_A
    %Here noise is Uniform (-5,5) --> Can also sample from noise
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    
    l_curr = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);

    if rand < .5
        pr_sigma_X = sigma_X - rand/20;
    else
        pr_sigma_X = sigma_X + rand/20;
    end;
    
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (pr_sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_X = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, pr_sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_X = exp(min(0, l_new_X - l_curr));

    if rand < .5
        pr_sigma_A = sigma_A - rand/20;
    else
        pr_sigma_A = sigma_A + rand/20;
    end;
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/pr_sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_A = likelihood(X, Z(:,1:K_plus+new_dishes), M, pr_sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_A = exp(min(0, l_new_A - l_curr));

    if rand < acc_X
        sigma_X = pr_sigma_X;
    end;
    if rand < acc_A
        sigma_A = pr_sigma_A;
    end;

    % Sample alpha from its conditional posterior.
    % Posterior according to Zoubin
    alpha = mygamrnd(1+K_plus, 1/(1+HN),1);
    
    % Save the chain at every 1000th iteration.
    if mod(e,1000) == 0
        s = strcat('chain_ibp_',num2str(e));
        save(s, 'chain');
    end;
end;

post_Z  = [acc , Z]
%Frequency table 
sum(Z)
sum(Z,2)

%gprMdl1 = fitrgp(seq(1:size(C1 , 1))',C1,'KernelFunction','squaredexponential');
%[ypred,ysd,yinit] = resubPredict(gprMdl1)
%plot(yinit)
%hold on
%plot(C1 , ':b')

C1 = post_Z(post_Z(:,2) == 1 , 1)
C2 = post_Z(post_Z(:,3) == 1 , 1)
C3 = post_Z(post_Z(:,4) == 1 , 1)
C4 = post_Z(post_Z(:,5) == 1 , 1)
%%%%%%%%%%%%% THIS IS NOT GOOD PROBABLY LATENT FEATURE IS NOT EXACTLY
%%%%%%%%%%%%% CLUSTER ??? %%%%%%%%%%%%%%%%%%
%This paper might worth reading 
%http://mlg.eng.cam.ac.uk/zoubin/papers/HelGha07over.pdf
figure
subplot(2,2,1)
plot(seq(1:size(C1,1)) , C1)
subplot(2,2,2)
plot(seq(1:size(C2,1)) , C2)
subplot(2,2,3)
plot(seq(1:size(C3,1)) , C3)
subplot(2,2,4)
plot(seq(1:size(C4,1)) , C4)

%%%%%%%%%%%%% THIS MIGHT BE A BETTER WAY TO DEFINE CLUSTER i.e
%%%%%%%%%%%%% overlapping cluster 
% 0 0 0 0
C1 = post_Z(post_Z(:,2) == 0 & post_Z(:,3) == 0 & post_Z(:,4) == 0 & post_Z(:,5) == 0  , 1)
% 1 0 0 0
C2 = post_Z(post_Z(:,2) == 1 & post_Z(:,3) == 0 & post_Z(:,4) == 0 & post_Z(:,5) == 0  , 1)
% 1 1 0 0
C3 = post_Z(post_Z(:,2) == 1 & post_Z(:,3) == 1 & post_Z(:,4) == 0 & post_Z(:,5) == 0  , 1)
% 1 1 1 0
C4 = post_Z(post_Z(:,2) == 1 & post_Z(:,3) == 1 & post_Z(:,4) == 1 & post_Z(:,5) == 0  , 1)

figure
subplot(2,2,1)
plot(seq(1:size(C1,1)) , C1)
subplot(2,2,2)
plot(seq(1:size(C2,1)) , C2)
subplot(2,2,3)
plot(seq(1:size(C3,1)) , C3)
subplot(2,2,4)
plot(seq(1:size(C4,1)) , C4)


