clear all; close all; clc;
%
%% Read in CSV File
obs_input_file1 = ""
obs_input_file2 = ""
input_1 = csvread(obs_input_file1,1,1); %offset row 1 col 1
input_2 = csvread(obs_input_file2,1,1); %offset row 1 col 1
unique_ID = intersect(unique(input_2(:,1)),unique(input_1(:,1)));

% Obs 1
input_1(:,2) = input_1(:,2) + 1; 
dict_obs_1 = containers.Map; %create dictionary, each key is 1 observation sequence
for idx = 1:length(unique_ID) %loop through unique ID, store each obs to dict
    current_portion = input_1(input_1(:,1)==unique_ID(idx),2);
    dict_obs_1(string(unique_ID(idx))) = current_portion;
end

% Obs 2
input_2(:,2) = input_2(:,2) + 1; %observation discrete states start with 1 to 6 here.
dict_obs_2 = containers.Map; %create dictionary, each key is 1 observation sequence
for idx = 1:length(unique_ID) %loop through unique ID, store each obs to dict
    current_portion = input_2(input_2(:,1)==unique_ID(idx),2);
    dict_obs_2(string(unique_ID(idx))) = current_portion;
end

%
%% HMM Initialization
N = 3; %number of hidden states
%there are 3 sets of observation states here, each dim is num of obs states
O = [max(unique(input_1(:,2))), max(unique(input_2(:,2)))]; 

% Since observation state O (last state) is most severe and state 1 is
% least, we assume that hidden state N is most severe and state 1 is least
initial = zeros(1, N);
initial(1,N) = 1; %Initialize all people start at the worst hidden state (concussion)

%transition = ones(N,N)/N; % Uniform Initialize transition probability matrix
transition = rand(N);
transition = transition-(triu(transition,1)+tril(transition,-2)); % Enforce Upper-trian=0, lower 2 train =0
transition = transition./repmat(sum(transition,2),[1,N]);

%emission = ones(N,O)/N; % Uniform Initialize emission probability matrix
emission_1 = rand(N,O(1));
emission_1 = emission_1./repmat(sum(emission_1,2),[1,O(1)]);
emission_2 = rand(N,O(2));
emission_2 = emission_2./repmat(sum(emission_2,2),[1,O(2)]);

%% IFR Constraint Set-up
[b_inequality_Trans, d_inequality_Trans] = IFR_mat_setup(N,N,(N-1)*(N-1));
[b_inequality_Emiss_1,d_inequality_Emiss_1] = IFR_mat_setup(N,O(1),(N-1)*(O(1)-1));
[b_inequality_Emiss_2,d_inequality_Emiss_2] = IFR_mat_setup(N,O(2),(N-1)*(O(2)-1));

%
%% E-M Algorithm, Multiple Observation Sequences
Total_EM_iters = 2;
total_time_our = zeros(1,Total_EM_iters);

for EM_iter = 1:Total_EM_iters
%% E-Step

% Initialize for E-step quantities
L_sumALL = zeros(N,1); %Store L vector (L matrix summed over time and obs seq)
L_0 = zeros(N,1); % Store P(S_0=i|O_1,...) sum over all obs seq
L_sumIfObs_1 = zeros(N,O(1)); 
L_sumIfObs_2 = zeros(N,O(2)); 
H_sumALL = zeros(N); %Store H matrix sum over time and observation seq

% Loop over all observation sequence
for i_ObsSeq = 1:length(unique_ID)
    
    observed_1 = dict_obs_1(string(unique_ID(i_ObsSeq)));
    observed_2 = dict_obs_2(string(unique_ID(i_ObsSeq)));
    
    min_T = min([length(observed_1),length(observed_2)]);
    observed_1 = observed_1(1:min_T);
    observed_2 = observed_2(1:min_T);

    %assert(length(observed_BESS)==length(observed_SAC));
    %assert(length(observed_SCAT)==length(observed_SAC));

    T = min_T;
    
    % Get Alpha and Beta from Forward-Backward Algorithm, which can be used to compute L and H (defined in Notes)
    Alpha = Forward_Alg(observed_1,observed_2,transition,emission_1,emission_2,initial);
    Beta = Backward_Alg(observed_1,observed_2,transition,emission_1,emission_2);
    
    % construct L; L(i,t)=P(S_t=i|O_1,...,O_T); rows are hidden states, columns are hidden state time steps
    L=Alpha.*Beta;
    if (sum(L,'All')<1e-40)
        L = L + 1e-40;
    end
    L=L./sum(L);
    
    % construct H matrix; the most likely transition from t to t+1 given all observations
    % H is a 3-dimensional matrix, where H(i,j,t)=P(S_(t-1)=i,S_t=j|O_1,...,O_T)
    % i,j range from 1 to n, t range from 1 to T
    for t=1:T
        for i=1:N
            for j=1:N
                H_mat(i,j,t)=(Alpha(i,t)*Beta(j,t+1)*emission_1(j,observed_1(t))*emission_2(j,observed_2(t))*transition(i,j));
            end
        end
    end
    for t=1:T
        H_mat(:,:,t)=H_mat(:,:,t)./sum(H_mat(:,:,t),'all'); %normalize, scale each slice t to sum up to 1
    end
    
    % Calculate summed over E-steps for M-step (save memory)
    L_0 = L_0 + L(:,1); %Sum L_0 over obs seq
    
    obs_indicator_1 = zeros(O(1),length(observed_1)); %Row i stores indicators where observation state i is in all time stamps.
    for ObsState_i = 1:O(1)
        obs_indicator_1(ObsState_i,:) = (observed_1 == ObsState_i);
    end
    for ObsState_i = 1:O
        for HidState_j = 1:N
            tmp_L_sumIfObs = L(HidState_j,2:(length(observed_1)+1)).*obs_indicator_1(ObsState_i,:);
            L_sumIfObs_1(HidState_j,ObsState_i) = L_sumIfObs_1(HidState_j,ObsState_i) + sum(tmp_L_sumIfObs,2);
        end
    end
    
    obs_indicator_2 = zeros(O(2),length(observed_2)); %Row i stores indicators where observation state i is in all time stamps.
    for ObsState_i = 1:O(2)
        obs_indicator_2(ObsState_i,:) = (observed_2 == ObsState_i);
    end
    for ObsState_i = 1:O
        for HidState_j = 1:N
            tmp_L_sumIfObs = L(HidState_j,2:(length(observed_2)+1)).*obs_indicator_2(ObsState_i,:);
            L_sumIfObs_2(HidState_j,ObsState_i) = L_sumIfObs_2(HidState_j,ObsState_i) + sum(tmp_L_sumIfObs,2);
        end
    end
    
    H_sumALL = H_sumALL + sum(H_mat, 3); %Sum H matrix over time and obs seq
end

%
%% M-Step
% Optimize Transition Probability Matrix (IFR)
tic
[P_our,u_matrix,w_matrix,v_matrix,iter_our] = GeneralCase_Solver(H_sumALL,b_inequality_Trans,d_inequality_Trans);
transition = P_our;

% Optimize Emission Probability Matrix (IFR)
[B_our,u_matrix,w_matrix,v_matrix,iter_our] = GeneralCase_Solver(L_sumIfObs_1,b_inequality_Emiss_1,d_inequality_Emiss_1);
emission_1 = B_our;

[B_our,u_matrix,w_matrix,v_matrix,iter_our] = GeneralCase_Solver(L_sumIfObs_2,b_inequality_Emiss_2,d_inequality_Emiss_2);
emission_2 = B_our;

% Update Initial Probabilities
initial = L_0./length(unique_ID);

total_time_our(EM_iter) = toc;

%
end


%
%% Function Definitions
%

function [ineq_mat, ineq_vec] = IFR_mat_setup(dim1,dim2,dim3)
% Each subrow is less or equal to the next row's subrow, here since row
% stochastic, the whole row already satisfies the whole row constraint.
% i.e. if 3x3 matrix, first row last two elements sum is less or equal to
% second row's last two elements' sum, then a_equality=[0,1,1,0,-1,-1,0,0,0]
% Thus, each set of constraint is represented in a NxN matrix, and there are m=(N-1)x(N-1) sets of ineq-constraints 
    ineq_mat=zeros(dim1,dim2,dim3); %sum_isum_j b_{ij}^mp_{ij}>=d^m
    thirdDim_index = 0;
    for j=2:dim2
        for i=1:(dim1-1)
            thirdDim_index=thirdDim_index+1;
            ineq_mat(i,j:dim2,thirdDim_index)=-1;
            ineq_mat(i+1,j:dim2,thirdDim_index)=1;
        end
    end
    ineq_vec=zeros(1,thirdDim_index); 
end


function [Alpha] = Forward_Alg(O_SAC, O_SCATSEV, A, B_SAC, B_SCATSEV, pi)
    [n,~]=size(A);
    T=length(O_SAC);
    
    %Alpha is going to be a n by (T+1) matrix s.t. 
    %Alpha(i,j)=P(O_1,...,O_(j-1),S_(j-1)=i), 
    %i.e. columns are time steps, rows are hidden states
    Alpha = zeros(n,T+1);
    for i=1:n        %initialize Alpha(1,i)=P(S_0=i)=pi_i
        Alpha(i,1)=pi(i);
    end
    for t=1:T      %loop over all observations
        for j=1:n      %loop over all hidden state space
            for i=1:n  %sum over (t)'s all state space
                Alpha(j,t+1)=Alpha(j,t+1)+A(i,j)*Alpha(i,t);
            end
            %alpha_j(t+1)=P(O_1,...,O_t, S_t=j)=P(O_t|S_t=j)*sum_i(A(i,j)*alpha_i(t))
            Alpha(j,t+1)=Alpha(j,t+1)*B_SAC(j,O_SAC(t))*B_SCATSEV(j,O_SCATSEV(t)); 
        end
    end
end


function [Beta] = Backward_Alg(O_SAC, O_SCATSEV, A, B_SAC, B_SCATSEV)
    [n,~]=size(A);
    T=length(O_SAC);
    
    %Beta is going to be a n by (T+1) matrix s.t. 
    %Beta_i(t)=P(O_(t+1),...,O_T|S_t=i)=sum_j B(j,O_(t+1))*A(i,j)*beta_j(t+1)
    %i.e. columns are time steps, rows are hidden states
    Beta = zeros(n,T+1);
    for i=1:n        %initialize Alpha(1,i)=P(S_0=i)=pi_i
        Beta(i,T+1)=1;
    end
    for t=1:T      %loop over all observations
        for j=1:n      %loop over all hidden state space
            for i=1:n  %sum over (t)'s all state space
                Beta(j,T+1-t)=Beta(j,T+1-t)+B_SAC(i,O_SAC(T+1-t))*B_SCATSEV(i,O_SCATSEV(T+1-t))*A(j,i)*Beta(i,T+2-t);
            end
        end
    end
end


% Our Approach
% Inputs:
% H_sum: 2D matrix store the H matrix summed over time and obs seq
% b_inequality: 3D matrix stores inequality constraints, last dim is # constraints
% d_inequality: 1D matrix stores inequality constraints' subject value
function [P,u_matrix,w_matrix,v_matrix,iter] = GeneralCase_Solver(H_sum,b_inequality,d_inequality)
    % initialization
    lambda=0;
    alpha=[];
    beta=0.05;
    
    [X, Y] = size(H_sum);
    
    % The rows of u_matrix, w_matrix is the u, w vector at current iteration; append using [u_matrix;new_u]
    % u_matrix is Lagrangian Multipliers for equality constraints, w_matrix is for inequality constraints
    %[~,~,L] = size(a_equality); %get total number of equality constraints
    [~,~,M] = size(b_inequality); %get total number of inequality constraints
    u_matrix = [];
    %u_matrix = zeros(1,L); %number of constraints equal to number of lagrange multipliers
    %u_tilde_matrix = u_matrix;
    w_matrix = zeros(1,M);
    w_tilde_matrix = w_matrix;
    
    Total_Iter = [30,10]; % define # of iterations for the whole algorithm and bisection method
    
    
    % repeat (computing v_i, p*_ij, u_j) until convergence 
    for iter=1:Total_Iter(1)
    %iter = 1; %initiate iterations for our method to converge (based on u_j, descent)
    %iter_Bisection = 0; %initiate iterations for bisection to converge
    %while 1
        % Bisection Method to find v_i in (\max_j{sum_lu^la_{ij}^l+sum_mw^mb_{ij}^m, infinity)
        % Define u_w_temp_matrix(i,j)=sum_lu^la_{ij}^l+sum_mw^mb_{ij}^m, easier for computing P later
        %u_w_temp_matrix = zeros(X,X); 
        %u_w_temp_matrix = sum(bsxfun(@times,a_equality,reshape(u_matrix(iter,:),1,1,[])),3) + sum(bsxfun(@times,b_inequality,reshape(w_matrix(iter,:),1,1,[])),3);
        u_w_temp_matrix = sum(bsxfun(@times,b_inequality,reshape(w_matrix(iter,:),1,1,[])),3);
        for i=1:X % loop over all i=1,...,X to find root v_i
            %for j=1:X
                %u_w_temp_matrix(i,j) = sum(squeeze(a_equality(i,j,:))'.*u_matrix(iter,:))+sum(squeeze(b_inequality(i,j,:))'.*w_matrix(iter,:));
                %u_w_temp_matrix(i,j) = sum(squeeze(b_inequality(i,j,:))'.*w_matrix(iter,:));
                %u_w_temp_matrix(i,j) = u_matrix(iter,j);
            %end
            %u_w_temp_matrix = permute(sum(permute(b_inequality,[3,2,1]).*repmat(w_matrix(iter,:)',[1,X])),[3,2,1]);
            max_u_w = max(u_w_temp_matrix(i,:)); %find max_j{sum_l u^la_{ij}^l+sum_m u^ma_{ij}^m}, the lower bound of v_i
            
            % pick the function value negative starting point
            vi_neg = 100; % pick 100 as the "big" number
            vi_neg_vec = repelem(vi_neg,Y); % vectorize for easier computation
            func_val_neg = sum(H_sum(i,:)./(vi_neg_vec - u_w_temp_matrix(i,:)))-1; %function value f_i(v_i)
            while func_val_neg > 0 %if function value not negative, change until it is negative
                vi_neg = vi_neg*10;
                vi_neg_vec = repelem(vi_neg,Y); % vectorize for easier computation
                func_val_neg = sum(H_sum(i,:)./(vi_neg_vec - u_w_temp_matrix(i,:)))-1;
            end
            % pick the function value positive starting point
            vi_pos = max_u_w+abs(normrnd(0,0.1));
            vi_pos_vec = repelem(vi_pos,Y);
            func_val_pos = sum(H_sum(i,:)./(vi_pos_vec - u_w_temp_matrix(i,:)))-1;
            iter_temp=1;
            while func_val_pos < 0 %check if function value is positive
                %fprintf("Error, function doesn't have postive value!\n");
                vi_pos = max_u_w+abs(normrnd(0,0.1^(iter_temp+1)));
                vi_pos_vec = repelem(vi_pos,Y);
                func_val_pos = sum(H_sum(i,:)./(vi_pos_vec - u_w_temp_matrix(i,:)))-1;
                iter_temp=iter_temp+1;
                if iter_temp>20
                    fprintf("Error, function doesn't have postive value!\n");
                    return
                end
            end
            % Start Bisection Method to find the root of vi
            for ind_Bisec = 1:Total_Iter(2)
            %while 1
                vi_new = (vi_neg+vi_pos)./2; %pick the middle point as the new point
                vi_new_vec = repelem(vi_new , Y); %vectorize vi_new for easier computation
                func_val_new = sum(H_sum(i,:)./(vi_new_vec - u_w_temp_matrix(i,:)))-1; % only need ith row in H_sum for computing v_i
                if func_val_new >0
                    vi_pos = vi_new;
                else
                    vi_neg = vi_new;
                end
                
                %if abs(vi_pos-vi_neg)./abs(vi_new) <=1e-7 %when to terminate bisection
                %if abs(vi_pos-vi_neg) <=1e-3 %when to terminate bisection
                %    iter_Bisection = iter_Bisection+1;
                    %v(i) = vi_neg;
                %    break
                %else
                %    iter_Bisection=iter_Bisection+1;
                %end   
            end
            v_matrix(iter,i) = vi_neg;
            %abs(vi_pos-vi_neg)./abs(vi_new)
        end
        
        
        % Derive p*_ij from v_i and u^l,w^m via equation 9:
        %   P_ij=H_sum(i,j)/(v_i-(sum_lu^la_{ij}^l+sum_mw^mb_{ij}^m))) 
        v_temp_matrix = repmat(v_matrix(iter,:)',[1,Y]); % v are rows multipliers
        P = H_sum./(v_temp_matrix - u_w_temp_matrix); 
        
        
        % Accelerated Steepest Descent to derive u_j from p*_ij
        lambda_new = (1+sqrt(1+4*(lambda(iter)).^2))./2;
        lambda = [lambda,lambda_new];
        alpha_new = (1-lambda(iter))./lambda(iter+1);
        alpha=[alpha,alpha_new];
        %update u and u_tilde vectors
        %u_tilde_new = u_matrix(iter,:) - beta.*(squeeze(sum(sum(P.*a_equality(:,:,:),1)))'-c_equality);
        %u_new = (1-alpha(iter)).*u_tilde_new + alpha(iter).*u_tilde_matrix(iter,:);
        %u_matrix=[u_matrix;u_new]; 
        %u_tilde_matrix = [u_tilde_matrix;u_tilde_new];
        %update w and w_tilde vectors
        w_tilde_new = w_matrix(iter,:) - beta.*(squeeze(sum(sum(P.*b_inequality(:,:,:),1)))'-d_inequality);
        w_tilde_new(w_tilde_new<0)=0;
        w_new = (1-alpha(iter)).*w_tilde_new + alpha(iter).*w_tilde_matrix(iter,:);
        w_new(w_new<0) = 0; % enforce lagrangian multipliers w to be strickly nonnegative
        w_matrix=[w_matrix;w_new]; 
        w_tilde_matrix = [w_tilde_matrix;w_tilde_new];
        
        %u_return_temp = norm(u_matrix(iter,:)-u_matrix(iter+1,:));
        w_return_temp = norm(w_matrix(iter,:)-w_matrix(iter+1,:))./norm(w_matrix(iter,:));
        %if (u_return_temp<=1e-3 && w_return_temp<=1e-3)
        %    iter=iter+1;
        %    return
        %else
        %    iter=iter+1;
        %end
    end
    
end





