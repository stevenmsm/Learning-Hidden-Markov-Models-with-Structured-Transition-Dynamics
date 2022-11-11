clear all; close all; clc;
%
%% Read in CSV File
input = csvread("",1,1); %offset row 1 col 1
input(:,2) = input(:,2) + 1; %observation discrete states start with 1 to 6 here.
dict_obs = containers.Map; %create dictionary, each key is 1 observation sequence
unique_ID = unique(input(:,1));
for idx = 1:length(unique_ID) %loop through unique ID, store each obs to dict
    current_portion = input(input(:,1)==unique_ID(idx),2);
    dict_obs(string(unique_ID(idx))) = current_portion;
end

%
%% HMM Initialization
N = 3; %number of hidden states

initial = zeros(1, N);
initial(1,N) = 1; %Initialize all people start at the worst hidden state (concussion)
%initial(1,1) = 1;

O = max(unique(input(:,2))); %number of observation states
transition_final_our = zeros(N);
emission_final_our = zeros(N, O);
Total_MCMCIter = 10;
Total_EM_iters = 15;


%% MCMC Iterations
for Total_iter = 1:Total_MCMCIter
% Since observation state O (last state) is most severe and state 1 is
% least, we assume that hidden state N is most severe and state 1 is least

% Transition Probability Matrix Random Init
transition_our = rand(N);
transition_our = transition_our./repmat(sum(transition_our,2),[1,N]);

% Emission Probability Matrix Random Init
emission_our = rand(N,O);
emission_our = emission_our./repmat(sum(emission_our,2),[1,O]);

%
%% E-M Algorithm, Multiple Observation Sequences
for EM_iter = 1:Total_EM_iters
    %% E-Step

    % Initialize for E-step quantities
    L_sumALL = zeros(N,1); %Store L vector (L matrix summed over time and obs seq)
    L_0 = zeros(N,1); % Store P(S_0=i|O_1,...) sum over all obs seq
    L_sumIfObs = zeros(N,O); 
    H_sumALL = zeros(N); %Store H matrix sum over time and observation seq
    
    % Loop over all observation sequence
    for i_ObsSeq = 1:length(unique_ID)

        observed = dict_obs(string(unique_ID(i_ObsSeq)));
        T = length(observed);

        % Get Alpha and Beta from Forward-Backward Algorithm, which can be used to compute L and H (defined in Notes)
        Alpha_our = Forward_Alg(observed,transition_our,emission_our,initial);
        Beta_our = Backward_Alg(observed,transition_our,emission_our);
        
        % construct L; L(i,t)=P(S_t=i|O_1,...,O_T); rows are hidden states, columns are hidden state time steps
        L_our=Alpha_our.*Beta_our;
        L_our=L_our./sum(L_our);
        
        % construct H matrix; the most likely transition from t to t+1 given all observations
        % H is a 3-dimensional matrix, where H(i,j,t)=P(S_(t-1)=i,S_t=j|O_1,...,O_T)
        % i,j range from 1 to n, t range from 1 to T
        for t=1:T
            for i=1:N
                for j=1:N
                    H_mat_our(i,j,t)=(Alpha_our(i,t)*Beta_our(j,t+1)*emission_our(j,observed(t))*transition_our(i,j));
                end
            end
        end
        for t=1:T
            H_mat_our(:,:,t)=H_mat_our(:,:,t)./sum(H_mat_our(:,:,t),'all'); %normalize, scale each slice t to sum up to 1
        end

        % Calculate summed over E-steps for M-step (save memory)
        L_sumALL = L_sumALL + sum(L_our,2); %Sum L over time stamps and obs seq
        L_0 = L_0 + L_our(:,1); %Sum L_0 over obs seq

        obs_indicator = zeros(O,length(observed)); %Row i stores indicators where observation state i is in all time stamps.
        for ObsState_i = 1:O
            obs_indicator(ObsState_i,:) = (observed == ObsState_i);
        end
        %L_sumIfObs will be used in emission probability matrix update
        %entry(j,y)=sum_r sum_t L_j(t)*I{Y^r_t==y} (sum obs seq and sum over time for those obs is y)
        for ObsState_i = 1:O
            for HidState_j = 1:N
                tmp_L_sumIfObs = L_our(HidState_j,2:(length(observed)+1)).*obs_indicator(ObsState_i,:);
                L_sumIfObs(HidState_j,ObsState_i) = L_sumIfObs(HidState_j,ObsState_i) + sum(tmp_L_sumIfObs,2);
            end
        end

        H_sumALL = H_sumALL + sum(H_mat_our, 3); %Sum H matrix over time and obs seq
    end

    %
    %% M-Step
    for row = 1:N
        transition_our(row,:) = H_sumALL(row,:)./L_sumALL(row,:);
        emission_our(row,:) = L_sumIfObs(row,:)./L_sumALL(row,:);
    end
    
    % Update Initial Probabilities
    initial = L_0./length(unique_ID);
    %
end
transition_final_our = transition_final_our + transition_our;
emission_final_our = emission_final_our + emission_our(:, 1:O);

end

transition_final_our = transition_final_our./Total_MCMCIter;
transition_final_our = transition_final_our./repmat(sum(transition_final_our,2),[1,N]);
emission_final_our = emission_final_our./Total_MCMCIter;
emission_final_our = emission_final_our./repmat(sum(emission_final_our,2),[1,O]);


%% Prediction on each testing patient: 
% 1. find L=P(S_t=s|O_1,...,O_T) for each test obs sequence (state belief vector)
% 2. compute predictive distribution: d_t = L*emission = P(O_t=o) (integrated/summed out hidden states)
kl_div_our = 0;
HMM_pred_our = zeros(1,O);
true_obs = zeros(1,O);

for i_ObsSeq = 1:length(unique_ID)
    patient_n_kl = 0;
    observed = dict_obs(string(unique_ID(i_ObsSeq)));
    
    % Get Alpha and Beta from Forward-Backward Algorithm, which can be used to compute L and H (defined in Notes)
    Alpha_our = Forward_Alg(observed,transition_final_our,emission_final_our,initial);
    Beta_our = Backward_Alg(observed,transition_final_our,emission_final_our);
  
    % construct L; L(i,t)=P(S_t=i|O_1,...,O_T); rows are hidden states, columns are hidden state time steps
    L_our=Alpha_our.*Beta_our;
    L_our=L_our./sum(L_our);
    pred_our = L_our'*emission_final_our;
    pred_our = pred_our(1:(length(observed)),:);
    pred_our = pred_our./repmat(sum(pred_our')',[1, O]);
    HMM_pred_our = HMM_pred_our + sum(pred_our,1)./length(observed);
    
    for idx = 1:length(observed)
        true_obs(1, observed(idx)) = true_obs(1, observed(idx)) + 1;
    end
end
HMM_pred_our = HMM_pred_our./length(unique_ID);

true_obs = true_obs./sum(true_obs);

HMM_pred_our = HMM_pred_our./sum(HMM_pred_our);

kl_div_our = sum(true_obs.*log(true_obs./HMM_pred_our));
kl_div_our = round(kl_div_our,3)




%
%% Function Definitions
%
function [Alpha] = Forward_Alg(O, A, B, pi)
    [n,~]=size(A);
    T=length(O);
    
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
            Alpha(j,t+1)=Alpha(j,t+1)*B(j,O(t)); 
        end
    end
end


function [Beta] = Backward_Alg(O, A, B)
    [n,~]=size(A);
    T=length(O);
    
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
                Beta(j,T+1-t)=Beta(j,T+1-t)+B(i,O(T+1-t))*A(j,i)*Beta(i,T+2-t);
            end
        end
    end
end


