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

outfolder = "";

%
%% HMM Initialization
N = 3; %number of hidden states

initial = zeros(1, N);
%initial(1,N) = 1; %Initialize all people start at the worst hidden state (concussion)
initial(1,1) = 1;

O = max(unique(input(:,2))); %number of observation states
transition_fmincon_final = zeros(N);
emission_fmincon_final = zeros(N, O);
Total_MCMCIter = 2;
Total_EM_iters = 2;
total_time_fmincon = 0;

options = optimoptions(@fmincon,'Algorithm','interior-point','Display','off');
options.Algorithm = 'trust-region-reflective';


%% IFR Constraint Set-up for Fmincon
% Transition Probability Matrix
Fmincon_Aeq_Trans = zeros(N,N*N); %since Aeq*P=beq, Aeq rows are the equality constraints (row sum to 1)
for i = 1:N
    Fmincon_Aeq_Trans(i,(i:N:end))=1;    %put in rows sum to 1 constraint
end
Fmincon_beq_Trans = ones(N,1);
% put in A*P0<=b for IFR
Fmincon_Aineq_Trans = zeros((N)*(N-1),N*N); %since Aeq*P=beq, Aeq rows are the equality constraints (row sum to 1)
thirdDim_index=0;
for j = 1:N
    for i = 1:N-1
        thirdDim_index = thirdDim_index+1;
        Fmincon_Aineq_Trans(thirdDim_index,((j-1)*N+i:N:end))=1;    %put in rows sum to 1 constraint
        Fmincon_Aineq_Trans(thirdDim_index,((j-1)*N+1+i:N:end))=-1;
    end
end
Fmincon_bineq_Trans=zeros((N)*(N-1),1);
Fmincon_lb_Trans = zeros(N*N,1);
Fmincon_ub_Trans = ones(N*N,1);

% Emission Probability Matrix
Fmincon_Aeq_Emis = zeros(N,N*O); %since Aeq*P=beq, Aeq rows are the equality constraints (row sum to 1)
for i = 1:N
    Fmincon_Aeq_Emis(i,(i:N:end))=1;    %put in rows sum to 1 constraint
end
Fmincon_beq_Emis = ones(N,1);
% put in A*P0<=b for IFR
Fmincon_Aineq_Emis = zeros((N)*(N-1),N*O); %since Aeq*P=beq, Aeq rows are the equality constraints (row sum to 1)
thirdDim_index=0;
for j = 1:N
    for i = 1:N-1
        thirdDim_index = thirdDim_index+1;
        Fmincon_Aineq_Emis(thirdDim_index,((j-1)*N+i:N:end))=1;    %put in rows sum to 1 constraint
        Fmincon_Aineq_Emis(thirdDim_index,((j-1)*N+1+i:N:end))=-1;
    end
end
Fmincon_bineq_Emis=zeros((N)*(N-1),1);
Fmincon_lb_Emis = zeros(N*O,1);
Fmincon_ub_Emis = ones(N*O,1);



%% MCMC Iterations
for Total_iter = 1:Total_MCMCIter
% Since observation state O (last state) is most severe and state 1 is
% least, we assume that hidden state N is most severe and state 1 is least

% Transition Probability Matrix Random Init
transition_fmincon = rand(N);
transition_fmincon = transition_fmincon-(triu(transition_fmincon,1)+tril(transition_fmincon,-2)); % Enforce Upper-trian=0, lower 2 train =0
transition_fmincon = transition_fmincon./repmat(sum(transition_fmincon,2),[1,N]);

% Emission Probability Matrix Random Init
emission_fmincon = rand(N,O);
emission_fmincon = emission_fmincon./repmat(sum(emission_fmincon,2),[1,O]);

%
%% E-M Algorithm, Multiple Observation Sequences
for EM_iter = 1:Total_EM_iters
    %% E-Step
    % Initialize for E-step quantities
    L_fmincon = containers.Map;
    H_fmincon = containers.Map;

    % Loop over all observation sequence
    for i_ObsSeq = 1:length(unique_ID)

        observed = dict_obs(string(unique_ID(i_ObsSeq)));
        T = length(observed);

        % Get Alpha and Beta from Forward-Backward Algorithm, which can be used to compute L and H (defined in Notes)
        Alpha_fmincon = Forward_Alg(observed,transition_fmincon,emission_fmincon,initial);
        Beta_fmincon = Backward_Alg(observed,transition_fmincon,emission_fmincon);

        % construct L; L(i,t)=P(S_t=i|O_1,...,O_T); rows are hidden states, columns are hidden state time steps
        L_tmp = Alpha_fmincon.*Beta_fmincon;
        L_fmincon(string(i_ObsSeq)) = L_tmp ./ sum(L_tmp);

        % construct H matrix; the most likely transition from t to t+1 given all observations
        % H is a 3-dimensional matrix, where H(i,j,t)=P(S_(t-1)=i,S_t=j|O_1,...,O_T)
        % i,j range from 1 to n, t range from 1 to T
        for t=1:T
            for i=1:N
                for j=1:N
                    H_mat_tmp(i,j,t)=(Alpha_fmincon(i,t)*Beta_fmincon(j,t+1)*emission_fmincon(j,observed(t))*transition_fmincon(i,j));
                end
            end
        end
        for t=1:T
            H_mat_tmp(:,:,t)=H_mat_tmp(:,:,t)./sum(H_mat_tmp(:,:,t),'all');
        end
        H_fmincon(string(i_ObsSeq)) = H_mat_tmp;
    end

    %
    %% M-Step
    % Optimize Transition Probability Matrix (IFR)
    transition_fmincon = reshape(transition_fmincon, [N*N,1]);
    tic
    transition_fmincon = fmincon(@(P)Likelihood_Func_H(P,H_fmincon),transition_fmincon,Fmincon_Aineq_Trans,Fmincon_bineq_Trans,Fmincon_Aeq_Trans,Fmincon_beq_Trans,Fmincon_lb_Trans,Fmincon_ub_Trans);
    total_time_fmincon = total_time_fmincon + toc;
    transition_fmincon = reshape(transition_fmincon,[N,N]);
    
    % Optimize Emission Probability Matrix (IFR)
    emission_fmincon = reshape(emission_fmincon, [N*O,1]);
    tic
    emission_fmincon = fmincon(@(P)Likelihood_Func_L(P, L_fmincon, dict_obs, unique_ID, O),emission_fmincon,Fmincon_Aineq_Emis,Fmincon_bineq_Emis,Fmincon_Aeq_Emis,Fmincon_beq_Emis,Fmincon_lb_Emis,Fmincon_ub_Emis);
    total_time_fmincon = total_time_fmincon + toc;
    emission_fmincon = reshape(emission_fmincon,[N,O]);

    %
end
transition_fmincon_final = transition_fmincon_final + transition_fmincon;
emission_fmincon_final = emission_fmincon_final + emission_fmincon(:, 1:O);

end

transition_fmincon_final = transition_fmincon_final./Total_MCMCIter;
transition_fmincon_final = transition_fmincon_final./repmat(sum(transition_fmincon_final,2),[1,N]);
emission_fmincon_final = emission_fmincon_final./Total_MCMCIter;
emission_fmincon_final = emission_fmincon_final./repmat(sum(emission_fmincon_final,2),[1,O]);

csvwrite(strcat(outfolder, '/transition_fmincon_',string(N),'.csv'),transition_fmincon_final)
csvwrite(strcat(outfolder, '/emission_fmincon_',string(N),'.csv'),emission_fmincon_final)
total_time_fmincon

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


% fmincon maximize likelihood function (minimize negative)
function Obj = Likelihood_Func_H(P,H)
    num_obs = H.Count;
    [N,~,~]=size(H('1'));
    P = reshape(P,[N,N]);
    result = 0;
    for idx_obs = 1:num_obs
        H_tmp = H(string(idx_obs));
        [~,~,T]=size(H_tmp);
        for t=1:T
            result = result + sum(H_tmp(:,:,t).*log(P),"all");
        end
    end
    Obj = -result;
end


function Obj = Likelihood_Func_L(P,L,SCATSEV_M, unique_ID, O)
    [N,~]=size(L('1'));
    P = reshape(P,[N,O]);
    result = 0;
    for idx_obs = 1:length(unique_ID)
        L_tmp = L(string(idx_obs));
        [~,T]=size(L_tmp);
        observed = SCATSEV_M(string(unique_ID(idx_obs)));
        for t=2:T
            for x=1:N
                result = result + L_tmp(x, (t-1))*log(P(x,observed(t-1)));
            end
        end
    end
    Obj = -result;
end

