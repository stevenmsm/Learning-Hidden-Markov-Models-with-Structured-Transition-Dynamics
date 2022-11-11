# Learning-Hidden-Markov-Models-with-Structured-Transition-Dynamics
This repository contains code needed to replicate the results in manuscript: "Learning Hidden Markov Models with Structured Transition Dynamics". Currently, the manuscript is submitted and underreview. Below, brief discriptions on each MATLAB files are provided.

"MultiObs.m" contains the code to learn unknown transition and emission matrices (with multiple linear constriants) of HMM models when there are multiple emission states from a single hidden states. Conditional independence is assumed for each of the emission states. 

"OneObs_LDKBFISTA.m" constains the code to learn unknown transition and emission matrices (with multiple linear constraints) of HMM models when there is a single emission under a single hidden state (the classic HMM structure). The HMM parameters learning is done using our proposed algorithm LDKB-FISTA proposed in our submitted manuscript.

"OneObs_Fmincon.m" constains the code to learn unknown transition and emission matrices (with multiple linear constraints) of HMM models when there is a single emission under a single hidden state (the classic HMM structure). The HMM parameters learning is done using MATLAB Fmincon function.

"BaumWelch.m" constains the code to learn unknown transition and emission matrices (no multiple linear constraints) of HMM models when there is a single emission under a single hidden state (the classic HMM structure). The HMM parameters learning is done using the classic Baum Welch algorithm.

