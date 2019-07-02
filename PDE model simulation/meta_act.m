%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is the metabolic capacity function, in the interval [0,1]
%% called \mu in the text
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function out=meta_act(C,N, L,T,param)

alpha=param.alpha;
beta=param.beta;
G2=param.G2;

out=(1-C).*N./(G2+N)./(1+alpha*T+beta*L);