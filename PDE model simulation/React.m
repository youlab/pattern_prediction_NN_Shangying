%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To integrate the advection part of the equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=React(t,vec, param,tt)

%% Carry over the parameters from param

N=param.N;
L=param.L;
m=param.m;

%% unpack the vectors used in the function
Ce=vec(1:N);
Ly=vec(N+1:2*N);
T=vec(2*N+1:3*N);
P=vec(3*N+1:4*N);
RFP=vec(4*N+1:5*N);
CFP=vec(5*N+1:6*N);
Nu=vec(6*N+1);
A=vec(6*N+2);

G3=param.G3;
G4=param.G4;
G5=param.G5;
G6=param.G6;
G7=param.G7;
G8=param.G8;
G9=param.G9;
gamma=param.gamma;
omega=param.omega;


%% The integrands appearing in the integro-differential nutrient and AHL equations
rad_vec=linspace(0,param.L,param.N)'; % radius of each grid point

a=find(Ce/max(Ce)<.9,1,'first');
dist=max((param.L/param.N*a-rad_vec),0);%param.L/param.N*max((a-rad_vec),0);

GK=(Ce>0).*(param.Kphi^param.exp_phi./(param.Kphi^param.exp_phi+dist.^param.exp_phi)+param.phi);%(Ce>0.1).*;

if tt<=10
    GK=ones(N,1).*(Ce>0);
end

Ce_mat=Ce.*meta_act(Ce,Nu, Ly,T,param).*rad_vec;  % the integrals are across the whole disc
T_mat=Ce.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*rad_vec./(1+P);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Right-hand side of the reaction ODEs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Cells %%%%%%%%
F_Ce=1*Ce.*meta_act(Ce,Nu, Ly,T,param);

%% Nutrient %%%%%%%%
F_Nu=-G3*L/(N-1)*2*pi*trapz(Ce_mat);  %for the intergration, we 
% use a simple trapezoidal rule;  the factor 2*pi comes from integral over
% disc

%% AHL %%%%%%%%%%%
F_A=G4*L/(N-1)*2*pi*trapz(T_mat)- G5*A;
% again, the factor 2*pi comes from the integral over the disc

%% Lysozyme %%%%%%%%%%
F_Ly= -Ly.*meta_act(Ce,Nu, Ly,T,param)- G6*Ly+ G7.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*A.^m./(1+A.^m); %

%% T7 %%%%%%%%%%%%
F_T= -T.*meta_act(Ce,Nu, Ly,T,param)-G8*T+G9.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T))./(1+P);

%% P %%%%%%%%%%%%
F_P=-P.*meta_act(Ce,Nu, Ly,T,param);

%% RFP (Lysozyme indicator) %%%%%%%%%%%%
F_RFP=-RFP.*meta_act(Ce,Nu, Ly,T,param)+(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*A.^m./(1+A.^m);

%% CFP (T7 indicator) %%%%%%%%%%%%
F_CFP=-CFP.*meta_act(Ce,Nu, Ly,T,param)+(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T))./(1+P);



%% Assemble them into the output vector
out=[F_Ce;F_Ly;F_T;F_P; F_RFP; F_CFP; F_Nu;F_A];

