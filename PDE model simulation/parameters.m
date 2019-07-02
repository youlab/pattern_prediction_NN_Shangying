%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PARAMETERS AND INITIAL FIELDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.L=80;  % length of interval
param.tmax=180; % time integration
param.N=2400+1;   % number of grid points; spatial step size h=L/(N-1) 
param.dt=.1; % time-stepping for fractional steps

%%%%%%%% the parameters %%%%%%%%%%%%%%%%%
% 
% param.m=2;  % Hill coefficient for a^m/(1+a^m) in the L-equation
% param.omega=0;
% param.alpha=0;
% param.beta=0;
% param.gamma=0;
% param.ahl=0;
% 
% param.Kphi=2;
% param.exp_phi=1;
% param.phi=0;
% 
% %the dimensionless groups; see write-up for their role in the dynamics %%
% param.G1=.07;%0.07;
% param.G2=.3;
% param.G3=0.0145/pi*fix;%0.01;
% param.G4=3/pi*fix;
% param.G5=0.3;                       
% param.G6=0.0144;                  
% param.G7=param.G6;
% param.G8=0.3;                     
% param.G9=5;
% param.G10=0.0038;
% param.G11=8.64e-5;
% param.G12=0.0013;


param.omega=0;
param.alpha=1.080296452582; %0-5
param.beta=1568.382561113985; %0-2000

%gene expression capacity parameters
param.Kphi=1.820769438301 ; %0-20
param.exp_phi=3; %1-5 randi(5)
param.phi=0;

%other parameters
L_pam=0.126527138348;  %0.189
alpha_c= 2.216483175632; %0.5-5 %1
alpha_T=5481.376089653286; %0-8000 %G9
alpha_L=8599.832485684996; %0-9000 %G10,G11,G12
const2=0.633543225105;  %constant coefficient for G2 %0.3
const3=1.008328692075; %constant coefficient for G3 %0.1-1 %0.6814
const4=1064.781764195363; %constant coefficient for G4 %141.4549
d_A= 0.257837667644 ; %G5 %0.1-1 %0.3
d_L=0.048419121645; %G6,G10,G11,G12 %0.005-0.02 %0.0144
d_T= 0.554038332647; %G8 %0.3
KC=0.001448665929;  % 0.001-0.005 %cell diffusion coefficient %0.0025;
KT=898.391319908734; %0-5000 %G9,G10
kD=0.042893992884; %G11 %0.037;
KP= 1173.549336076218 ; %0-5000 %G12
fix=1/3^2; %1/0.9^2; %fix0 is inversely correlated to domain size^2 %G3,G4
param.m=2;  % Hill coefficient for A^m/(1+A^m) in the L-equation
param.gamma=0; %hill function constant in the L-equation





%the dimensionless groups; see write-up for their role in the dynamics %%
param.G1=KC/L_pam^2/alpha_c;%.07;%0.07;
param.G2=const2;%.3;
param.G3=const3*L_pam^3/alpha_c*fix;%0.0145/pi*fix;%0.01;
param.G4=const4*L_pam^3/alpha_c*fix;%3/pi*fix; 
param.G5=d_A/alpha_c;%0.3;                       
param.G6=d_L/alpha_c;%0.0144;                  
param.G7=param.G6;
param.G8=d_T/alpha_c;%0.3;                     
param.G9=alpha_T/(alpha_c*KT);%5;
param.G10=KT*d_L/alpha_L;%0.0038;
param.G11=d_L/alpha_L/kD; %8.64e-5;
param.G12=KP*d_L/alpha_L;%0.0013;
param.GM=1/alpha_c;
param.GC=param.GM;


%% Initial values  %%%%%%%%%%%%%%%%%%%%%%%%%%
param.ahl=0; %initial AHL concentration
param.Nu0=1; % initial nutrient concentration


%%%%%%% creating initial Cell distribution
xx=linspace(0,pi,param.N)';
%param.pertC=.2*exp(-(xx).^2/.005)';
param.pertC=.2*exp(-(xx).^2/.005);
%param.pertC=1+cos(xx);


%%%%%% in0tial T7 distribution %%%%%%%%
param.T0=0.1*param.pertC;%0.1*param.pertC;
