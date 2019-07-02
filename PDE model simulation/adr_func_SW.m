%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To integrate the advection,diffusion and reaction of the equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=adr_func_SW(t,vec,param,tt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% The parameters used in this function        %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=param.N;
m=param.m;
length=param.L;
h=length/N;
G1=param.G1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reassemble the vectors
Ly=vec(1:N);   
T=vec(N+1:2*N);
P=vec(2*N+1:3*N);
RFP=vec(3*N+1:4*N);
CFP=vec(4*N+1:5*N);
Ce=vec(5*N+1:6*N);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The FD stencils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lp=[Ly(2:N); Ly(N-1)]; % shifted to right (note L(N+1)=L(N-1) because of no-flux condition
Lm=[Ly(2); Ly(1:N-1)]; % shifted to left  (note L(0)=L(2) because of no-flux condition

Tp=[T(2:N); T(N-1)];
Tm=[T(2); T(1:N-1)];

Cep=[Ce(2:N); Ce(N-1)];
Cem=[Ce(2); Ce(1:N-1)];

Pp=[P(2:N); P(N-1)];
Pm=[P(2); P(1:N-1)];

RFPp=[RFP(2:N); RFP(N-1)];
RFPm=[RFP(2); RFP(1:N-1)];

CFPp=[CFP(2:N); CFP(N-1)];
CFPm=[CFP(2); CFP(1:N-1)];


rad_vec=linspace(0,param.L,param.N)'; % radius of each grid point

%% The integrands appearing in the integro-differential nutrient and AHL equations

a=find(Ce/max(Ce)<.9,1,'first');
dist=max((param.L/param.N*a-rad_vec),0);%param.L/param.N*max((a-rad_vec),0);

GK=(Ce>0).*(param.Kphi^param.exp_phi./(param.Kphi^param.exp_phi+dist.^param.exp_phi)+param.phi);%(Ce>0.1).*;

if tt<=10
    GK=ones(N,1).*(Ce>0);
end

Ce_mat=Ce.*meta_act(Ce,Nu, Ly,T,param).*rad_vec;  % the integrals are across the whole disc
T_mat=Ce.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*rad_vec./(1+P);


%% The rhs of the discretized PDE (advection parts)
%% Note the 1e-5 is there to avoid troubles with zero-values of Ce
%%%%advection part
dLydt_a=G1./(Ce+1e-5).*1/(4*h^2).*(Lp-Lm).*(Cep-Cem);
dTdt_a=G1./(Ce+1e-5).*1/(4*h^2).*(Tp-Tm).*(Cep-Cem);
dPdt_a=G1./(Ce+1e-5).*1/(4*h^2).*(Pp-Pm).*(Cep-Cem);
dRFPdt_a=G1./(Ce+1e-5).*1/(4*h^2).*(RFPp-RFPm).*(Cep-Cem);
dCFPdt_a=G1./(Ce+1e-5).*1/(4*h^2).*(CFPp-CFPm).*(Cep-Cem);


%%%diffusion part
dCedt_d=G1/h^2*(Cep-2*Ce+Cem)+ G1/2/h./rad_vec.*(Cep-Cem);
dCedt_d(1)=2*G1/h^2*(Cep(1)-2*Ce(1)+Cem(1)); % origin is special: no 1/r term (singular), but factor 2

%%%Reaction part

%% Cells %%%%%%%%
dCedt_r=1*Ce.*meta_act(Ce,Nu, Ly,T,param);

%% Nutrient %%%%%%%%
dNudt_r=-G3*length/(N-1)*2*pi*trapz(Ce_mat);  %for the intergration, we 
% use a simple trapezoidal rule;  the factor 2*pi comes from integral over
% disc

%% AHL %%%%%%%%%%%
dAdt_r=G4*length/(N-1)*2*pi*trapz(T_mat)- G5*A;
% again, the factor 2*pi comes from the integral over the disc

%% Lysozyme %%%%%%%%%%
dLydt_r= -Ly.*meta_act(Ce,Nu, Ly,T,param)- G6*Ly+ G7.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*A.^m./(1+A.^m); %

%% T7 %%%%%%%%%%%%
dTdt_r= -T.*meta_act(Ce,Nu, Ly,T,param)-G8*T+G9.*(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T))./(1+P);

%% P %%%%%%%%%%%%
dPdt_r=-P.*meta_act(Ce,Nu, Ly,T,param);

%% RFP (Lysozyme indicator) %%%%%%%%%%%%
dRFPdt_r=-RFP.*meta_act(Ce,Nu, Ly,T,param)+(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T)).*A.^m./(1+A.^m);

%% CFP (T7 indicator) %%%%%%%%%%%%
dCFPdt_r=-CFP.*meta_act(Ce,Nu, Ly,T,param)+(meta_act(Ce,Nu, Ly,T,param)).^omega.*GK.*(gamma+T./(1+T))./(1+P);



%%%Integrating three parts together

dLydt=dLydt_a+dLydt_r;
dTdt=dTdt_a+dTdt_r;
dPdt=dPdt_a+dPdt_r;
dRFPdt=dRFPdt_a+dRFPdt_r;
dCFPdt=dCFPdt_a+dCFPdt_r;
dCedt=dCedt_d+dCedt_r;
dNudt=dNudt_r;
dAdt=dAdt_r;

%% Assemble them into the output vector,send it back to the main function
out=[dLydt;dTdt;dPdt;dRFPdt;dCFPdt;dCedt;dNudt;dAdt];


function output = meta_act(C,N,L,T,param)

alpha = param.alpha;
beta  = param.beta;
G2    = param.G2;

output = (1-C).*N./(G2+N)./(1+alpha*T+beta*L);




