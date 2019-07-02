%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% FRACTIONAL STEP SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is the main script to solve the PDE system describing the
%% cell-lysozyme-T7 dynamics; it is called by wrapper.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dataCe, dataNu, dataL, dataAHL, dataT, dataP, dataRFP, dataCFP, tdata ]=spec_wrapper_function_SW(param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters and Initial Distributions   %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=param.N; % load the parameter values

 Ce0=param.pertC; % initialize the distributions
 Nu0=param.Nu0;
L0=zeros(N,1);
AHL0=param.ahl;
 T0=param.T0;
P0=zeros(N,1);
CFP0=zeros(N,1);
RFP0=zeros(N,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tplot =1; %clf, drawnow, set(gcf,'renderer','zbuffer')
plotgap = round(tplot/param.dt);
dt = tplot/plotgap;
nplots = round(param.tmax/tplot);
options = odeset('InitialStep',0.1);
%options=odeset();

t=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nplots
    for n=1:plotgap
        t=t+dt;
        
        %inputs=[dataL;dataT;dataP;dataRFP;dataCFP;dataCe;dataNu;dataAHL];
        inputs=[L0;T0;P0;RFP0;CFP0;Ce0;Nu0;AHL0];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%% Advection, Diffusion and Reaction %%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        sol=ode45(@adr_func_SW,[0 dt],inputs,options,param,t);
        
        vec=(deval(sol,dt));
        L0=vec(1:N);
        T0=vec(N+1:2*N);
        P0=vec(2*N+1:3*N);
        RFP0=vec(3*N+1:4*N);
        CFP0=vec(4*N+1:5*N);
        Ce0=vec(5*N+1:6*N);
        Nu0=vec(6*N+1);
        AHL0=vec(6*N+2);
    

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% update the quasi-steady state    
        %%% by projecgting back onto the equilibrated L-T7-P 
        %%% concentrations 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        L_temp=1/2*(L0-param.G10*T0-param.G11+ sqrt((L0-param.G10*T0-param.G11).^2+4*param.G11*(L0+param.G12*P0)));
        P0=P0+1/param.G12*(L0-L_temp);
        T0=T0-1/param.G10*(L0-L_temp);
        L0=L_temp;

    end
    
    
    %% load into the history vector
    

    dataCe = real(Ce0);
    dataNu   = real(Nu0);
    dataAHL    = real(AHL0);
    dataL    = real(L0);
    dataT    = real(T0);
    dataP    = real(P0);
    dataRFP  = real(RFP0);
    dataCFP  = real(CFP0);
    tdata=t;

    %[i nplots]
    
end
