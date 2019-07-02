%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% FRACTIONAL STEP SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is the main script to solve the PDE system describing the
%% cell-lysozyme-T7 dynamics; it is called by wrapper.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dataCe dataNu dataL dataAHL dataT dataP dataRFP dataCFP tdata ]=spec_wrapper_function(param)
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% allocate matrices that store the dynamic fields %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataCe=zeros(nplots+1,N); % cells
dataCe(1,:)=Ce0;

dataNu=zeros(nplots+1,1); % nutrient
dataNu(1)=param.Nu0;

dataAHL=zeros(nplots+1,1); % AHL

dataL=zeros(nplots+1,N); % Lysozyme

dataT=zeros(nplots+1,N); % T7
dataT(1,:)=T0;

dataP=zeros(nplots+1,N); %T7-Lysozyme complex

dataRFP=zeros(nplots+1,N); %RFP (coupled to lysozyme expression)

dataCFP=zeros(nplots+1,N);  % CFP (coupled to T7 expression)

tdata=0; t=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nplots
    for n=1:plotgap
        t=t+dt;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%% Fractional Step 1: Advection %%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        sol=ode45(@Advect,[0 dt],[L0;T0;P0;RFP0;CFP0],options,Ce0,param);
                      
        Nu1=Nu0;
        AHL1=AHL0;
        Ce1=Ce0;

        vec=(deval(sol,dt));
        L1=vec(1:N);
        T1=vec(N+1:2*N);
        P1=vec(2*N+1:3*N);
        RFP1=vec(3*N+1:4*N);
        CFP1=vec(4*N+1:5*N);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%% Fractional Step 2: Diffusion %%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        sol=ode15s(@Diffusion,[0 dt],Ce1,options,param);       
        
        Ce1=deval(sol,dt);  % upadte the diffusion part       
                       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%% Fractional Step 3: Reaction %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vec2=[Ce1; L1;T1;P1;RFP1;CFP1;Nu1;AHL1];
        sol=ode45(@React,[0 dt],vec2,options,param,t);
        vec3=deval(sol,dt); 
        
        Ce0=vec3(1:N);  % preparing the next iteration
        L0=vec3(N+1:2*N); 
        T0=vec3(2*N+1:3*N);
        P0=vec3(3*N+1:4*N);
        RFP0=vec3(4*N+1:5*N);
        CFP0=vec3(5*N+1:6*N);
        Nu0=vec3(6*N+1);
        AHL0=vec3(6*N+2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%% Fractional Step 4: 
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
    
    dataCe(i+1,:)=Ce0;
    dataNu(i+1)=Nu0;
    dataAHL(i+1)=AHL0;
    dataL(i+1,:)=L0;
    dataT(i+1,:)=T0;
    dataP(i+1,:)=P0;
    dataRFP(i+1,:)=RFP0;
    dataCFP(i+1,:)=CFP0;

    
    tdata=[tdata; t];
    [i nplots]
    
end