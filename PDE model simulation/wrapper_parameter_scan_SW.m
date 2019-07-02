%%%%% this is the wrapper to produce the paper figure for the evolution
%Shangying Wang
%last modified: 06/17/2017

close all
clear all
%tic
%setenv('SLURM_ARRAY_TASK_ID','10');
taskID=getenv('SLURM_ARRAY_TASK_ID');

%# iterations
offset = 1;


%reinitialize the random number generator based on the time and the process-ID of the matlab to make sure every cluster node run differently
rng('shuffle'); % seed with the current time
rngState = rng; % current state of rng

%%% deltaSeed can be any data unique to the process, 
%%% including position in the process queue
deltaSeed = uint32(feature('getpid'));

seed = rngState.Seed + deltaSeed;
rng(seed); % set the rng to use the modified seed,which would combine the current time with the process-ID of the matlab instance to generate the seed.

%% Parameters  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.L=80;  % length of interval
param.tmax=150; % time integration
param.N=2000+1;   % number of grid points; spatial step size h=L/(N-1)
param.dt=0.01; % time-stepping for fractional steps

n=10000;
%data_xls1=zeros(n,25);
for ii=1:n
       %metabolic burden term 1/(1+alpha*T+beta*L)^omega
    param.omega=0;
    param.alpha=rand*5; %0-5
    param.beta=rand*2000; %0-2000
    
    param.m=2;  % Hill coefficient for A^m/(1+A^m) in the L-equation
    param.gamma=0; %hill function constant in the L-equation
    
    %gene expression capacity parameters
    param.Kphi=rand*10+1e-4; %0-10
    param.exp_phi=rand*5; %0-5 randi(5)
    param.phi=0;
    
    L_pam=0.18898;%0.1+rand*(0.5-0.1);  %0.189
    alpha_c=1;%0.5+rand*(5-0.5); %0.5-5 %1
    alpha_T=80+rand*(8000-80); %0-8000 %G9
    alpha_L=90+rand*(9000-90); %0-9000 %G10,G11,G12
    const2=0.3;%0.1+rand*(1-0.1);  %constant coefficient for G2 %0.3
    %const3=0.1+rand*(10-0.1); %constant coefficient for G3 %0.1-10 %0.6814
    %const4=10+rand*(1500-10); %constant coefficient for G4 %10-1500 %141.4998
    d_A=0.3;%0.1+rand*(1-0.1); %G5 %0.1-1 %0.3
    d_L=0.0144;%0.01+rand*(0.05-0.01); %G6,G10,G11,G12 %0.005-0.02 %0.0144
    d_T=0.3;%0.1+rand*(1-0.1); %G8 %0.3
    KC=0.0025;%0.001+rand*(0.005-0.001);  % 0.001-0.005 %cell diffusion coefficient %0.0025;
    KT=50+rand*(5000-50); %0-5000 %G9,G10
    kD=0.037;%0.01+rand*(0.05-0.01); %G11 %0.037;
    KP=50+rand*(5000-50); %0-5000 %G12
    domainR=randi(6)*0.4+0.6; %1-3.0 domain radius
    %domainR=1.0;
    fix=1/domainR^2; %1/0.9^2; %fix0 is inversely correlated to domain size^2 %G3,G4
    

%     norm_idx_params=[10 5 2000 10 5 8000 9000 5000 5000 3.0];    
%      z=m_parameters.*norm_idx_params;
%      param.G4 = z(1);
%      param.alpha = z(2);
%      param.beta = z(3);
%      param.Kphi = z(4);
%      param.exp_phi = z(5);
%      alpha_T = z(6);
%      alpha_L = z(7);
%      KT = z(8);
%      KP = z(9);
%      domainR = z(10);
%      fix=1/domainR^2;
    %the dimensionless groups; see write-up for their role in the dynamics %%
    param.G1=KC/L_pam^2/alpha_c;%.07;%0.07;
    param.G2=const2;%.3;
    param.G3=0.0145/pi*fix;%const3*L_pam^3/alpha_c*fix;%0.0145/pi*fix;%0.01;
    G4=rand*10+1e-4; %0-10
    param.G4=G4*fix;   % 0-10 %const4*L_pam^3/alpha_c*fix;%3/pi*fix;
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
    
    
    %% scale invariance test
    % fix0=[1/6.2^2 1/5.8^2 1/5.4^2 1/5^2 1/4.6^2 1/4.2^2 1/3.8^2 1/3.4^2 1/3^2 1/2.6^2 1/2.2^2 1/1.8^2 1/1.4^2 1];
    %fix0=[1/0.9^2 1/0.8^2 1/0.6^2];
    % No=size(fix0,2);
    % maxrecord=zeros(1,No);
    % minrecord=zeros(1,No);
    % colonyrecord=zeros(1,No);
    % ringwidthrecord=zeros(1,No);
    % ringwidthrecord2=zeros(1,No);
    % ringwidthrecord3=zeros(1,No);
    
    % fix=1;
    % parameters;
    % colonyfinalvalue=zeros(No,2*param.N-1);
    
    
    %parameters_s
    RL=param.L;
    N=param.N;
    mirrL=2*param.L;
    mirrN=2*param.N-1;
    
    %solve the PDE
    [dataCe,data_Nu,data_L,data_AHL,data_T,data_P,data_RFP,data_CFP,t_data]=spec_wrapper_function_SW(param);
    
    totalmcherry=real(dataCe).*real(data_RFP);
    %pks=findpeaks(totalmcherry); %find peaks of the distribution
    [pks,locs,w1,p1]=findpeaks(totalmcherry); %find peaks of the distribution
    neg_v=sum(totalmcherry<0);
    %to check the results
%     if size(pks,1)>6 || neg_v>0 %if number of peaks larger than 2, rerun the model with reduced dt
%         param.dt=0.005; % time-stepping for fractional steps
%         %solve the PDE
%         [dataCe,data_Nu,data_L,data_AHL,data_T,data_P,data_RFP,data_CFP,t_data]=spec_wrapper_functionS(param);
%         totalmcherry=real(dataCe).*real(data_RFP);
%         neg_v=sum(totalmcherry<0);
% 	[pks,locs,w2,p2]=findpeaks(totalmcherry); 
	if size(pks,1)>10 || neg_v>0  %%do not record the results if distribution has negative value or the peak values are too much (probably artifact)
		continue;
	end
    %end

    
    
    %savedata
    %# output file name
    fName1 = fullfile(pwd, sprintf('datas/datas001_%s_%s_%s.mat',datestr(now,'dd_mm_yyyy_AM'),taskID,num2str(offset)));
%    fName2 = fullfile(pwd, sprintf('figures/figures001_%s_%s_%s.png',datestr(now,'dd_mm_yyyy_AM'),taskID,num2str(offset)));
    %# delete existing file
    if exist(fName1, 'file'), delete(fName1); end
    
    %mfile=matfile(fName1,'Writable',true);
    
    
    idx_end=floor(param.N/4)*4+1;
    idx_save=1:4:idx_end;
    %data_xls1(ii,:)=[param.alpha param.beta param.m  param.gamma param.Kphi param.exp_phi param.phi L_pam alpha_c alpha_T alpha_L const2 const3 const4 d_A d_L d_T KC KT kD KP domainR];
    %[param.alpha param.beta param.m param.omega param.gamma param.Kphi param.exp_phi param.phi L_pam alpha_c alpha_T alpha_L const2 const3 const4 d_A d_L d_T KC KT kD KP domainR];

    %norm_idx_params=[5 2000 20 5 0.5 5 8000 9000 1 10 1500 1 0.05 1 0.005 5000 0.05 5000 4.0]; %19 parameters
    norm_idx_params=[10 5 2000 10 5 8000 9000 5000 5000 3.0]; %9 parameters
    m_parameters=[ G4 param.alpha param.beta param.Kphi param.exp_phi alpha_T alpha_L KT KP domainR]./norm_idx_params;
    m_data=[t_data;data_Nu;dataCe(idx_save);data_RFP(idx_save);data_CFP(idx_save)];
    %data_xls2=[data_AHL(tfinal),dataCe(tfinal,idx_save),data_RFP(tfinal,idx_save),data_CFP(tfinal,idx_save)]';
    %xlswrite(fName1, data_xls2, 1, sprintf('A%d',1));
    save(fName1,'m_parameters','m_data');
    % saved_matrix=[zeros(size(m_data,1),1) m_data];
    % idxs=length(m_parameters);
    % saved_matrix(1:idxs,1)=m_parameters;
    % csvwrite(fName1,saved_matrix);
    


  %plot and save figures
    %figure(1)
 %   xx=linspace(0,param.L,param.N)';
 %   f = figure('visible','off');
 %   plot(xx,totalmcherry,'LineWidth',3)
 %   saveas(f, fName2, 'png')
    
 %   set(gca,'fontsize',20)
 %   xlim([0 param.L]) 
    %# calculate cell range to fit matrix (placed below previous one)
    %cellRange = xlcalcrange('A1', offset,0, 1,data_xls);
    offset = offset + 1;
    
end

%toc
%# output file name
%fName = fullfile(pwd, sprintf('parameter_file_%s_%s.mat',datestr(now,'dd_mm_yyyy'),taskID));
%# delete existing file
%if exist(fName, 'file'), delete(fName); end
%save(fName,'data_xls1');
