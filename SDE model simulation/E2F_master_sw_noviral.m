%Initialize Environment
%clear;clc;close all;
tic
%setenv('SLURM_ARRAY_TASK_ID','10');
taskID=getenv('SLURM_ARRAY_TASK_ID');

%reinitialize the random number generator based on the time and the process-ID of the matlab to make sure every cluster node run differently
rng('shuffle'); % seed with the current time
rngState = rng; % current state of rng

%%% deltaSeed can be any data unique to the process,
%%% including position in the process queue
deltaSeed = uint32(feature('getpid'));

seed = rngState.Seed + deltaSeed;
rng(seed); % set the rng to use the modified seed,which would combine the current time with the process-ID of the matlab instance to generate the seed.
Sfinal=1.0;
dt=0.01;               %Time step
endTime=50;             %Time span to run each individual trace in hours
sigma=2;                %Scaling for intrinsic 1
delta=100;                %Scaling for extrinsic 10


%Number Time Courses
TRIALS = 10000;
vlabels={'Myc';'E2Fm';'E2Fp';'CD';'RB';'CE';'RP';'RE';'AF';'MR'};
%DO WORK 43 parameters
param_norm=[5.0 0.25 2.0 0.75 2.0 0.15 2.5 1.75 0.9 900 18 90 90 0.035 0.075 0.05 4.0 1.0 5.0 2.5 1.25 1.25 0.75 12.5 2.5 0.75 500 3.0 0.05 0.05 4.6 4.6 0.75 1.25 1.75 7.5 7.5 0.3 0.3 0.15 3.5 0.6 14.0];

examples=100000;
for ll=1:examples
    %    kMYCspan = 9;
    V=10^-15; N=6.02E17;%N:items_per_micromole
    Z=V*N;
    x0=Z*[0 0 0 0 0 0 0 0 0 0]; %Initial conditions
    
    %randomly generate all the parameters
    %Kinetic Parameters
    %Sfinal=0.1+rand*(2.0-0.1);%[0:0.25:10]./100;            %Final serum level
    kMC=0.2+rand*(5.0-0.2);                 %(uM*h^-1) Myc synthesis rate
    kS=0.01+rand*(0.25-0.01);                %(uM*h^-1) EFm synthesis rate (serum)
    kEFm=0.08+rand*(2.0-0.08);               %(uM*h^-1) EFm synthesis rate
    kb=0.03+rand*(0.75-0.03);                %(uM*h^-1) EFp-independent synthesis rate for E2Fm (Serum)
    kEFp=0.08+rand*(2.0-0.08);               %(h^-1)    E2F translation rate
    kCD=0.006+rand*(0.15-0.006);               %(uM*h^-1) CYCD synthesis rate (MYC)
    kCDS=0.1+rand*(2.5-0.1);              %(uM*h^-1) CYCD synthesis rate (Serum)
    kCE=0.07+rand*(1.75-0.07);               %(uM*h^-1) Cyclin E synthesis rate
    kRB=0.036+rand*(0.9-0.036);               %(uM*h^-1) Basal RB synthesis rate
    kRE=36+rand*(900-36);                %((uM*h)^-1) RB-E2F complex formation rate
    kRBDP=0.72+rand*(18.0-0.72);              %(uM*h^-1) RB dephosphorylation rate
    kRBP1=3.6+rand*(90-3.6);               %(h^-1) RB phosphorylation rate mediated by CYCD
    kRBP2=3.6+rand*(90-3.6);               %(h^-1) RB phosphorylation rate mediated by CYCE
    kAFb=0.001+rand*(0.035-0.001);             %(uM*h^-1) Basal ARF synthesis rate
    kAFEF=0.003+rand*(0.075-0.003);            %(uM*h^-1) E2F-dependent ARF synthesis rate
    kAFMC=0.002+rand*(0.05-0.002);                %(uM*h^-1) Myc-dependent ARF synthesis rate
    kMREF=0.16+rand*(4.0-0.16);                %(uM*h^-1) E2F-dependent miRNA synthesis rate
    kMRMC=0.04+rand*(1.0-0.04);              %(uM*h^-1) Myc-dependent miRNA synthesis rate
    KAFMC=0.2+rand*(5.0-0.2);              %(uM) Half-maximal constant for MYC-dependent ARF induction
    KAFEF=0.1+rand*(2.5-0.1);              %(uM) Half-maximal constant for E2F-dependent ARF induction
    KMRMC=0.05+rand*(1.25-0.05);             %(uM) Half-maximal constant for Myc-dependent miRNA induction
    KMREF=0.05+rand*(1.25-0.05);             %(uM) Half-maximal constant for E2F-dependent miRNA induction
    KMC=0.03+rand*(0.75-0.03);               %(uM) Half-maximal constant for E2F induction by MYC/E2F
    KMC1=0.5+rand*(12.5-0.5);               %(uM) Half-maximal constant for E2F induction by MYC
    KS=0.1+rand*(2.5-0.1);                 %(%)  Half-maximal constant for serum
    KEF=0.03+rand*(0.75-0.03);              %(uM) Half-maximal constant for E2F autoinduction
    KR=20+rand*(500-20);                  %(uM) Half-maximal constant for repression of EFm by MYC (adjusted to match experiment)
    KMR=0.12+rand*(3.0-0.12);                %(uM) Half-maximal constant for miRNA suppression of EFp
    KAFR=0.002+rand*(0.05-0.002);              %(uM) ARF-mediated EFp decay
    KRP=0.002+rand*(0.05-0.002);               %(uM) Half-maximal constant for RP dephosphorylation
    KCD=0.184+rand*(4.6-0.184);               %(uM) Half-maximal constant for RB phosphorylation by Cyclin D
    KCE=0.184+rand*(4.6-0.184);               %(uM) Half-maximal constant for RB phosphorylation by Cyclin E
    KMCCD=0.03+rand*(0.75-0.03);             %(uM) Half-maximal constant for CYCD synthesis by MYC
    
    
    dEFm=0.05+rand*(1.25-0.05);              %(h^-1) Degradation constant for E2Fm
    dEFp=0.07+rand*(1.75-0.07);              %(h^-1) Degradation constant for E2Fp
    dCD=0.3+rand*(7.5-0.3);                 %(h^-1) Degradation constant for Cyclin D
    dCE=0.3+rand*(7.5-0.3);                %(h^-1) Degradation constant for Cyclin E
    dRB=0.012+rand*(0.3-0.012);                %(h^-1) Degradation constant for RB
    dRP=0.012+rand*(0.3-0.012);               %(h^-1) Degradation constant for Phospho RB
    dRE=0.006+rand*(0.15-0.006);               %(h^-1) Degradation constant for RB-E2F complex
    dMC=0.14+rand*(3.5-0.14);                %(h^-1) Degradation constant for MYC
    dAF=0.024+rand*(0.6-0.024);             %(h^-1) Degradation constant for ARF
    dMR=0.56+rand*(14.0-0.56);                %(h^-1) Degradation constant for miRNA
    
    
    %Converting into molecule numbers
    paraset=[kMC,kS,kEFm,kb,kEFp,kCD,kCDS,kCE,kRB,kRE,kRBDP,kRBP1,kRBP2,kAFb,kAFEF,kAFMC,kMREF,kMRMC,...
        KAFMC,KAFEF,KMRMC,KMREF,KMC,KMC1,KS,KEF,KR,KMR,KAFR,KRP,KCD,KCE,KMCCD,...
        dEFm,dEFp,dCD,dCE,dRB,dRP,dRE,dMC,dAF,dMR];
    
    mat=zeros(TRIALS,10);
    %r1=zeros(1,TRIALS);
    flag=zeros(TRIALS,1);
    for ii = 1:TRIALS
        
        [Tspan,x]=StochSimE2F_sw(dt,endTime,sigma,delta,x0,Sfinal,paraset,Z); %SERUM STIMULATION
        mat(ii,:)=x(end,:)/Z;%log10(x(end,:)+1e-20);
        
        %     if mat(ii,1)<xmin || mat (ii,1)>xmax || mat(ii,2)<ymin || mat (ii,2)>ymax
        %         flag(ii)=1;
        %     end
    end
    
    xmin = [0,0,0,0,0,0,0,0,0,0]; % lower bound
    xmax = [50,80,100,20,30,100,300,300,20.0,10.0]; % upper bound
    npoints=1000;
    dis_data=zeros(10,npoints);
    max(mat,[],1)
    f1 = figure('visible','on');
    set(gcf, 'Position', [1000, 1000, 1000, 500])
    for zz=1:10
        subplot(2,5,zz)
        hold on;
        %h=histfit(mat(:,zz),nbins,'kernel');
        nn = numel(mat(:,zz));
        nbins = ceil(sqrt(nn));
        [bincounts,binedges] = histcounts(mat(:,zz),nbins);
        bincenters = binedges(1:end-1)+diff(binedges)/2;
        
        % Plot the histogram with no gap between bars.
        hh = bar(bincenters,bincounts,1);
        
        % Normalize the density to match the total area of the histogram
        binwidth = binedges(2)-binedges(1); % Finds the width of each bin
        area = nn * binwidth;
        thred=min(2.0,xmax(zz)/10);
        edges = [linspace(0,thred*0.99,100),linspace(thred,xmax(zz),npoints-100)];
        %dis_data(zz,:) = pdf(pd,edges)*trapz(h(1).XData,h(1).YData);
        pd = fitdist(mat(:,zz),'kernel');
        dis_data(zz,:) = pdf(pd,edges);
        idx1=find(dis_data(zz,:));
        xp=edges(idx1(1):idx1(end));
        yp=dis_data(zz,idx1(1):idx1(end))*area;
        plot(xp,yp,'LineWidth',2)
        xlabel(vlabels(zz));
        set(gca,'linewidth',2)
        set(gca,'FontSize', 15)
    end
    
    fName = fullfile(pwd, sprintf('figureD/distribution_myc_ef_%s_%s_%s.png',datestr(now,'dd_mm_yyyy_AM'),taskID,num2str(ll)));
    saveas(f1,fName, 'png')
    
    fName1 = fullfile(pwd, sprintf('datas/datas_myc_e2f_%s_%s_%s.mat',datestr(now,'dd_mm_yyyy_AM'),taskID,num2str(ll)));
    if exist(fName1, 'file'), delete(fName1); end
    paraset_list=paraset./param_norm;
    save(fName1,'paraset_list','dis_data');
end

toc
%fclose all;
