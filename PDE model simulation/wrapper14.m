%%%%% this is the wrapper to produce the paper figure for the evolution
%%%%% over times [30 60 90]
close all
clear all
%% scale invariance test
fix=1;
alpha_c=1;
d_L=0.0144;
k_on=400;
k_off=10800;
SCORE=zeros(1000,19);
mycell=cell(1,1000);
SCORE2=zeros(1000,19);
mycell2=cell(1,1000);
for jj=1:10
    %%
    alpha_T=300+rand*(8000-300);
    alpha_L=800+rand*(9000-800);
    KT=10+rand*(100-10);
    KP=10+rand*(400-10);
    K=6+rand*(400-6);
    %%
    param.omega=0;
    param.Kphi=rand*20+1e-4;
    param.exp_phi=rand*5;
    param.phi=rand*2;
    
    GG4=rand*10+1e-4;   % 0-10
    param.G9=alpha_T/alpha_c/KT;
    param.G10=KT/(alpha_L/d_L);
    param.G11=d_L/alpha_L*k_off/k_on;
    param.G12=KP/(alpha_L/d_L);
    param.alpha=rand*5;
    param.beta=0;
    
    for fix=[1 1/3^2]
        parameters
        param.G3=0.0145/pi*fix;%0.01;
        param.G4=GG4*fix;
        
        [dataCe dataNu dataL dataAHL dataT dataP dataRFP dataCFP tdata]=spec_wrapper_func(param);
        dataCe=[dataCe(:,end:-1:2),dataCe];
        dataRFP=[dataRFP(:,end:-1:2),dataRFP];
        dataCFP=[dataCFP(:,end:-1:2),dataCFP];
        
        param.L=2*param.L;
        param.N=2*param.N-1;
        
        record_L2=zeros(1,(1+param.N)/2);
        maxRFP=zeros(1,param.tmax+1);
        minRFP=zeros(1,param.tmax+1);
        
        TotalmCherry=(dataRFP).*dataCe;
        
        record_l=TotalmCherry(:,1:(1+param.N)/2);
        
        for h=1:param.tmax+1
            count=0;
            count2=0;
            record_L=zeros(1,(1+param.N)/2);
            record_L2=zeros(1,(1+param.N)/2);
            colony_start=find((dataCe(h,:)>max(max(dataCe(h,:)))/10),1);
            if colony_start<3
                colony_start=3;
            else
            end
            for j=colony_start:(1+param.N)/2-20
                if (record_l(h,j)-record_l(h,j-1))>0&&(record_l(h,j)-record_l(h,j-2))>0&&(record_l(h,j)-record_l(h,j+1))>0&&(record_l(h,j)-record_l(h,j+2))>0
                    count=count+1;
                    record_L(count)=j;
                elseif (record_l(h,j)-record_l(h,j-1))<0&&(record_l(h,j)-record_l(h,j-2))<0&&(record_l(h,j)-record_l(h,j+1))<0&&(record_l(h,j)-record_l(h,j+2))<0
                    count2=count2+1;
                    record_L2(count2)=j;
                else
                end
            end
            maxRFP(1,h)=sum(record_L)/count;
            minRFP(1,h)=sum(record_L2)/count2;
        end
        
        maxRFP_final=maxRFP;minRFP_final=minRFP;
        
        maxRFP_final=param.L/2-maxRFP_final.*param.L./param.N;
        minRFP_final=param.L/2-minRFP_final.*param.L./param.N;
        time=0:1:param.tmax;
        
        count=0;
        for i=1:param.tmax+1
            if isnan(maxRFP_final(1,i))==0&& isnan(minRFP_final(1,i))==0
                count=count+1;
            else
            end
        end
        importantmax=zeros(1,count);
        importantmin=zeros(1,count);
        count=0;
        for i=1:param.tmax+1
            if isnan(maxRFP_final(1,i))==0&& isnan(minRFP_final(1,i))==0
                count=count+1;
                importantmax(count)=maxRFP_final(1,i);
                importantmin(count)=minRFP_final(1,i);
            else
            end
        end
        score1=std(importantmax)/mean(importantmax);
        score2=std(importantmin)/mean(importantmin);
        meanvalue=mean(importantmax);
        max_ahl=max(max(dataAHL));
        if fix==1
            SCORE(jj,:)=[param.omega param.Kphi param.G4 param.G9 param.G10 param.G11 param.G12 param.alpha param.beta score1 score2 param.exp_phi max_ahl alpha_T alpha_L KT KP K param.phi];
            if isnan(SCORE(jj,10))==0&&isnan(SCORE(jj,11))==0
                mycell{jj}=TotalmCherry;
            else
            end
        else
            SCORE2(jj,:)=[param.omega param.Kphi param.G4 param.G9 param.G10 param.G11 param.G12 param.alpha param.beta score1 score2 param.exp_phi max_ahl alpha_T alpha_L KT KP K param.phi];
            if isnan(SCORE2(jj,10))==0&&isnan(SCORE2(jj,11))==0
                mycell2{jj}=TotalmCherry;
            else
            end
        end
    end
    
    save SCORE15
end