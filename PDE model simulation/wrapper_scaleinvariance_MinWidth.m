%%%%% this is the wrapper to produce the paper figure for the evolution
%%%%% over times [30 60 90]
close all
clear all
tic

%% scale invariance test
%fix0=[1/6.2^2 1/5.8^2 1/5.4^2 1/5^2 1/4.6^2 1/4.2^2 1/3.8^2 1/3.4^2 1/3^2 1/2.6^2 1/2.2^2 1/1.8^2 1/1.4^2 1];
fix0=[1/3^2 1/2.6^2 1/2.2^2 1/1.8^2 1/1.4^2 1];
%fix0=[1/0.9^2 1/0.8^2 1/0.6^2];
fix0=1/0.9^2;
No=size(fix0,2);
maxrecord=zeros(1,No);
minrecord=zeros(1,No);
colonyrecord=zeros(1,No);
ringwidthrecord=zeros(1,No);
ringwidthrecord2=zeros(1,No);
ringwidthrecord3=zeros(1,No);

fix=1;parameters;colonyfinalvalue=zeros(No,2*param.N-1);

for i=1:No
    
    fix=fix0(i);
    parameters
    
    count=0;
    record_L=zeros(1,(param.N+1)/2);
    record_value=zeros(1,(param.N+1)/2);
    count2=0;
    record_L2=zeros(1,(param.N+1)/2);
    record_value2=zeros(1,(param.N+1)/2);
    
    [dataCe dataNu dataL dataAHL dataT dataP dataRFP dataCFP tdata]=spec_wrapper_functionS(param);
    %%
    dataCe=[dataCe(:,end:-1:2),dataCe];
    dataRFP=[dataRFP(:,end:-1:2),dataRFP];
    dataCFP=[dataCFP(:,end:-1:2),dataCFP];
    
    param.L=2*param.L;
    param.N=2*param.N-1;
    
    %%
    tfinall=find(dataNu>0.01);
    tfinal=min(tfinall(end)+27,param.tmax);
%       tfinal=param.tmax;
    
    record_l=dataRFP(tfinal,1:(param.N+1)/2).*dataCe(tfinal,1:(param.N+1)/2);
    colony_start=find((dataCe(tfinal,:)>max(max(dataCe(tfinal,:)))/2.5),1);
    colonyfinalvalue(i,:)=dataCe(tfinal,:);
    colony_1=find((dataCe(tfinal,:)>0.08),1);
    for j=colony_start:(param.N+1)/2-20
        if (record_l(1,j)-record_l(1,j-1))>0&&(record_l(1,j)-record_l(1,j-10))>0&&(record_l(1,j)-record_l(1,j+1))>0&&(record_l(1,j)-record_l(1,j+10))>0
            count=count+1;
            record_L(count)=j;
            record_value(count)=record_l(1,j);
        elseif (record_l(1,j)-record_l(1,j-1))<0&&(record_l(1,j)-record_l(1,j+1))<0&&(record_l(1,j)-record_l(1,j-10))<0&&(record_l(1,j)-record_l(1,j+10))<0
            count2=count2+1;
            record_L2(count2)=j;
            record_value2(count2)=record_l(1,j);
        else
        end
    end
    maxRFP_Radius=record_L(1);
    minRFP_Radius=record_L2(1);
    %     minRFP_Radius=max(record_L2);
    maxRFP_value=record_value(1);
    minRFP_value=record_value2(1);
    
    out=zeros(1,round(maxRFP_Radius)-round(colony_start)+1);
    for k=colony_start:round(maxRFP_Radius)
        out(k-colony_start+1)=abs(record_l(k)-minRFP_value);
    end
    mark1=find(out==min(min(out)));
    mark_out=sum(mark1)/size(mark1,2)+round(colony_start)-1;
    ring_out=mark_out*param.L/param.N;
    
    mark_in=minRFP_Radius;
    ring_in=minRFP_Radius*param.L/param.N;
    
    ringwidthrecord(i)=ring_in-ring_out;%%2.5 proportion after
    %%
    colonyrecord(i)=param.L/2-colony_start*param.L/param.N;
    maxrecord(i)=param.L/2-maxRFP_Radius*param.L/param.N;
    minrecord(i)=param.L/2-minRFP_Radius*param.L/param.N;
    ringwidthrecord2(i)=(maxRFP_Radius+minRFP_Radius)/2*param.L/param.N-(colony_start+maxRFP_Radius)/2*param.L/param.N; %%min-big slope
    ringwidthrecord3(i)=(maxRFP_Radius+minRFP_Radius)/2*param.L/param.N-ring_out; %%min-big slope
    save(['fix',num2str(1/fix^0.5),'.mat'])
    %%
    parameter_trails=figure;
    subplot(2,2,1);
    imagesc(real(dataRFP.*dataCe));
    set(gca,'FontSize',12)
    title('mCherry')
    subplot(2,2,2);
    imagesc(real(dataCFP.*dataCe));
    set(gca,'FontSize',12)
    title('CFP')
    subplot(2,2,3);
    imagesc(real(dataCe));
    set(gca,'FontSize',12)
    title('C')
    subplot(2,2,4);
    plot(real(dataCe(tfinal,:).*dataRFP(tfinal,:)));
    hold on
    plot(colony_start,0:0.001:1,'--g')
    hold on
    plot(mark_in,0:0.001:1,'--r')
    hold on
    plot(mark_out,0:0.001:1,'--r')
    set(gca,'FontSize',12)
    xlim([0 param.N])
    ylim([0,max(max(dataCe(tfinal,:).*(dataRFP(tfinal,:))))])
    title(['ring width',num2str(ringwidthrecord(i))])
    saveas(parameter_trails,['domain',num2str(1/fix^0.5),'.tif']);
end
%%
for i=1:No
    %     colony_obvious=find((colonyfinalvalue(i,:)>max(max(colonyfinalvalue(i,:)))/10),1);
    colony_obvious=find((colonyfinalvalue(i,:)>max(max(colonyfinalvalue(i,:)))/2.5),1);
    colonyrecord(i)=param.L/2-colony_obvious*param.L/param.N;
end

%%
domain=(1./fix0.^0.5);
colony=colonyrecord;
ringwidth=ringwidthrecord;
a=figure;
set(gca,'FontSize',25)
plot(domain,colony,'ob','MarkerSize',15,'LineWidth',2)
hold on
plot(domain,ringwidth,'or','MarkerSize',15,'LineWidth',2)
p1 = polyfit(domain(9:end),ringwidth(9:end),1);
slope1=p1(1,1);
intercept1=p1(1,2);
p2 = polyfit(domain,colony,1);
slope2=p2(1,1);
intercept2=p2(1,2);
x=0:0.1:max(domain)+0.1;
hold on
plot(x,slope1.*x+intercept1,'k','LineWidth',2)
hold on
plot(x,slope2.*x+intercept2,'k','LineWidth',2)
% hold on
% plot(domain,maxrecord,'k*')
xlim([0 max(domain)+0.1]);
ylim([0 round(max(colony))+1]);
% set(gca,'FontSize',14)
% xlim([min(domain)-0.1 max(domain)+0.1]);
% ylim([0 inf]);
hleg=legend('Colony Radius','Ring Width');
set(hleg,'Location','NorthWest')
xlabel('Domain Radius')
ylabel('Distance')
% title('Colony Radius vs. Domain Radius', 'FontSize', 12)
% print -depsc2 SimulationScale

b=figure;
set(gca,'FontSize',25)
plot(domain,ringwidth./colony,'or','MarkerSize',15,'LineWidth',2)
hold on
plot([0 max(domain)+0.1],mean(ringwidth./colony)*ones(1,2),'--k','LineWidth',2)
% xlim([min(domain)-0.1 max(domain)+0.1])
xlim([0 max(domain)+0.1])
ylim([0 1])
set(gca,'YTick',[0:0.2:1])
xlabel('Domain Radius')
ylabel('Proportion')
% title('Porportion of Ring width to the Colony size vs. Domain Radius', 'FontSize', 12)
print -depsc2 SimulationPorportion

std(ringwidth./colony)

toc