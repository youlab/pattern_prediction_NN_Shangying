clear all
close all

theta=linspace(0,2*pi,10);
axistheta=linspace(0,2*pi,500);
farthest=22;
set(gca,'FontSize',35)
plot(farthest*cos(axistheta),farthest*sin(axistheta),'--k')
hold on
plot(2*cos(axistheta),2*sin(axistheta),'--k')
hold on
%text(0,0,'0')
hold on
labels ={'G_4' '\alpha'  '\beta' 'n' 'K_\phi'   '\alpha_T' '\alpha_L' 'K_T' 'K_P'};
%labels1={'10' '5' '2000' '5' '10'  '8e3' '9e3' '5e3' '5e3'};
for j=1:9
    plot([2*cos(theta(j)) (farthest+1)*cos(theta(j))], [2*sin(theta(j)) (farthest+1)*sin(theta(j))],':k')
    hold on
%     text((farthest+3)*cos(theta(j)),(farthest+3)*sin(theta(j)), labels1{j});
    text((farthest+5)*cos(theta(j)),(farthest+5)*sin(theta(j)), labels{j});
    %     text((1.5)*cos(theta(j)),(1.5)*sin(theta(j)), labels2{j});
    %     hold on
end
cc=0;
for h=3:3%1:9
    if h==1
        load('SCORE16.mat')
    elseif h==2
        load('SCORE15.mat')
    elseif h==3
        load('SCORE14.mat')
    elseif h==4
%         load('SCORE26.mat')
    elseif h==5
        load('SCORE25.mat')
    elseif h==6
        load('SCORE24.mat')
    elseif h==7
        load('SCORE36.mat')
    elseif h==8
        load('SCORE35.mat')
    elseif h==9
        load('SCORE34.mat')
    end
    score=real(SCORE(1:find(SCORE(:,2)==0,1)-1,:));
    size(score)
    %     scoreOriginal=[score(:,3) score(:,2) score(:,12) score(:,19) score(:,14) score(:,15) score(:,16) score(:,17)];
    scoreE=[score(:,3)*2 score(:,2) score(:,12)*4 score(:,19)*10 score(:,8)*4 score(:,14)/400 score(:,15)/450 score(:,16)/250 score(:,17)/250 score(:,3)*2];
    
    for j=1:size(score,1)
        plot((scoreE(j,:)+2).*cos(theta),(scoreE(j,:)+2).*sin(theta),'-r')
        hold on
        if isnan(SCORE(j,10))==0&&isnan(SCORE(j,11))==0&&SCORE(j,11)>0&&SCORE(j,13)<18&&SCORE(j,13)>1
            %%%%%%
            record_l=mycell{1,j}(param.tmax+1,1:(1+param.N)/2);
            count=0;
            LL=0;
            %         count2=0;
            colony_start=find((dataCe(end,:)>max(max(dataCe(end,:)))/10),1);
            if colony_start<3
                colony_start=3;
            else
            end
            for jj=colony_start:(1+param.N)/2-50
                if (record_l(1,jj)-record_l(1,jj-1))>0&&(record_l(1,jj)-record_l(1,jj-2))>0&&(record_l(1,jj)-record_l(1,jj+1))>0&&(record_l(1,jj)-record_l(1,jj+2))>0
                    count=count+1;
                    LL(count)=jj;
                    %             elseif (record_l(1,jj)-record_l(1,jj-1))<0&&(record_l(1,jj)-record_l(1,jj-2))<0&&(record_l(1,jj)-record_l(1,jj+1))<0&&(record_l(1,jj)-record_l(1,jj+2))<0
                    %                 count2=count2+1;
                    %                 L2(count2)=jj;
                else
                end
            end
            if std(LL)<10 && mean(LL)<570 && LL(1)>0
                cc=cc+1;
                SS(cc,:)=score(j,:);
                SSmycell{cc}=mycell{j};
            end
        else
             %%%
        end
    end
    axis equal
    set(gca, 'XTick', [],'YTick', []);
    
end
xlim([-28 28])
ylim([-28 28])

scoreEE=[SS(:,3)*2 SS(:,2) SS(:,12)*4 SS(:,19)*10 SS(:,8)*4 SS(:,14)/400 SS(:,15)/450 SS(:,16)/250 SS(:,17)/250 SS(:,3)*2];
for i=1:size(SS,1) 
plot((scoreEE(i,:)+2).*cos(theta),(scoreEE(i,:)+2).*sin(theta),'-b')
hold on
end
print -depsc2 PatternOverAll


% figure;
% babe=0;
% for j=1:size(SS,1)
%     if babe<25
%         babe=babe+1;
%         subplot(5,5,babe)
%         imagesc(real(SSmycell{j}));
%         title(num2str(j))
%         hold on
%     else
%         figure('units','normalized','outerposition',[0 0 1 1])
%         babe=0;
%     end
% end
