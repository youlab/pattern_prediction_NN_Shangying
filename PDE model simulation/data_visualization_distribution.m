%preprocess data and extract data to be feed into NNs

% Initialization
clear ; close all; clc
%cd ('I:\SW\PDE_data_storage\05_cluster_nan_fix1')
%cd I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\new_code_13_params_022018\datas
%cd ('I:\SW\PDE_data_storage\01_cluster_old')

readFiles = dir('datas\datas*.mat');
numfiles = length(readFiles);
%mydata = cell(1, numfiles);
%myparameter = cell(1, numfiles);
%norm_idx_params=[5 2000 20 5 0.5 5 8000 9000 1 10 1500 1 0.05 1 0.005 5000 0.05 5000 5.0];
%[param.alpha param.beta param.Kphi param.exp_phi L_pam alpha_c alpha_T alpha_L const2 const3 const4 d_A d_L d_T KC KT kD KP domainR];
%total varying parameters: 19
ll=501;
sample=501;

lens=13+sample+1+1;
saved_matrix=zeros(numfiles,lens);
local_peak_num=zeros(numfiles,1);

for k = 1:numfiles%numfiles
    k
    mat_name=readFiles(k).name;
    load(strcat('datas',mat_name));
    parameters=m_parameters;
    %data_Nu=m_data(2,end);
    dataCe=m_data(3:(sample+2),end);
    data_RFP=m_data((ll+3):(ll+sample+2),end);
    TotalmCherry=real(data_RFP.*dataCe);
    [peakval,~]=max(TotalmCherry);
    log_peak=log(peakval);
    norm_mCherry=TotalmCherry'/peakval;
    norm_mCherry_flip=[fliplr(norm_mCherry), norm_mCherry(:,2:end)]; %get the full distribution
    dataCe_flip=[flipud(dataCe);dataCe(2:end)];
    colony_start=find((dataCe_flip>max(dataCe)/2.5),1);
    [peaks,locs,widths,proms]=findpeaks(norm_mCherry_flip,'MinPeakDistance',50,'MinPeakProminence',0.1);
    
    if isempty(locs) || sum(norm_mCherry_flip<0)>0
        local_peak_n=0;
    else
        local_peak_n=length(locs);
    end
    
    
    local_peak_num(k)=local_peak_n;
    saved_matrix(k,:)=[parameters norm_mCherry log_peak local_peak_num(k)];
end

idx1=saved_matrix(:,end)==0;
sum(idx1)
saved_matrix(idx1,:)=[];


saved_matrix1=saved_matrix(1:1000,:);
saved_matrix2=saved_matrix(1:10000,:);
saved_matrix3=saved_matrix(1:100000,:);
saved_matrix4=saved_matrix(1:400000,:);
saved_matrix5=saved_matrix(400001:440000,:);
saved_matrix6=saved_matrix(440001:480000,:);
X = saved_matrix(:,1:10);
TotalmCherry=saved_matrix(:,11:(end-2));
pval=saved_matrix(:,end-1);
%pval2=saved_matrix2(:,end-1);

% Randomly select 36 data points to display
sel = randperm(size(TotalmCherry,1));
%sel=randperm(size(saved_matrix3,1));
sel = sel(1:70);

xdis=TotalmCherry(sel,:);
ydis=pval(sel);

[m n] = size(xdis);

% Compute number of items to display
display_rows = 7;
display_cols = 10;
% display_rows = 5;
% display_cols = ceil(m / display_rows);
% Copy each example into a patch on the display array
curr_ex = 1;
figure(1);
hold on;
for j = 1:display_rows
    for i = 1:display_cols
        
        % Copy the patch
        subplot(display_rows,display_cols,curr_ex);
        %tCherry=exp(ydis(curr_ex))*[fliplr(xdis(curr_ex,:)),xdis(curr_ex,2:end)];
        tCherry=exp(ydis(curr_ex))*xdis(curr_ex,:);
        plot(real(tCherry),'LineWidth',3);
        set(gca, 'XTick', [], 'YTick', [])
        xlim([0 length(tCherry)]);
        
        
        
        curr_ex = curr_ex + 1;
    end
    if curr_ex > m,
        break;
    end
end
set(gca,'FontSize',20)




%display the variaty of peak value range
[list,idx]=sort(pval);
%pv=log10(exp(peakval));
figure(2)
semilogy(exp(list))
%plot(list)
title('peak value range','linewidth',4)


figure(3)
histogram(pval)
title('peak value distribution','linewidth',4)

file_name = fullfile('I:\PDE_model', sprintf('all_data.csv'));
csvwrite(file_name,saved_matrix1);
% file_name = fullfile('I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\datasize_analysis', sprintf('data_031618_10000.csv'));
% csvwrite(file_name,saved_matrix2);
% file_name = fullfile('I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\datasize_analysis', sprintf('data_031618_100000.csv'));
% csvwrite(file_name,saved_matrix3);
% file_name = fullfile('I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\datasize_analysis', sprintf('data_031618_400000.csv'));
% csvwrite(file_name,saved_matrix4);
% file_name = fullfile('I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\datasize_analysis', sprintf('data_031618_validation40000.csv'));
% csvwrite(file_name,saved_matrix5);
% file_name = fullfile('I:\SW\PDE_data_storage\11_13_variables_all_range_001\simulated_data\datasize_analysis', sprintf('data_031618_test40000.csv'));
% csvwrite(file_name,saved_matrix6);
