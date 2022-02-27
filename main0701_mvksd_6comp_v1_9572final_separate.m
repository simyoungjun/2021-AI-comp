clear
close all
 load('train_A.mat')
 TA=A;
 train_length=length(listing);
   load('data_mat.mat')
   test_length=length(listing);
   A=cat(3,TA,A);
%   
%   power_std=3;
%  power_mean=[2,30];
%  partial_power_ratio=.3;
 
 
%load('test17.mat')

 dim_A=size(A);
 
 k=dim_A(3);
% [list_outlier,Power_xyz]=outlier_detector (A, listing,power_std,power_mean , partial_power_ratio);
 A1(:,:)= abs(fft(A(:,1,:)));
A2(:,:)= abs(fft(A(:,2,:)));
A3(:,:)= abs(fft(A(:,3,:)));

% feature 
tone_mag=[abs(A1(61,:));abs(A2(61,:));abs(A3(61,:))];
A1(61,:)=0;
A1(1941,:)=0;
A2(61,:)=0;
A2(1941,:)=0;
A3(61,:)=0;
A3(1941,:)=0;
A1_mag=sqrt(sum(A1(1:1000,:).^2))./tone_mag(1,:);
A2_mag=sqrt(sum(A2(1:1000,:).^2))./tone_mag(2,:);
A3_mag=sqrt(sum(A3(1:1000,:).^2))./tone_mag(3,:);

A_total=[A1(1:1000,:);A2(1:1000,:);A3(1:1000,:)];
A_total_pw=mean(A_total.^2)';
A_total_std=std(A_total.^2)';

%%%%%%%%%%%%%
feature_vector=[ tone_mag' A_total_pw  A_total_std];
 feature_vector_new=[ tone_mag'  A1_mag' A2_mag' A3_mag'];
 
%%%%%%%%%%%%%%%%%%%%%%%%


sel_feature=feature_vector_new(1:train_length,:);
pts=feature_vector_new(train_length+1:end,:);
std_feature=std(sel_feature);

d=size(feature_vector_new,2);

c=(4/(d+2)/train_length)^(1/(d+4));
bw=std_feature*c;
%[idx, C,sumd,D,midx,info]= kmedoids(sel_feature,2,'Distance', 'sqeuclidean');
f_name='test0701_mvksd_total_6ft_v2_02_separate.csv';
pa=0.02 %0.0423
[score_01_1,f] = scoring(sel_feature,pts,pa,bw);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pa=1 %0.0423
% [score_01_2,f(:,2)] = scoring(sel_feature,pts,pa,bw);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sel_feature=feature_vector ; %(1:train_length,:);
% pts=feature_vector(train_length+1:end,:);
% std_feature=std(sel_feature);

%d=size(feature_vector ,2);

% c=(4/(d+2)/train_length)^(1/(d+4));
% bw=std_feature*c;
% 
% pa_ref=0.007; 
% 
% [score_01_2,f(:,3)] = scoring(sel_feature ,pts,pa_ref,bw);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% diff_scoring=score_01_1-score_01_2;
% figure,plot(diff_scoring)
score_01=score_01_1;

% figure, plot([sort(score_01_1) sort(score_01_2)] )
% norm(diff_scoring)


idx_zeroone_test=score_01;
 filename = readtable('sample_submission0625_v1.csv');
 fn = filename.rand_filename;
fn = cell2mat(fn);
fn = string(fn);
score_sort_list=zeros(length(fn),1);
for kk=1:length(fn)
    score_pos=find(fn==listing(kk).name);
    score_sort_list(score_pos)=idx_zeroone_test(kk);
end
filename.anomaly_score=score_sort_list;

writetable(filename, f_name, 'Delimiter',',','QuoteStrings',true)
 figure, plot(sort(idx_zeroone_test))
return
