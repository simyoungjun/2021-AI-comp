function [score_01,f] = scoring(sel_feature,pts,pa,bw)
%UNTITLED 이 함수의 요약 설명 위치
%   자세한 설명 위치

f = mvksdensity(sel_feature,pts,'Bandwidth',pa*bw);
score=abs(log10(f));
score_01=score;
 indx_noninf=find(score(:,1)~=inf);
 indx_inf=find(score(:,1)==inf);
 
score_01(indx_noninf)=(score(indx_noninf)-min(score(indx_noninf)))/(max(score(indx_noninf))-min(score(indx_noninf)));
score_01(indx_inf)=1;

end

