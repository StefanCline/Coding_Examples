function [ewt,mfb]=EWT1D_Modified(f,params,kbounds)

% Here, boundaries are imported from the Multi-Otsu NN Approach
%  as such, boundaries are not calculated

% testing out the output from the norm flow situation here
% this was for fsig2, where the output was from the NN
%    results were good!
% boundaries = [0.02235943; 0.06237243; 0.10384114];
% this was for ECG, out put from the Multi-Otus Norm Flow NN
% boundaries = [0.45340616; 0.07097941; 1.18459965; 0.83451871; 0.7893087;  1.460094; 1.03008774; 2.27397038; 2.53616895; 3.00593969; 1.50160673; 0.57332442; 0.17914847; 2.75229442; 1.55980544; 2.11541548; 2.3415545;  0.83995257; 2.69866077; 1.55311697; 2.78620032; 2.74840419; 2.4133053; 0.45359548; 1.06531227; 0.90271438; 1.75018342; 1.32748553; 1.08675937; 2.80111874; 0.84116928; 1.74437401; 0.54826974; 0.59846512; 0.93400286; 2.03225008; 1.18588059; 0.50230717; 1.6809052;  3.14113614; 1.43372607; 1.20818402; 1.82726773; 1.48517724; 0.11069132; 0.72297747];

ff=fft(f);
boundaries = kbounds;

% We build the corresponding filter bank
switch lower(params.wavname)
    case 'littlewood-paley'
        mfb=EWT_LP_FilterBank(boundaries,length(ff),~isreal(f));
    case 'shannon'
        mfb=EWT_Shannon_FilterBank(boundaries,length(ff),~isreal(f));
        
    case 'gabor1'
        mfb=EWT_Gabor1_FilterBank(boundaries,length(ff),~isreal(f));

    case 'gabor2'
        mfb=EWT_Gabor2_FilterBank(boundaries,length(ff),~isreal(f));

    case 'meyer'
        mfb=EWT_Meyer_FilterBank(boundaries,length(ff),~isreal(f));
end
% We filter the signal to extract each subband
ewt=cell(length(mfb),1);

switch lower(params.wavname)
    case 'littlewood-paley'
        if isreal(f)
            for k=1:length(mfb)
                ewt{k}=real(ifft(conj(mfb{k}).*ff));
            end
        else
            for k=1:length(mfb)
                ewt{k}=ifft(conj(mfb{k}).*ff);
            end
        end
    case {'shannon','gabor1','gabor2','meyer'}
        for k=1:length(mfb)
            ewt{k}=ifft(conj(mfb{k}).*ff);
        end
end