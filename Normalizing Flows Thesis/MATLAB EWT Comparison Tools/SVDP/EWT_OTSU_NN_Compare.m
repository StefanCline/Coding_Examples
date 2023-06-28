% Utilizing the Neural Net from Norm Flows with Multi-Otsu Idea Here
clear all;
close all;
clc;

%% Import of all of the various signals and breaks
[x1_ts,x2_ts,tvals,x1_kbreaks,x2_kbreaks] = pickle_load_VDP_MONN();
%[x1_ts,x2_ts,tvals,x1_kbreaks,x2_kbreaks] = pickle_load_VDP_MONN_low_sig();
ts_sig_name = 'x2';

if ts_sig_name == 'x1'
    f = x1_ts'; % this is the main signal choice
    t = 0:1/length(f):1-1/length(f);
end
if ts_sig_name == 'x2'
    f = x2_ts'; % this is the main signal choice
    t = 0:1/length(f):1-1/length(f);
end

%% User setup

% Choose the signal you want to analyze, x1(t) or x2(t), x1_ts or x2_ts
% (i.e. the time series)
signal = f;
params.SamplingRate = -1; 

%choose wavelet (littlewood-paley,shannon,gabor1 (rays=plain Gaussians),
% gabor2 (rays=half Gaussian/half cst))
params.wavname = 'littlewood-paley';

% Choose the wanted global trend removal (none,plaw,poly,morpho,tophat,opening)
params.globtrend = 'none';
params.degree=6; % degree for the polynomial interpolation

% Choose the wanted regularization (none,gaussian,average,closing)
params.reg = 'none';
params.lengthFilter = 10;
params.sigmaFilter = 1.5;

% Choose the wanted detection method (locmax,locmaxmin,
% adaptive,adaptivereg,scalespace)
params.detect = 'scalespace';
params.typeDetect='otsu'; %for scalespace:otsu,halfnormal,empiricallaw,mean,kmeans
params.N = 4; % maximum number of bands
params.completion = 0; % choose if you want to force to have params.N modes
                       % in case the algorithm found less ones (0 or 1)
%params.InitBounds = [4 8 13 30];
params.InitBounds = [2 25];

% Perform the detection on the log spectrum instead the spectrum
params.log=0;
subresf=1;
InitBounds = params.InitBounds;

%% FOR THE UNALTERED EWT  
% We perform the empirical transform and its inverse
% compute the EWT (and get the corresponding filter bank and list of 
% boundaries, as well as the reconstruction error)
[ewt,mfb,boundaries]=EWT1D(f,params);
rec=iEWT1D(ewt,mfb,~isreal(f));
EWT_no_alteration_rec_error = norm(f-rec,Inf);
Show_EWT(ewt);

%% NOW FOR THE ALTERED EWT1D Function
if ts_sig_name == 'x1'
    all_k = size(x1_kbreaks);
    kbreaks_mat = x1_kbreaks;
end
if ts_sig_name == 'x2'
    all_k = size(x2_kbreaks);
    kbreaks_mat = x2_kbreaks;
end
EWT_altered_rec_error = 1e20; % initially set value to be execssively high, will be replaced, records best reconstruction value and index

for ii = 1:all_k(1) % outer loop specifies which kbreaks row we're on
    disp("On "+ii+" of "+ all_k(1))
    input_kbreaks = [];
    for jj = 1:(all_k(2)-1) % inner loop builds a kbreaks mat to input into the EWT1D_Modified function
        if kbreaks_mat(ii,jj) ~= 0
            input_kbreaks(jj) = kbreaks_mat(ii,jj);
        end
    end
    [ewt,mfb] = EWT1D_Modified(f,params,input_kbreaks');
    REC=iEWT1D(ewt,mfb,~isreal(f));
    EWT_altered_rec_error_new = norm(f-REC,Inf);
    if EWT_altered_rec_error_new < EWT_altered_rec_error
        ticker_val = kbreaks_mat(ii,end)-1;
        best_rec = REC;
        EWT_altered_rec_error = EWT_altered_rec_error_new;
        best_ewt = ewt;
    end
end
Show_EWT(best_ewt);

%% Plotting the Original EWT, and the Otsu NN, Reconstructions

% original EWT
figure('Name','Unaltered EWT Result');
subplot(2,1,1);
plot(f);
title('Original signal',fontsize=16);
set(gca,"FontSize",16);
subplot(2,1,2);
plot(rec);
title('Reconstructed signal',fontsize=16);
xlabel("Rec Error: "+EWT_no_alteration_rec_error)
set(gca,"FontSize",16);
% disp('Reconstruction error:');
% norm(f-rec,Inf)

% NN Norm Flow Otsu breaks
figure('Name','Otsu Norm Flow Result');
subplot(2,1,1);
plot(f);
title('Original signal',fontsize=16);
set(gca,"FontSize",16);
subplot(2,1,2);
plot(REC);
title('Reconstructed signal',fontsize=16);
xlabel("Rec Error: "+EWT_altered_rec_error)
set(gca,"FontSize",16);
% disp('Reconstruction error:');
% norm(f-REC,Inf)









%% Keep for maybe later
% 
% % Choose the results you want to display (Show=1, Not Show=0)
% Bound=1;   % Display the detected boundaries on the spectrum
% Comp=0;    % Display the EWT components
% Rec=1;     % Display the reconstructed signal
% TFplane=0; % Display the time-frequency plane (by using the Hilbert 
%            % transform). You can decrease the frequency resolution by
%            % changing the subresf variable below. (WORKS ONLY FOR REAL
%            % SIGNALS
% Demd=0;    % Display the Hilbert-Huang transform (WORKS ONLY FOR REAL 
%            % SIGNALS AND YOU NEED TO HAVE FLANDRIN'S EMD TOOLBOX)

% if Bound==1 %Show the boundaries on the spectrum
%     div=1;
%     if (strcmp(params.detect,'adaptive')||strcmp(params.detect,'adaptivereg'))
%         Show_EWT_Boundaries(f,boundaries,div,params.SamplingRate,InitBounds);
%     else
%         Show_EWT_Boundaries(f,boundaries,div,params.SamplingRate);
%     end
% end
% 
% if Comp==1 %Show the EWT components and the reconstructed signal
%         Show_EWT(ewt);
% end

% if Rec==1
%         %compute the reconstruction
%         rec=iEWT1D(ewt,mfb,~isreal(f));
%         if isreal(f)
%             figure;
%             subplot(2,1,1);plot(f);title('Original signal',fontsize=16);
%             set(gca,"FontSize",16);
%             subplot(2,1,2);plot(rec);title('Reconstructed signal',fontsize=16);
%             set(gca,"FontSize",16);
%             disp('Reconstruction error:');
%             norm(f-rec,Inf)
%         else
%             figure;
%             subplot(2,2,1);plot(real(f));title('Original signal - real part',fontsize=16);
%             subplot(2,2,2);plot(imag(f));title('Original signal - imaginary part',fontsize=16);
%             subplot(2,2,3);plot(real(rec));title('Reconstructed signal - real part',fontsize=16);
%             subplot(2,2,4);plot(imag(rec));title('Reconstructed signal - imaginary part',fontsize=16);
%             set(gca,"FontSize",16);
%             disp('Reconstruction error:');
%             norm(f-rec,Inf)            
%         end
% end

