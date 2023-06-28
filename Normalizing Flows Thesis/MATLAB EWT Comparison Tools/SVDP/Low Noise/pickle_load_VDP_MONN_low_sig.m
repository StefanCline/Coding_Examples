%testing out loading a pkl file

function [x1_ts,x2_ts,tvals,x1_kbreaks,x2_kbreaks] = pickle_load_VDP_MONN_low_sig()

% % importing for general file
% filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder\Im_VDP_Time_Series_Data_Pickle.pkl';
% fid=py.open(filename,'rb');
% data=py.pickle.load(fid);
% %And numpy arrays to doubles
% data=double(data);

% importing for x1 after fft
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_x1_fft.pkl';
fid=py.open(filename,'rb');
x1_fft=py.pickle.load(fid);
%And numpy arrays to doubles
x1_fft=double(x1_fft);

% importing for x2 after fft
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_x2_fft.pkl';
fid=py.open(filename,'rb');
x2_fft=py.pickle.load(fid);
%And numpy arrays to doubles
x2_fft=double(x2_fft);

% importing for omega
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_omega.pkl';
fid=py.open(filename,'rb');
omega=py.pickle.load(fid);
%And numpy arrays to doubles
omega=double(omega);

% importing for tvals
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_tvals.pkl';
fid=py.open(filename,'rb');
tvals=py.pickle.load(fid);
%And numpy arrays to doubles
tvals=double(tvals);

% importing for x1 time series data
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_x1_ts.pkl';
fid=py.open(filename,'rb');
x1_ts=py.pickle.load(fid);
%And numpy arrays to doubles
x1_ts=double(x1_ts);

% importing for x2 time series data
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\Im_VDP_Time_Series_Data_Pickle_x2_ts.pkl';
fid=py.open(filename,'rb');
x2_ts=py.pickle.load(fid);
%And numpy arrays to doubles
x2_ts=double(x2_ts);

% importing for x1 kbreaks
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\x1_low_sig\Im_VDP_VDP_kbreaks_pickle_Pickle_x1_low_sig.pkl';
fid=py.open(filename,'rb');
x1_kbreaks=py.pickle.load(fid);
%And numpy arrays to doubles
x1_kbreaks=double(x1_kbreaks);

% importing for x2 kbreaks
filename = 'F:\Research_Thesis\NormFlows\Image Runs\VDP\x1_x2_data_folder_low_sig\x2_low_sig\Im_VDP_VDP_kbreaks_pickle_Pickle_x2_low_sig.pkl';
fid=py.open(filename,'rb');
x2_kbreaks=py.pickle.load(fid);
%And numpy arrays to doubles
x2_kbreaks=double(x2_kbreaks);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing the loads were done correctly and plotting 
% figure
% plot(x1_ts,x2_ts)
% title('SVDP Oscillator: x1 v x2')
% xlabel('x1')
% ylabel('x2')
% 
% figure
% plot(tvals,x1_ts)
% title('Time Series x1')
% xlabel('t')
% ylabel('x1')
% 
% figure
% plot(tvals,x2_ts)
% title('Time Series x2')
% xlabel('t')
% ylabel('x2')
% 
% figure
% loglog(omega,x1_fft)
% title('FFT of x1')
% xlabel('$\omega$',Interpreter='latex')
% ylabel('x1')
% 
% figure
% loglog(omega,x2_fft)
% title('FFT of x2')
% xlabel('$\omega$',Interpreter='latex')
% ylabel('x2')