[data, Fs] = audioread('test.mp3');
% Then data is the audio sampling, Fs is sampling frequency.

% Length of each FFT period in seconds.
duration = 1/10;
% Length of each FFT sample
L = Fs*duration;
timeLabels = Fs*(0:(L/2))/L;

numExamples = floor(size(data,1)/L) - 1;

output = zeros(floor(L/2)+1,numExamples);

for i = 1:numExamples
    window = data(L*i:L*i+L-1);
    clearvars fft;
    fftoutput = fft(window);
    
    % some weird MATLAB example code relating to getting the FFT to a
    % real-valued function
    P2 = abs(fftoutput/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    if mod(i,100) == 0
        plot(timeLabels,P1);
        hold on;
    end
    
    output(:,i) = P1;
end

title('Fourier Transform (Amplitude Spectrum) of Recording Snapshot')
xlabel('f (Hz)')
ylabel('|P1(f)|')



csvwrite('test.csv',output);