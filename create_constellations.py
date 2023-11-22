# %%
import numpy as np
#import matplotlib.pyplot as plt
#from scipy import fft, signal
#from scipy.io.wavfile import read
#import wave
#from scipy.stats import zscore

# %%



def stft_numpy(x, fs, nperseg=256, noverlap=None):
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * fs)
    window_length_samples += window_length_samples % 2

    if noverlap is None:
        noverlap = nperseg // 2  # Default overlap to 50%

    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    hop_length = nperseg - noverlap
    if hop_length <= 0:
        raise ValueError("Hop length must be positive.")

    n_windows = (len(x) - nperseg) // hop_length + 1

    # Apply Hanning window function
    window = np.hanning(nperseg)

    # Print information for debugging
    print("n_windows:", n_windows)
    print("window_length_samples:", window_length_samples)

    # Check if x has valid length
    if len(x) < nperseg:
        raise ValueError("Input signal length is less than nperseg.")

    # Initialize stft_matrix
    stft_matrix = np.zeros((nperseg // 2 + 1, n_windows), dtype=np.complex128)

    for i in range(n_windows):
        start = i * hop_length
        end = start + nperseg
        segment = x[start:end]

        # Convert segment to float64 before applying the window
        segment = segment.astype(np.float64)
        segment *= window

        # Print information for debugging
        print("Processing window", i, " - Start:", start, "End:", end)
        print("Segment shape:", segment.shape)

        # Calculate FFT and assign to stft_matrix
        stft_matrix[:, i] = np.fft.fft(segment)[:nperseg // 2 + 1]

    # Select only positive frequencies
    frequencies = np.fft.fftfreq(window_length_samples, d=1.0 / fs)[:window_length_samples // 2 + 1]
    positive_freq_indices = frequencies >= 0
    frequencies = frequencies[positive_freq_indices]
    stft_matrix = stft_matrix[positive_freq_indices, :]
    stft_matrix /= np.max(np.abs(stft_matrix))

    # Plotting
    time_frames = np.arange(n_windows) * hop_length / fs
    freq_bins = np.sort(frequencies)
    '''
    plt.pcolormesh(time_frames, freq_bins, np.abs(stft_matrix), shading='auto')
    plt.title('Custom NumPy STFT')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar()
    plt.show()
    '''
    
    return stft_matrix, frequencies, time_frames









def create_constellation(audio, Fs):
    # Parameters
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 15

    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples

    song_input = np.pad(audio, (0, amount_to_pad))
    
    print("window_length_samples")
    print(window_length_samples)
 


   
    # Use the numpy version of stft

    #stft_matrix, freq_bins, time_frames = stft_numpy(audio)
    # Normalize to the range [-1, 1]
    #song_input = song_input / np.max(np.abs(song_input), axis=0)
    #stft_matrix, freq_bins, time_frames = stft_numpy(song_input,Fs)
    #stft_matrix, freq_bins, time_frames = stft_numpy(song_input,Fs)
    # Example usage

    window_length_seconds = 0.5
    overlap = 0.5
    
    #Fs = 22050
    nperseg = 1024
    noverlap = 512    
    
    

    # Assuming you have some audio signal in the variable 'audio'
    #stft_matrix, freq_bins, time_frames = stft_numpy(audio, Fs, int(window_length_seconds * Fs), overlap)
    #stft_matrix, freq_bins, time_frames = stft_numpy(song_input, Fs, int(window_length_seconds * Fs), overlap)
    
    #stft_matrix, freq_bins, time_frames = stft_numpy(audio, Fs, nperseg, noverlap)
    
    stft_matrix, freq_bins, time_frames = stft_numpy(song_input, Fs, nperseg=window_length_samples, noverlap=window_length_samples // 2)
    #new_noverlap = round(window_length_samples // 1.91)
    #stft_matrix, freq_bins, time_frames = stft_numpy(song_input, Fs, nperseg=window_length_samples, noverlap=new_noverlap)
    
    
    #print("NumPy - Frequency bins:", freq_bins)
    #print("NumPy - Time frames:", time_frames)
    

    
    
    '''
    print("song_input numpy")
    print(song_input)
    print('frequencies numpy')
    print(freq_bins)
    print('times numpy')
    print(time_frames)
    print('stft numpy')
    print(stft_matrix)
    '''


    

    constellation_map = []

    for time_idx, window in enumerate(stft_matrix.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
    
        # Use NumPy for z-score normalization
        normalized_spectrum = (spectrum - np.mean(spectrum)) / np.std(spectrum)

        # Find peaks - these correspond to interesting features
        positive_peaks = np.where(np.diff(np.sign(normalized_spectrum)) == -2)[0] + 1  # Find peaks
        sorted_peaks = positive_peaks[np.argsort(normalized_spectrum[positive_peaks])[-num_peaks:]]  # Select the largest peaks
        for peak in sorted_peaks:
            frequency = freq_bins[peak]
            constellation_map.append([time_idx, frequency])
        
        # Print the size of constellation_map and its content
        print("Constellation map size:", len(constellation_map))
        #print("Constellation map content:", constellation_map)
        
        # Ensure the constellation_map is not empty before returning
    '''    
    if len(constellation_map) > 0:
        return np.array(constellation_map)
    else:
        print("Constellation map is empty.")
        return None    
    return np.array(constellation_map)
    '''
    return np.array(constellation_map)



