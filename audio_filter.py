from scipy.signal import butter, lfilter
from scipy.io import wavfile


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def getWav(input_file):

  rate, data = wavfile.read(input_file)

  # convert to mono
  if(len(data.shape) > 1):
    data = 0.5*np.sum(data, axis=1)

  # normalize data
  print('data orig max min', np.max(data), np.min(data))
  data = data/np.max(np.absolute(data))
  print('data max min', np.max(data), np.min(data))


  return data

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 44100.0
    lowcut = 500.0
    highcut = 600.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    #T = 1.0
    #nsamples = int(T * fs)
    x = getWav('../sounds/songsinmyhead/b/01blame.wav')
    nsamples = x.shape[0]
    T = int(nsamples/fs)
    
    #t = np.linspace(0, T, nsamples, endpoint=False)
    t = np.arange(nsamples)
    # a = 0.02
    # f0 = 600.0
    # x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    # x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    # x += a * np.cos(2 * np.pi * f0 * t + .11)
    # x += 0.03 * np.cos(2 * np.pi * 2000 * t)

 
    
    print('x max min', np.max(x), np.min(x))
    print('t max min', np.max(t), np.min(t))
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    print('y max min', np.max(y), np.min(y))
    plt.plot(t, y, label='Filtered signal')
    # plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    # plt.grid(True)
    # plt.axis('tight')
    # plt.legend(loc='upper left')

    plt.show()