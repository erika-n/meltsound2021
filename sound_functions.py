import numpy as np
from scipy.io import wavfile
import audio_filter





def getWav(input_file):

  rate, data = wavfile.read(input_file)

  # convert to mono
  if(len(data.shape) > 1):
    data = 0.5*np.sum(data, axis=1)

  # normalize data
  data = data/np.max(np.absolute(data))


  return data


def writeWav(output_wav, output_path):
  # save -1 to 1 normalized data as 16 bit integer wav
  rate = 44100

  output_wav /= np.max(np.absolute(output_wav))
  output_wav *= 0.4*2**16
  output_wav = output_wav.astype(np.int16)
  print('output wav data')
  print(output_wav)
  print('output wav data shape', output_wav.shape)
  print('writing wav data to', output_path)
  wavfile.write(output_path, rate, output_wav)


def toTimeDomain(data):
  

    complex_data = magPhaseToComplex(data)
    complex_data = complex_data.reshape((-1, complex_data.shape[-1]))
    print('complex data shape', complex_data.shape)
    
    real_data = np.fft.irfft(complex_data, complex_data.shape[1], axis=1)
    real_data = real_data.flatten()

    return real_data



def getFFT(data,  fftwidth, timewidth):

    channels = 2 # mag and phase
    fft_chunks = (int( data.shape[0]/fftwidth))
    sounddata = np.resize(data, (fft_chunks, fftwidth) )
    timechunks = int(sounddata.shape[0]/timewidth)
    sounddata = sounddata[:timechunks*timewidth]
    
    fftdata = np.fft.fft(sounddata, fftwidth, axis=1)  

    mag = np.absolute(fftdata)
    phase = np.angle(fftdata)

    output = np.ndarray((timechunks, channels, timewidth, fftwidth))
    for t in range(timechunks):
        output[t, 0] = mag[t*timewidth:(t+ 1)*timewidth]
        output[t, 1] = phase[t*timewidth:(t+ 1)*timewidth]
    return output





def magPhaseToComplex(data):
    mag = data[:, 0, :, :]
    phase = data[:, 1, :, :]

    data_complex = mag*np.exp(1.j*phase)
    return data_complex

def filterBank(data, fs=44100, order=2, n = 20, step=500):

    fs = 44100
    order = 3
    fb = []
    for i in range(1, 1 + n):
        lowcut = i*step
        highcut = (i + 1)*step
        filtered =  audio_filter.butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
        fb.append(filtered)
    return np.array(fb)
    
def unFilter(filtered):
    return np.sum(filtered, axis=0)


def testSoundFunctions():
    filename = '../sounds/songsinmyhead/b/01blame.wav'

    data = getWav(filename)
    fs = 44100
    data = data[fs*30:fs*35]
    print('data 0 shape', data.shape)
    print('data[:20]', data[:20])
    fftdata = getFFT(data, 100,100)
    print('fftdata shape', fftdata.shape)
    # print('FFT max, min, avg:')
    # print(np.max(fftdata))
    # print(np.min(fftdata))
    # print(np.average(fftdata))

    data = toTimeDomain(fftdata)
    print('data.shape', data.shape)
    print('data[:20]', data[:20])

    writeWav(data, 'outputs/test.wav')


if __name__ == "__main__":
    testSoundFunctions()
