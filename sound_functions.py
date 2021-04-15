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

    real_data = np.fft.irfft(complex_data, complex_data.shape[1], axis=1)
    real_data = np.reshape(real_data, (real_data.shape[0]*real_data.shape[1], 2))

    return real_data



def getFFT(data, seconds=1,  max_out = 0, frames_per_second = 30, rate=44100):
    fftwidth = int(rate/frames_per_second)


    sounddata = np.resize(data, (int( data.shape[0]/fftwidth), fftwidth, 2) )
  
    fftdata = np.fft.fft(sounddata, fftwidth, axis=1)  

    mag = np.absolute(fftdata)
    phase = np.angle(fftdata)


    # 2 channel output
    output = np.ndarray((mag.shape[0], mag.shape[1], 2, 2))
    output[:,:, :, 0] = mag
    output[:, :,:, 1] = phase
    return output





def magPhaseToComplex(data):
    mag = data[:, :, :, 0]
    phase = data[:, :, :, 1]

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


def testSoundFunctions():
    filename = '../sounds/songsinmyhead/b/01blame.wav'
    outFilename = 'output/testSoundFunctions.wav'
    data = getWav(filename)
    fs = 44100
    data = data[fs*30:fs*35]

    # fftdata = getFFT(data, seconds=0.5, frames_per_second=60, max_out=0)
    # # print('FFT max, min, avg:')
    # # print(np.max(fftdata))
    # # print(np.min(fftdata))
    # # print(np.average(fftdata))

    # data = toTimeDomain(np.array(fftdata))

    # lowcut = 440
    # highcut = 450
    # fs = 44100
    # order = 3
    # data = audio_filter.butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
    # print('data max', np.max(data))
    # writeWav(data, outFilename)
    # print("saved", outFilename)
    filtered = filterBank(data, n=45, step=200)
    print('filtered max', np.max(filtered))
    total = np.sum(filtered, axis=0)
    print('total shape', total.shape)
    print('total max', np.max(total))
    writeWav(filtered[4], 'output/filtered_4.wav')
    writeWav(total, 'output/total.wav')


if __name__ == "__main__":
    testSoundFunctions()
