import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import librosa
import torchaudio
import torch.nn as nn

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):

    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec #[batch_size,n_fft/2+1,frames]

def amp_pha_specturm(y, n_fft, hop_size, win_size):

    hann_window=torch.hann_window(win_size).to(y.device)

    stft_spec=torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,center=True) #[batch_size, n_fft//2+1, frames, 2]

    rea=stft_spec[:,:,:,0] #[batch_size, n_fft//2+1, frames]
    imag=stft_spec[:,:,:,1] #[batch_size, n_fft//2+1, frames]

    log_amplitude=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]
    phase=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, rea, imag

def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):
    training_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:
      
      src=os.path.join(input_training_wav_list,files)
      training_files.append(src)
    
    validation_files=[]
    filelist=os.listdir(input_validation_wav_list)
    for files in filelist:
      src=os.path.join(input_validation_wav_list,files)
      validation_files.append(src)

    return training_files, validation_files


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax,meloss, split=True, shuffle=True, n_cache_reuse=1,
                 device=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.meloss=meloss

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio = load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio) #[T]
        audio = audio.unsqueeze(0) #[1,T]

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_size] #[1,T]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                              center=True)
        meloss1 = mel_spectrogram(audio, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.meloss,
                              center=True)
        log_amplitude, phase, rea, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size) #[1,n_fft/2+1,frames]


        return (mel.squeeze(), log_amplitude.squeeze(), phase.squeeze(), rea.squeeze(), imag.squeeze(), audio.squeeze(0),meloss1.squeeze())

    def __len__(self):
        return len(self.audio_files)
