# -*- coding: utf-8 -*-

from __future__ import print_function, division

from Hyperparameters import Hyperparams as Hp
import numpy as np
import tensorflow as tf
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import os,sys,io

__author__ = "Tong Wang"


''' 
adopt from others research, https://www.github.com/kyubyong/expressive_tacotron/utils.py
'''

def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=Hp.sample_rate)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - Hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=Hp.num_fft,
                          hop_length=Hp.hop_length,
                          win_length=Hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(Hp.sample_rate, Hp.num_fft, Hp.num_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - Hp.ref_db + Hp.max_db) / Hp.max_db, 1e-8, 1)
    mag = np.clip((mag - Hp.ref_db + Hp.max_db) / Hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogrom2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * Hp.max_db) - Hp.max_db + Hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -Hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(Hp.grilimn_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, Hp.num_fft, Hp.hop_length, win_length=Hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, Hp.hop_length, win_length=Hp.win_length, window="hann")


def plot_alignment(alignment, gs, mode, path = None):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    mode: "save_fig" or "with_return". "save_fig":save fig locally, "with_return":return plot for tensorboard
    """

    plt.imshow(alignment, cmap='hot', interpolation='nearest')
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    plt.title('{} Steps'.format(gs))

    if mode == "save_fig":
        if path is not None:
            plt.savefig('{}/alignment_{}k.png'.format(path, gs//Hp.save_model_step), format='jpg')
        else:
            print ("Warning! You need specify the saved path! The temporal path is {}".format(Hp.logdir))
            plt.savefig('{}/alignment_{}k.png'.format(Hp.logdir, gs//Hp.save_model_step), format='jpg')

    elif mode == "with_return":
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot = tf.image.decode_png(buf.getvalue(), channels=4)
        plot = tf.expand_dims(plot,0)
        return plot

    else:
        print ("Error Mode! Exit!")
        sys.exit(0)


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = Hp.reduction_factor - (t % Hp.reduction_factor) if t % Hp.reduction_factor != 0 else 0 # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, Hp.num_mels*Hp.reduction_factor)), mag
