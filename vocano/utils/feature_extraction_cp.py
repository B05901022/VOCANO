# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:14:39 2020

@author: Austin Hsu
"""

import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
import numpy as np
import cupy as cp
import torch
import os
from ..model.Patch_CNN import KitModel
from tqdm import tqdm

def read_file(filename: str) -> (np.array, int):
    """Read files"""
    # --- Load File ---
    sample_rate, audio = scipy.io.wavfile.read(filename)
    audio = audio.astype(np.float32)/32767.0
    if len(audio.shape)==2:
        audio = audio.mean(axis=-1)
        
    #  --- Resample ---
    if sample_rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sample_rate)
    return audio

# -----------------------------------------------------------------------------
#                             Feature Extraction
# -----------------------------------------------------------------------------

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/Hop)*Hop, Hop)
    N = int(fs/fr)
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, round(N/2), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1                                             
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])        
    
    tfr = cp.asarray(tfr)
    tfr = cp.fft.fft(tfr, n=N, axis=0)
    tfr = abs(tfr)
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = cp.power(X, g)
    else:
        X = cp.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    freq_band_transformation = cp.asarray(freq_band_transformation)
    tfrL = cp.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/(q + 1e-8)
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    
    freq_band_transformation = cp.asarray(freq_band_transformation)
    tfrL = cp.dot(freq_band_transformation, ceps)
    return tfrL, central_freq

def gen_spectral_flux(S, invert=False, norm=True):
    flux = cp.diff(S)
    first_col = cp.zeros((S.shape[0],1))
    flux = cp.hstack((first_col, flux))
    
    if invert:
        flux = flux * (-1.0)

    flux = cp.where(flux < 0, 0.0, flux)

    if norm:
        flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

    return flux

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    tfr, f, t, N = STFT(x, fr, fs, Hop, h)
    tfr = cp.power(tfr, g[0])
    tfr0 = tfr # original STFT
    #ceps = np.zeros(tfr.shape)

    for gc in range(1, NumofLayer):
        if np.remainder(gc, 2) == 1:
            tc_idx = round(fs*tc) # 16
            ceps = cp.real(cp.fft.fft(tfr, axis=0))/np.sqrt(N)
            ceps = nonlinear_func(ceps, g[gc], tc_idx)
        else:
            fc_idx = round(fc/fr) # 40
            tfr = cp.real(cp.fft.fft(ceps, axis=0))/np.sqrt(N)
            tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    q = np.arange(HighQuefIdx)/float(fs)
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies

def cfp_feature_extraction(audio: np.array) -> np.array:
    # --- Args ---
    fs = 16000.0 # sampling frequency
    Hop = 320 # hop size (in sample)
    h3 = scipy.signal.blackmanharris(743) # window size - 2048
    h2 = scipy.signal.blackmanharris(372) # window size - 1024
    h1 = scipy.signal.blackmanharris(186) # window size - 512
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    num_per_oct = 48 # Number of bins per octave
        
    # --- CFP Filterbank ---
    tfrL01, tfrLF1, tfrLQ1, f1, q1, t1, CenFreq1 = CFP_filterbank(audio, fr, fs, Hop, h1, fc, tc, g, num_per_oct)
    tfrL02, tfrLF2, tfrLQ2, f2, q2, t2, CenFreq2 = CFP_filterbank(audio, fr, fs, Hop, h2, fc, tc, g, num_per_oct)
    tfrL03, tfrLF3, tfrLQ3, f3, q3, t3, CenFreq3 = CFP_filterbank(audio, fr, fs, Hop, h3, fc, tc, g, num_per_oct)
    
    return tfrL01, tfrLF1, tfrLQ1, tfrL02, tfrLF2, tfrLQ2, tfrL03, tfrLF3, tfrLQ3

def full_feature_extraction(
        tfrL01, tfrLF1, tfrLQ1,
        tfrL02, tfrLF2, tfrLQ2,
        tfrL03, tfrLF3, tfrLQ3
        ):
    Z1 = tfrLF1 * tfrLQ1
    ZN1 = (Z1 - np.mean(Z1)) / np.std(Z1)
    Z2 = tfrLF2 * tfrLQ2
    ZN2 = (Z2 - np.mean(Z2)) / np.std(Z2)
    Z3 = tfrLF3 * tfrLQ3
    ZN3 = (Z3 - np.mean(Z3)) / np.std(Z3)
    SN1 = gen_spectral_flux(tfrL01, invert=False, norm=True)
    SN2 = gen_spectral_flux(tfrL02, invert=False, norm=True)
    SN3 = gen_spectral_flux(tfrL03, invert=False, norm=True)
    SIN1 = gen_spectral_flux(tfrL01, invert=True, norm=True)
    SIN2 = gen_spectral_flux(tfrL02, invert=True, norm=True)
    SIN3 = gen_spectral_flux(tfrL03, invert=True, norm=True)
    SN = cp.concatenate((SN1, SN2, SN3), axis=0)
    SIN = cp.concatenate((SIN1, SIN2, SIN3), axis=0)
    ZN = cp.concatenate((ZN1, ZN2, ZN3), axis=0)
    SN_SIN_ZN = cp.concatenate((SN, SIN, ZN), axis=0)
    return SN_SIN_ZN

# -----------------------------------------------------------------------------
#                             Melody Extraction
# -----------------------------------------------------------------------------

def melody_feature_extraction(x):
    # --- Args ---
    fs = 16000.0 # sampling frequency
    x = x.astype('float32')
    Hop = 320 # hop size (in sample)
    h = scipy.signal.blackmanharris(2049) # window size
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48 # Number of bins per octave
    
    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
    Z = tfrLF * tfrLQ
    return Z, t, CenFreq

class temp_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).unsqueeze(1).float()
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

def patch_extraction(Z, patch_size, th):
    # Z is the input spectrogram or any kind of time-frequency representation
    M, N = Z.shape    
    half_ps = int(np.floor(float(patch_size)/2)) #12

    Z = np.pad(Z, ((0, half_ps), (half_ps, half_ps)))
    
    M, N = Z.shape
    
    data = []
    mapping = []
    counter = 0
    for t_idx in range(half_ps, N-half_ps):
        LOCS = findpeaks(Z[:,t_idx], th)
        for mm in range(0, len(LOCS)):
            if LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter<300000:# and PKS[mm]> 0.5*max(Z[:,t_idx]):
                patch = Z[np.ix_(range(LOCS[mm]-half_ps, LOCS[mm]+half_ps+1), range(t_idx-half_ps, t_idx+half_ps+1))]
                data.append(patch)
                mapping.append((LOCS[mm], t_idx))
                counter = counter + 1
            elif LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter>=300000:
                print('Out of the biggest size. Please shorten the input audio.')
                
    data = np.array(data[:-1])
    mapping = np.array(mapping[:-1])
    Z = Z[:M-half_ps,:]
    return data, mapping, half_ps, N, Z

def patch_prediction(modelname, data, patch_size,
                     batch_size, num_workers, pin_memory, device):
    modelname = os.path.join('./checkpoint/',modelname+'.npy')
    model = KitModel(weight_file=modelname).to(device)
    dataset = temp_dataset(data=data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    pred = []
    model = model.eval()
    for b_id, b_x in enumerate(tqdm(dataloader)):
        b_x = b_x.to(device)
        pred.append(model(b_x).detach().cpu().numpy())
    return np.concatenate(pred)

def contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method):
    PredContour = np.zeros(N)

    pred = pred[:,1]
    pred_idx = np.where(pred>0.5)
    MM = mapping[pred_idx[0],:]
    pred_prob = pred[pred_idx[0]]
    MM = np.append(MM, np.reshape(pred_prob, [len(pred_prob),1]), axis=1)
    MM = MM[MM[:,1].argsort()]    
    
    for t_idx in range(half_ps, N-half_ps):
        Candidate = MM[np.where(MM[:,1]==t_idx)[0],:]
        if Candidate.shape[0] >= 2:
            if max_method == 'posterior':
                fi = np.where(Candidate[:,2]==np.max(Candidate[:,2]))
                fi = fi[0]
            elif max_method == 'prior':
                fi = Z[Candidate[:,0].astype('int'),t_idx].argmax(axis=0)
            fi = fi.astype('int')
            PredContour[Candidate[fi,1].astype('int')] = Candidate[fi,0] 
        elif Candidate.shape[0] == 1:
            PredContour[Candidate[0,1].astype('int')] = Candidate[0,0] 
    
    # clip the padding of time
    PredContour = PredContour[range(half_ps, N-half_ps)]
    
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    
    Z = Z[:, range(half_ps, N-half_ps)]
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def findpeaks(x, th):
    # x is an input column vector
    M = x.shape[0]
    pre = x[1:M - 1] - x[0:M - 2]
    pre = np.sign(pre)
    
    post = x[1:M - 1] - x[2:]
    post = np.sign(post)

    mask = pre * post
    ext_mask = np.pad(mask,1)
    
    locs = np.where(ext_mask==1)
    locs = locs[0]
    return locs

def melody_extraction(audio, batch_size, num_workers, pin_memory, device):
    
    # --- Args ---
    patch_size = 25
    th = 0.5
    modelname = 'model3_patch25'
    max_method = 'posterior'
        
    # --- Feature Extraction ---
    Z, t, CenFreq = melody_feature_extraction(audio)
    Z = cp.asnumpy(Z)
    
    # --- Patch Extraction ---
    data, mapping, half_ps, N, Z = patch_extraction(Z, patch_size, th)
    
    # --- Pitch Prediction ---
    pred = patch_prediction(modelname, data, patch_size, batch_size, num_workers, pin_memory, device)
    result = contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method)

    return result

# -----------------------------------------------------------------------------
#                             Complete Flow
# -----------------------------------------------------------------------------

def test_flow(filename, use_ground_truth=False, batch_size=64, num_workers=0, pin_memory=False, device=torch.device("cpu")):
    audio = read_file(filename)
    feat = full_feature_extraction(*cfp_feature_extraction(audio))
    pitch = None
    if not use_ground_truth:
        print(f"Melody extraction start...")
        pitch = melody_extraction(audio, batch_size, num_workers, pin_memory, device)[:,1]
        print(f"Melody successfully extracted.")
    return cp.asnumpy(feat), pitch