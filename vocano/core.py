# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:14:04 2020

@author: Austin Hsu
"""

import os
import torch
import numpy as np
import importlib
import pretty_midi
import platform
import scipy.io.wavfile as wavfile
from apex import amp
from pathlib import Path
from tqdm import tqdm

from google_drive_downloader import GoogleDriveDownloader as gdd

from .model.PyramidNet_ShakeDrop import PyramidNet_ShakeDrop
from .utils.evaluate_tools import Smooth_sdt6_modified, Naive_pitch
from .utils.dataset import EvalDataset
from .utils.est2midi import Est2MIDI

def package_check(package_name: str):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

CUPY_EXIST = package_check("cupy")

from .utils.feature_extraction import test_flow as feature_extraction_np
if CUPY_EXIST:
    import cupy as cp
    from .utils.feature_extraction_cp import test_flow as feature_extraction_cp

OS_PLATFORM = platform.system()

class SingingVoiceTranscription:
    
    FILE_ID = {'PyramidNet_ShakeDrop': '1m9YT7207CXQv1KdU0ivkRQrwvPnOuR3W',
               'Patch_CNN': '1tq_LcZwWQYV7wM6dBAeDZeQn39UNkPdl'}
    DOWNLOAD_PATH = {'PyramidNet_ShakeDrop': './checkpoint/model.pt',
                     'Patch_CNN': './checkpoint/model3_patch25.npy'}
    
    def __init__(self, args):
        """
        args.keys: 
            # --- directory/filename ---
            (n)name
            (fd)feat_dir
            (pd)pitch_dir
            (md)midi_dir
            (od)output_wav_dir
            (wd)wavfile_dir
            (gd)pitch_gt_dir
            (ckpt)checkpoint_file
            # --- system ---
            (s)save_extracted
            (use_pre)use_pre_extracted
            (use_gt)use_groundtruth
            (d)device
            (use_cp)use_cp
            (bz)batch_size
            (nw)num_workers
            (pn)pin_memory
            (al)amp_level
        """        
        self.args = args

        # --- device ---
        self._select_device()
    
    def _select_device(self):
        # cpu/selected gpu/auto-select gpu
        if self.args.device == "cpu":
            self.device = torch.device("cpu")
        elif self.args.device == "auto":
            if OS_PLATFORM == "Linux":
                os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
                for idx, memory in enumerate(memory_available):
                    print(f"cuda:{idx} available memory: {memory}")
                self.device = torch.device(f"cuda:{np.argmax(memory_available)}")
                print(f"Selected cuda:{np.argmax(memory_available)} as device")
                torch.cuda.set_device(int(np.argmax(memory_available)))
            else:
                raise OSError(f"{OS_PLATFORM} does not support auto method.")
        else:
            self.device = torch.device(f"cuda:{self.args.device}")
            torch.cuda.set_device(int(self.args.device))
    
    def _free_gpu(self):
        # free gpu usage
        if CUPY_EXIST:
            cp._default_memory_pool.free_all_blocks()
        torch.cuda.empty_cache()
        
    def _download_from_googledrive(self, file_id, dest_path):
        gdd.download_file_from_google_drive(file_id, dest_path)
    
    def download_ckpt(self):
        for file in self.FILE_ID:
            self._download_from_googledrive(self.FILE_ID[file], self.DOWNLOAD_PATH[file])
    
    def transcription(self):
        
        # --- download model ---
        if not Path("./checkpoint/model.pt").is_file():
            print(f"PyramidNet model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['PyramidNet_ShakeDrop'], self.DOWNLOAD_PATH['PyramidNet_ShakeDrop'])
            print(f"PyramidNet model checkpoint downloaded.")
        if not Path("./checkpoint/model3_patch25.npy").is_file():
            print(f"Patch-CNN model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['Patch_CNN'], self.DOWNLOAD_PATH['Patch_CNN'])
            print(f"Patch-CNN model checkpoint downloaded.")
            
        # --- feature/melody extraction ---
        if self.args.use_pre_extracted:
            feat_name = self.args.feat_dir / f"{self.args.name}_feat.npy"
            pitch_name = self.args.pitch_dir / f"{self.args.name}_pitch.npy"
            if feat_name.is_file() and pitch_name.is_file():
                self.feature = np.load(feat_name)
                self.pitch = np.load(pitch_name)
                print(f"Using pre-extracted feat {feat_name} and melody {pitch_name}")
            else:
                raise IOError(f"Given pre-extracted feature {str(feat_name)} and pitch contour {str(pitch_name)} does not exist.")
        else:
            self.data_preprocessing()

        # --- load model ---
        self.load_model()
        
        # --- vocal transcription ---
        self.voice_transcription()
        
        # --- gen midi/wav ---
        self.gen_midi()
        self.gen_wav()
        
        # --- save midi/wav ---
        self.save_midi(self.midi, self.args.midi_dir)
        self.save_wav(self.synth_midi, self.args.output_wav_dir)
        
    def data_preprocessing(self):
        
        # --- download model ---
        if not Path("./checkpoint/model3_patch25.npy").is_file():
            print(f"Patch-CNN model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['Patch_CNN'], self.DOWNLOAD_PATH['Patch_CNN'])
            print(f"Patch-CNN model checkpoint downloaded.")
        
        # --- feature/melody extraction ---
        self.feature, self.pitch = self.feature_extraction(self.args.wavfile_dir, self.args.use_cp)
        
        # --- use groundtruth pitch if selected ---
        if self.args.use_groundtruth:
            self.pitch = np.load(self.args.pitch_gt_dir)

        # --- save extracted feature/pitch ---
        if self.args.save_extracted:
            self.save_pitch_contour(self.pitch, self.args.pitch_dir)
            self.save_feature(self.feature, self.args.feat_dir)
    
    def feature_extraction(self, wavfile_dir, use_cp):
        # numpy/cupy for CFP extraction
        print(f"Feature extraction start...")
        if use_cp:
            try:
                feature, pitch = feature_extraction_cp(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                       batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                       pin_memory=self.args.pin_memory, device=self.device)
            except:
                if not CUPY_EXIST:
                    raise ImportError(f"CuPy package need to be installed to enable --use_cp")
                else:
                    self._free_gpu()
                    print(f"Filesize too large. Trying with numpy solution.")
                    feature, pitch = feature_extraction_np(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                           batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                           pin_memory=self.args.pin_memory, device=self.device)
        else:
            feature, pitch = feature_extraction_np(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                   batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                   pin_memory=self.args.pin_memory, device=self.device)
        
        # --- free GPU ---
        self._free_gpu()
        
        print(f"Feature successfully extracted.")
        return feature, pitch
    
    def load_model(self):
        # load melody extraction model
        # load note segmentation model
        # model.to(device)
        self.feature_extractor = PyramidNet_ShakeDrop(depth=110, alpha=270, shakedrop=True)
        
        # --- load checkpoint ---
        checkpoint = torch.load(self.args.checkpoint_file)
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor = amp.initialize(
            self.feature_extractor,
            opt_level=self.args.amp_level
        )
        self.feature_extractor.load_state_dict(checkpoint['model'])
        amp.load_state_dict(checkpoint['amp'])
        
        # --- evaluate mode ---
        self.feature_extractor.eval()
        
    def voice_transcription(self):
        # --- data loader ---
        test_dataset = EvalDataset(self.feature)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                                  num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
        # --- test loop ---
        outputs = []
        for batch_idx, feat in enumerate(tqdm(test_loader)):
            feat = feat.to(self.device)
            sdt_hat = self.feature_extractor.forward(feat)
            sdt_hat = torch.nn.functional.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
            outputs.append(sdt_hat.detach().cpu())
        outputs = torch.cat(outputs)
        
        # --- post processing ---
        p_np = self.pitch
        predict_on_notes_np = outputs.numpy()
        
        pitch_intervals, sSeq_np, dSeq_np, onSeq_np, offSeq_np, conflict_ratio = Smooth_sdt6_modified(predict_on_notes_np, threshold=0.5) # list of onset secs, ndarray
        freq_est = Naive_pitch(p_np, pitch_intervals)
        
        # --- for midi ---
        self.est = np.hstack((pitch_intervals, freq_est.reshape((-1,1))))
        self.sdt = np.hstack((sSeq_np.reshape((-1,1)), dSeq_np.reshape((-1,1)), onSeq_np.reshape((-1,1)), offSeq_np.reshape((-1,1))))
        
        # --- free GPU ---
        self._free_gpu()
        
    def gen_midi(self):
        # generate midi file from prediction or from groundtruth
        self.midi = Est2MIDI(self.est)
        
    def gen_wav(self):
        # generate wav from midi file
        self.synth_midi = self.midi.synthesize().astype(np.float32)
        
    def save_midi(self, midi: pretty_midi.pretty_midi.PrettyMIDI, save_dir: Path):
        # save midi file
        print(f"Writing midi...")
        save_dir.mkdir(parents=True, exist_ok=True)
        midi.write(str(save_dir / f"{self.args.name}.mid"))
        print(f"Midi successfully saved to {save_dir}")        
        
    def save_wav(self, synth_midi: np.ndarray, save_dir: Path):
        # save wav file
        print(f"Writing wav...")
        save_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(save_dir / f"{self.args.name}.wav", 44100, synth_midi)
        print(f"Wav successfully saved to {save_dir}")        
        
    def save_pitch_contour(self, pitch: np.ndarray, save_dir: Path):
        # save extracted melody
        print(f"Writing pitch contour information...")
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{self.args.name}_pitch.npy", pitch)
        print(f"Pitch contour successfully saved to {save_dir}")
        
    def save_feature(self, feature: np.ndarray, save_dir: Path):
        # save extracted feature
        print(f"Writing CFP feature information...")
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{self.args.name}_feat.npy", feature)
        print(f"Feature successfully saved to {save_dir}")        