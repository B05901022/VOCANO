# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:27:03 2020

@author: Austin Hsu
"""

import argparse
from pathlib import Path
from .core import SingingVoiceTranscription

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="vocano.preprocess",
        description="Extract CFP feature and melody from given wavfile."
        )
    
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="default",
        help="Name of generated files."
        )
    parser.add_argument(
        "-fd",
        "--feat_dir",
        type=Path,
        default="generated/feat",
        help="Directory to store CFP feature."
        )
    parser.add_argument(
        "-pd",
        "--pitch_dir",
        type=Path,
        default="generated/pitch",
        help="Directory to store extracted melody."
        )
    parser.add_argument(
        "-wd",
        "--wavfile_dir",
        type=Path,
        default="example.wav",
        help="File to be transcribed."
        )
    parser.add_argument(
        "-s",
        "--save_extracted",
        action="store_true",
        help="Save pre-extracted feat/melody."
        )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to be used. Can be set to \"auto\ for selecting the least used gpu."
        )
    parser.add_argument(
        "-use_cp",
        "--use_cp",
        action="store_true",
        help="Use CuPy for gpu accelerated feature extraction. Recommended for machines that have large gpu memory."
        )
    parser.add_argument(
        "-use_gt",
        "--use_groundtruth",
        action="store_true",
        help="Use ground truth pitch. Pitch groundtruth needs to be selected."
        )
    parser.add_argument(
        "-bz",
        "--batch_size",
        type=int,
        default=64,
        help="Batchsize for inferencing."
        )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for inferencing."
        )
    parser.add_argument(
        "-pn",
        "--pin_memory",
        action="store_true",
        help="Pin memory for inferencing."
        )
    args = parser.parse_args()
    
    solver = SingingVoiceTranscription(args)
    
    solver.data_preprocessing()
    