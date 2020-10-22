# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:44:27 2019

@author: Austin Hsu
"""

import pretty_midi
import numpy as np
import sys

def Est2MIDI(est):
    
    # --- I/O Description ---
    # === Input ===
    # est: evaluated result from Transciption.py
    # === Output ===
    # piano_chord: MIDI result
    # -----------------------
    
    # --- Data Preparation ---
    t1 = est[:,0].reshape((est.shape[0],))
    t2 = est[:,1].reshape((est.shape[0],))
    f  = est[:,2].reshape((est.shape[0],))
    # ------------------------
    
    # --- PrettyMIDI ---
    # Create a PrettyMIDI object
    piano_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    # Iterate over note names, which will be converted to note number later
    conflicts = 0
    for f_idx in range(f.shape[0]):
        # Retrieve the MIDI note number for this note name
        note_pitch = int(round(pretty_midi.hz_to_note_number(f[f_idx])))
        if note_pitch > 127 or note_pitch < 0:
            #print(note_pitch)
            conflicts += 1
            note_pitch = 0 if note_pitch < 0 else 127
        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(
            velocity=100, pitch=note_pitch, start=t1[f_idx], end=t2[f_idx])
        # Add it to our cello instrument
        piano.notes.append(note)
    print('Conflict Times:', conflicts, ' | Conflict Ratio:', conflicts/f.shape[0], '(%d/%d)'%(conflicts,f.shape[0]))
    piano_chord.remove_invalid_notes()
    # Add the cello instrument to the PrettyMIDI object
    piano_chord.instruments.append(piano)
    # ------------------
    
    return piano_chord