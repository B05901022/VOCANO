# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:11:59 2020

@author: Austin Hsu
"""

import numpy as np

def find_first_bellow_thres(aSeq):
    activate = False
    first_bellow_frame = 0
    for i in range(len(aSeq)):
        if aSeq[i] > 0.5:
            activate = True
        if activate and aSeq[i] < 0.5:
            first_bellow_frame = i
            break
    return first_bellow_frame

def Smooth_sdt6_modified(predict_sdt, threshold=0.5):
    # predict shape: (time step, 3)
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    sSeq = []
    dSeq = []
    onSeq = []
    offSeq = []
    
    for num in range(predict_sdt.shape[0]):
        if num > 1 and num < predict_sdt.shape[0]-2:
            sSeq.append(predict_sdt[num][0].astype(np.float64))
            dSeq.append(predict_sdt[num][1].astype(np.float64))
            onSeq.append(np.dot(predict_sdt[num-2:num+3, 3], Filter) / 2.5)
            offSeq.append(np.dot(predict_sdt[num-2:num+3, 5], Filter) / 2.5)

        else:
            sSeq.append(predict_sdt[num][0].astype(np.float64))
            dSeq.append(predict_sdt[num][1].astype(np.float64))
            onSeq.append(predict_sdt[num][3])
            offSeq.append(predict_sdt[num][5])   
    
    ##############################
    # Peak strategy
    ##############################
    
    # find peak of transition
    # peak time = frame*0.02+0.01
    onpeaks = []
    if onSeq[0] > onSeq[1] and onSeq[0] > onSeq[2] and onSeq[0] > threshold:
        onpeaks.append(0)
    if onSeq[1] > onSeq[0] and onSeq[1] > onSeq[2] and onSeq[1] > onSeq[3] and onSeq[1] > threshold:
        onpeaks.append(1)
    for num in range(len(onSeq)):
        if num > 1 and num < len(onSeq)-2:
            if onSeq[num] > onSeq[num-1] and onSeq[num] > onSeq[num-2] and onSeq[num] > onSeq[num+1] and onSeq[num] > onSeq[num+2] and onSeq[num] > threshold:
                onpeaks.append(num)

    if onSeq[-1] > onSeq[-2] and onSeq[-1] > onSeq[-3] and onSeq[-1] > threshold:
        onpeaks.append(len(onSeq)-1)
    if onSeq[-2] > onSeq[-1] and onSeq[-2] > onSeq[-3] and onSeq[-2] > onSeq[-4] and onSeq[-2] > threshold:
        onpeaks.append(len(onSeq)-2)


    offpeaks = []
    if offSeq[0] > offSeq[1] and offSeq[0] > offSeq[2] and offSeq[0] > threshold:
        offpeaks.append(0)
    if offSeq[1] > offSeq[0] and offSeq[1] > offSeq[2] and offSeq[1] > offSeq[3] and offSeq[1] > threshold:
        offpeaks.append(1)
    for num in range(len(offSeq)):
        if num > 1 and num < len(offSeq)-2:
            if offSeq[num] > offSeq[num-1] and offSeq[num] > offSeq[num-2] and offSeq[num] > offSeq[num+1] and offSeq[num] > offSeq[num+2] and offSeq[num] > threshold:
                offpeaks.append(num)

    if offSeq[-1] > offSeq[-2] and offSeq[-1] > offSeq[-3] and offSeq[-1] > threshold:
        offpeaks.append(len(offSeq)-1)
    if offSeq[-2] > offSeq[-1] and offSeq[-2] > offSeq[-3] and offSeq[-2] > offSeq[-4] and offSeq[-2] > threshold:
        offpeaks.append(len(offSeq)-2)

    # determine onset/offset by silence, duration
    # intervalSD = [0,1,0,1,...], 0:silence, 1:duration
    if len(onpeaks) == 0 or len(offpeaks) == 0:
        return None
    
    # Clearing out offsets before first onset (since onset is more accurate)
    orig_offpeaks = offpeaks
    offpeaks = [i for i in orig_offpeaks if i>onpeaks[0]]
    
    Tpeaks = onpeaks + offpeaks
    Tpeaks.sort()

    intervalSD = [0]

    for i in range(len(Tpeaks)-1):
        current_sd = 0 if sum(sSeq[Tpeaks[i]:Tpeaks[i+1]]) > sum(dSeq[Tpeaks[i]:Tpeaks[i+1]]) else 1
        intervalSD.append(current_sd)
    intervalSD.append(0)


    MissingT= 0
    AddingT = 0
    est_intervals = []
    t_idx = 0
    while t_idx < len(Tpeaks):
        if t_idx == len(Tpeaks)-1:
            break
        if t_idx == 0 and Tpeaks[t_idx] not in onpeaks:
            t_idx += 1

        if Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in offpeaks:
            if Tpeaks[t_idx] == Tpeaks[t_idx+1]:
                t_idx += 1
                continue
            if Tpeaks[t_idx+1] > Tpeaks[t_idx]+1: 
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*Tpeaks[t_idx+1]+0.01])
            assert(Tpeaks[t_idx] < Tpeaks[t_idx+1])
            t_idx += 2
        elif Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in onpeaks:
            offset_inserted = find_first_bellow_thres(dSeq[Tpeaks[t_idx]:Tpeaks[t_idx+1]]) + Tpeaks[t_idx]
            if offset_inserted != Tpeaks[t_idx] and offset_inserted > Tpeaks[t_idx]+1:
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*offset_inserted+0.01])
                AddingT += 1
                assert(Tpeaks[t_idx] < offset_inserted)
            else:
                MissingT += 1
            t_idx += 1
        elif Tpeaks[t_idx] in offpeaks:
            t_idx += 1
    
    #print("Conflict ratio: ", MissingT/(len(Tpeaks)+AddingT))

    # Modify 1
    sSeq_np = np.ndarray(shape=(len(sSeq),), dtype=float, buffer=np.array(sSeq))
    dSeq_np = np.ndarray(shape=(len(dSeq),), dtype=float, buffer=np.array(dSeq))
    onSeq_np = np.ndarray(shape=(len(onSeq),), dtype=float, buffer=np.array(onSeq))
    offSeq_np = np.ndarray(shape=(len(offSeq),), dtype=float, buffer=np.array(offSeq))

    return np.ndarray(shape=(len(est_intervals),2), dtype=float, buffer=np.array(est_intervals)),  sSeq_np, dSeq_np, onSeq_np, offSeq_np, MissingT/(len(Tpeaks)+AddingT)

def Naive_pitch(pitch_step, pitch_intervals):
    pitch_est = np.zeros((pitch_intervals.shape[0],))

    for i in range(pitch_intervals.shape[0]):
        start_frame = int((pitch_intervals[i][0]-0.01) / 0.02)
        end_frame = int((pitch_intervals[i][1]-0.01) / 0.02)
        if end_frame == start_frame + 1 or end_frame == start_frame:
            pitch_est[i] = pitch_step[start_frame] if pitch_step[start_frame] != 0 else 1.0
        else:
            pitch_est[i] = np.median(pitch_step[start_frame:end_frame]) if np.median(pitch_step[start_frame:end_frame]) != 0 else 1.0

    return pitch_est

def pitch2freq(pitch_np):
    freq_l = [ (2**((pitch_np[i]-69)/12))*440 for i in range(pitch_np.shape[0]) ]
    return np.ndarray(shape=(len(freq_l),), dtype=float, buffer=np.array(freq_l))

def freq2pitch(freq_np):
    pitch_np = 69+12*np.log2(freq_np/440)
    return pitch_np

def minimumEditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if abs(char1 - char2)<0.5:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

def to_semitone(freq):
    return 12*np.log2(freq/440)+69

def eval_note_acc(gt, est):
    gt = to_semitone(gt)
    est = to_semitone(est)
    dist = minimumEditDistance(gt,est)
    note_error_rate = float(dist) / len(est)
    note_accuracy = 1 - note_error_rate
    return note_accuracy