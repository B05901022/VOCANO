# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:01:13 2020

@author: Austin Hsu
"""

import torch
    
class EvalDataset(torch.utils.data.Dataset):
    
    def __init__(self, feature, num_feat=9, k=9):
        
        # --- Args ---
        self.window_size = 2*k+1
        self.k = k
        
        self.feature = torch.from_numpy(feature).float()
        self.feature = self.feature.reshape((num_feat,1566//num_feat,-1))
        self.len = self.feature.shape[-1]
        
        # --- Pad Length ---
        self.feature = torch.cat([
            torch.zeros((num_feat,1566//num_feat,k)),
            self.feature,
            torch.zeros((num_feat,1566//num_feat,k))
            ], dim=-1)
        
        # --- Transform ---
        self.data_normalize = lambda x: (x-torch.mean(x))/(torch.std(x)+1e-8)
        self._DataPreprocess()
        
    def __getitem__(self, index):
        frame_feat = self.feature[:, :, index:index+self.window_size]
        return frame_feat
    
    def _DataPreprocess(self):
        # --- Normalize ---
        self.feature = self.data_normalize(self.feature)
    
    def __len__(self):
        return self.len
