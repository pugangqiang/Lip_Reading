from .3DCvT import VideoCNN
import torch
import torch.nn as nn
import random


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()   
        
        self.args = args
        
        self.video_cnn = VideoCNN()        
        if(self.args.border):
            in_dim = 1920 + 1
        else:
            in_dim = 1920
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        
            

        self.v_cls = nn.Linear(1024*2, self.args.n_class)     
        self.dropout = nn.Dropout(p=dropout)        

    def forward(self, v, border=None):
        self.gru.flatten_parameters()
                                
        f_v = self.video_cnn(v)  
        f_v = self.dropout(f_v)        
    
        if(self.args.border):
            border = border[:,:,None]
            f = torch.cat([f_v, border], -1)
            h, _ = self.gru(f)
        else:            
            h, _ = self.gru(f_v)
  
                                                                                                        
        y_v = self.v_cls(self.dropout(h)).mean(1)
        
        return y_v