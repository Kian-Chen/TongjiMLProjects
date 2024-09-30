import os
import torch
from models import (ModernTCN, Flashformer, iFlashformer, Flowformer,
                    iFlowformer, Informer, iInformer, Reformer, iReformer,
                    Transformer, iTransformer, DLinear, LSTM)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ModernTCN': ModernTCN,
            'Flashformer': Flashformer,
            'iFlashformer': iFlashformer,
            'Flowformer ': Flowformer,
            'iFlowformer': iFlowformer,
            'Informer': Informer,
            'iInformer': iInformer,
            'Reformer': Reformer,
            'iReformer': iReformer,
            'Transformer': Transformer,
            'iTransformer': iTransformer,
            'DLinear': DLinear,
            'LSTM': LSTM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass