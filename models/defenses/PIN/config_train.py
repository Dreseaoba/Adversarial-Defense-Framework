# -*- coding: utf-8 -*-
"""
configuration of AID
RenMin
"""

class Config(object):
    def __init__(self):
        self.epoch = 50
        self.batch = 5
        self.lr_encoder = 1
        self.lr_pcafc = 1
        self.momentum = 0.9
        self.decay_step = 20
        self.decay_rate = 0.8
        
        self.noise_scale = 0.04
        self.lamb_sparse = 0.015
        self.lamb_mean = 1.
        
        self.data_folder = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/datasets/deepfake/train_pca'
        self.ckpt_dae = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/checkpoint/dae2_200.pth'
        self.eigen_face = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/face_eigen/dpfk32_eig.pth'
        
        self.save_step = 2
        self.encoder_path = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/checkpoint/encoder'
        self.pcafc_path = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/checkpoint/pca_fc'

        # eval
        self.en_ckpt = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/checkpoint/encoder_6.pth'
        self.pca_ckpt = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/checkpoint/pca_fc_6.pth'


    def epochGet(self):
        return self.epoch
    def batchGet(self):
        return self.batch
    def lr_encoderGet(self):
        return self.lr_encoder
    def lr_pcafcGet(self):
        return self.lr_pcafc
    def momentumGet(self):
        return self.momentum
    def decay_stepGet(self):
        return self.decay_step
    def decay_rateGet(self):
        return self.decay_rate
    def noise_scaleGet(self):
        return self.noise_scale
    def lamb_sparseGet(self):
        return self.lamb_sparse
    def lamb_meanGet(self):
        return self.lamb_mean
    def data_folderGet(self):
        return self.data_folder
    def ckpt_daeGet(self):
        return self.ckpt_dae
    def save_stepGet(self):
        return self.save_step
    def encoder_pathGet(self):
        return self.encoder_path
    def pcafc_pathGet(self):
        return self.pcafc_path
    def eigen_faceGet(self):
        return self.eigen_face

    def ckpt_encoderGet(self):
        return self.en_ckpt
    def ckpt_PCAfcGet(self):
        return self.pca_ckpt