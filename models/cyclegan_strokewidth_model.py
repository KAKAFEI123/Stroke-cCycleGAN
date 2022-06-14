import torch
import itertools
from .base_model import BaseModel
from . import networks
import numpy
import cv2
import itertools

class CycleGANStrokewidthModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.stroke_lamda = 1
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        self.visual_names = visual_names_A + visual_names_B
        self.model_names = ['G_A','Embedding']
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.num_widthdegree_emb, opt.ngf, 
                                    opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G0(opt.output_nc, opt.input_nc, opt.ngf, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netEmbedding = torch.nn.Embedding(opt.num_widthdegree, opt.num_widthdegree_emb).cuda()
        self.netEmbedding = torch.nn.DataParallel(self.netEmbedding, self.gpu_ids)


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.stroke_width = input['stroke_width'].cuda()
        self.strokeWidthEmb = self.netEmbedding(self.stroke_width)
        

    def forward(self):
        self.fake_B = self.netG_A(self.real_A, self.strokeWidthEmb) 
        self.rec_A = self.netG_B(self.fake_B) 
        self.fake_A = self.netG_B(self.real_B)  
        self.rec_B = self.netG_A(self.fake_A, self.strokeWidthEmb)
