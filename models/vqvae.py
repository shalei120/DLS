
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from TextDecoder import TransformerDecoder
from Hyperparameters import args
from torch import Tensor

from typing import Optional,Any, Callable, Dict, List, Tuple
import json
import clip

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, external_info,  save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.w2i = external_info['w2i']
        self.i2w = external_info['i2w']
        # with open('./tokens.txt','w') as f:
        #     num = len(self.i2w)
        #     for i in range(num):
        #         f.write(self.i2w[i])
        #         f.write('\n')
        #     f.close()
        self.load_clip()
        # self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.encoder = self.clipmodel.encode_image
        self.TextEncoder = self.clipmodel.encode_text
        self.pre_quantization_conv = nn.Conv2d(
            self.clip_tf_width, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        self.TextDecoder = TransformerDecoder(self.w2i,self.i2w, self.embedding_tgt, embed_dim = self.clip_tf_width)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None



    def load_clip(self):
        self.clipmodel, self.clip_preprocess, info = clip.load("ViT-B/32")
        self.clipmodel.to(args['device']).eval()
        self.input_resolution = self.clipmodel.visual.input_resolution
        self.context_length = self.clipmodel.context_length
        self.vocab_size = self.clipmodel.vocab_size
        self.embedding_tgt = self.clipmodel.token_embedding
        args['embeddingSize'] = self.embedding_tgt.weight.size(1)
        self.clip_tf_width = info['transformer_width']

    def build_image2image(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity

    def encode_into_DLS(self, x, input_mode = 't'):
        if input_mode == 't':
            z_e = self.TextEncoder(x)
            z_e  = z_e.unsqueeze(2).unsqueeze(3)
        elif input_mode == 'i':
            z_e = self.encoder(x)
        z_e = z_e.to(torch.float32)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        z_q = z_q[:,:,0,0]

        return embedding_loss, z_q, perplexity


    def forward(self, x, text_ext=None, input_mode = 't', output_mode = 't', verbose=False):
        '''
        :param x:
        :param input_mode: t for text, i for image
        :param output_mode:
        :param verbose:
        :return:
        '''
        embedding_loss, z_q, perplexity = self.encode_into_DLS(x, input_mode)
        if output_mode == 't':
            x_hat, _ = self.TextDecoder(prev_output_tokens = text_ext['dec_input'], encoder_out = {
                'encoder_out': z_q.unsqueeze(0)
            })
        elif output_mode == 'i':
            x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)




