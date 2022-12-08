import os,time
import torch

class HP:
    def __init__(self):
        self.args = self.predefined_args()

    def predefined_args(self):
        args = {}
        args['createDataset'] = True
        args['playDataset'] = 10
        args['device'] = "cuda:1" if torch.cuda.is_available() else "cpu"
        args['rootDir'] = './artifacts/'#bolt.ARTIFACT_DIR

        args['maxLength'] = 100
        args['vocabularySize'] = 40000

        args['n_hiddens'] = 100 #300
        args['n_updates']  = 5000
        args['n_residual_hiddens']  = 32
        args['n_residual_layers']  = 2
        args['numLayers'] = 1
        args['softmaxSamples'] = 0
        args['initEmbeddings'] = True
        args['embeddingSize'] = 512 #300

        # args['embeddingSource'] = "GoogleNews-vectors-negative300.bin"

        args['vq_embedding_dim'] = 64
        args['vq_n_embeddings'] = 512
        args['vq_beta'] = 0.25
        args['learning_rate'] = 3e-4
        args['log_interval'] = 50

        args['adaptive_softmax_cutoff'] = None
        args['decoder_learned_pos'] = False
        args["cross_self_attention"] = False
        args['decoder_normalize_before'] = False
        args['adaptive_softmax_dropout'] = 0
        args['tie_adaptive_weights'] = False
        args['eval_tokenized_bleu'] = False
        # args['adaptive_softmax_factor'] =
        # args['tie_adaptive_proj'] =
        args["quant_noise_pq"] = 0
        args["quant_noise_pq_block_size"] = 8
        args['activation_fn'] = 'relu'
        args["activation_dropout"] = 0.0
        # args["relu_dropout"] =
        args['decoder_normalize_before'] = False
        # args["char_inputs"]
        args['decoder_ffn_embed_dim'] = 1024
        args['decoder_attention_heads'] = 4
        args['attention_dropout'] = 0.0
        args["encoder_embed_dim"] = args['vq_embedding_dim']
        args["layernorm_embedding"] = False

        args['numEpochs'] = 1000
        args['saveEvery'] = 2000
        args['batchSize'] = 32
        args['learningRate'] = 0.001
        args['dropout'] = 0.3
        args['clip'] = 5.0

        args['maxLengthEnco'] = args['maxLength']
        args['maxLengthDeco'] = args['maxLength'] + 1

        args['temperature'] =1.0
        args['save'] =True
        args['filename'] = time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()



        return args

args = HP().args