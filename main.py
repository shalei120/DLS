import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils,json
from models.vqvae import VQVAE
from textdata import TextData_COCO, make_batches
from Hyperparameters import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from sequence_generator import SequenceGenerator
import search
from argparse import Namespace
# parser = argparse.ArgumentParser()
#
# """
# Hyperparameters
# """
# timestamp = utils.readable_timestamp()
#
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--n_updates", type=int, default=5000)
# parser.add_argument("--n_hiddens", type=int, default=128)
# parser.add_argument("--n_residual_hiddens", type=int, default=32)
# parser.add_argument("--n_residual_layers", type=int, default=2)
# parser.add_argument("--embedding_dim", type=int, default=64)
# parser.add_argument("--n_embeddings", type=int, default=512)
# parser.add_argument("--beta", type=float, default=.25)
# parser.add_argument("--learning_rate", type=float, default=3e-4)
# parser.add_argument("--log_interval", type=int, default=50)
# parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
# parser.add_argument("-save", action="store_true")
# parser.add_argument("--filename",  type=str, default=timestamp)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--data', '-d')

cmdargs = parser.parse_args()

index2dataname = {
    0: 'CIFAR10',
    1: 'MSCOCO'
}

if cmdargs.gpu is None:
    args['device'] = 'cpu'
else:
    args['device'] = 'cuda:' + str(cmdargs.gpu)


if cmdargs.data is None:
    args['dataset'] = 'CIFAR10'
else:
    args['dataset'] = index2dataname[int(cmdargs.data)]

args['dataset'] = 'MSCOCO'
args['mode'] = 't->t'
# args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args['save']:
    print('Results will be saved in '+ args['rootDir'] + args['filename'] + '.pth')

"""
Load data and define batch data loaders
"""

class task:
    def __init__(self):
        self.external_info = {
            'w2i': None,
            'i2w': None
        }
        self.batches = dict()
        if args['dataset'] == 'CIFAR10':
            training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
                args['dataset'], args['batchSize'])
            self.batches ['train'] = make_batches(training_loader)
            self.batches['val'] = make_batches(validation_loader)
        elif args['dataset'] == 'MSCOCO':
            textdata = TextData_COCO()
            self.external_info['w2i'] = textdata.word2index
            self.external_info['i2w'] = textdata.index2word
            self.batches ['train'] = textdata.getBatches()
            self.batches['val'] = textdata.getBatches('valid')
            self.batches['test'] = textdata.getBatches('test')


        """
        Set up VQ-VAE model with components defined in ./models/ folder
        """

        self.model = VQVAE(args['n_hiddens'], args['n_residual_hiddens'],
                      args['n_residual_layers'], args['vq_n_embeddings'], args['vq_embedding_dim'],
                      args['vq_beta'], self.external_info).to(args['device'])

        gen_args = json.loads('{"beam":5,"max_len_a":1.2,"max_len_b":10}')
        self.sequence_generator = self.build_generator(
            [self.model], Namespace(**gen_args)
        )

        """
        Set up optimizer and training loop
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=args['learning_rate'], amsgrad=True)

        self.model.train()

        self.results = {
            'n_updates': 0,
            'recon_errors': [],
            'loss_vals': [],
            'perplexities': [],
        }

    def run(self):
        self.train()


    def train(self):

        in_mode, out_mode = args['mode'].split('->')

        CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(args['n_updates']):
            for batch in self.batches['train']:
                if in_mode == 't':
                    x = batch.encoderSeqs.to(args['device'])
                    x_decinput = batch.decoderSeqs.to(args['device'])
                    x_target = batch.targetSeqs.to(args['device'])
                elif in_mode == 'i':
                    x = batch.encoder_image.to(args['device'])
                    x_decinput = None
                    x_target = None

                self.optimizer.zero_grad()

                embedding_loss, x_hat, perplexity = self.model(x, text_ext = {
                    'dec_input':x_decinput,
                    'target': x_target
                },input_mode = in_mode, output_mode = out_mode)

                if in_mode == 'i':
                    recon_loss = torch.mean((x_hat - x)**2) / x_train_var
                    loss = recon_loss + embedding_loss
                elif in_mode == 't':
                    recon_loss = CEloss(torch.transpose(x_hat, 1, 2), x_target)
                    recon_loss = recon_loss.mean()
                    loss = recon_loss + embedding_loss

                loss.backward()
                self.optimizer.step()

                self.results["recon_errors"].append(recon_loss.cpu().detach().numpy())
                self.results["perplexities"].append(perplexity.cpu().detach().numpy())
                self.results["loss_vals"].append(loss.cpu().detach().numpy())
                self.results["n_updates"] = i

                if i % args['log_interval'] == 0:
                    """
                    save model and print values
                    """
                    self.evaluate('val')
                    if args['save']:
                        hyperparameters = args
                        utils.save_model_and_results(
                            self.model, self.results, hyperparameters, args['filename'])

                    print('Update #', i, 'Recon Error:',
                          np.mean(self.results["recon_errors"][-args['log_interval']:]),
                          'Loss', np.mean(self.results["loss_vals"][-args['log_interval']:]),
                          'Perplexity:', np.mean(self.results["perplexities"][-args['log_interval']:]))


    def evaluate(self, dataset_name):

        in_mode, out_mode = args['mode'].split('->')

        hyps, refs = [], []

        for batch in self.batches[dataset_name]:
            if in_mode == 't':
                x = batch.encoderSeqs.to(args['device'])
                raw_target = batch.raw_source
            elif in_mode == 'i':
                x = batch.encoder_image.to(args['device'])
                raw_target = None

            with torch.no_grad():
                gen_out = self.sequence_generator.generate({
                    'net_input': {
                        'src_tokens': x,
                        'input_mode': in_mode
                    }
                }, prefix_tokens=None, constraints=None
                )
            target_seq = raw_target

            # print([len(g) for g in gen_out])
            for i in range(len(gen_out)):
                try:
                    indexes = gen_out[i][0]["tokens"]
                except:
                    print(len(gen_out),i,gen_out)
                    indexes = gen_out[i][0]["tokens"]

                indexes = indexes.int().cpu()
                h = self.Make_string(self.external_info['w2i'], indexes, '</w>', unk_string="UNKNOWNTOKENINREF")

                endpos = h.index('<|endoftext|>')
                h = h[:endpos]
                r = target_seq[i]
                hyps.append(h)
                refs.append([r])

                if i == 0:
                    print("example hypothesis: " + h)
                    print("example reference: " + r)

        bleu_ori = corpus_bleu(gold_ans, pred_ans)
        return bleu_ori

    def build_generator(self, models, args):

        self.target_dictionary = self.external_info['w2i']

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy
        )

    def Make_string(
        self,tgt_dict,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        # if torch.is_tensor(tensor) and tensor.dim() == 2:
        #     return "\n".join(
        #         self.Make_string(tgt_dict, t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
        #         for t in tensor
        #     )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(tgt_dict['<|endoftext|>'])

        def token_string(i):
            i = i.item()
            # if i == tgt_dict['<unk>']:
            #     if unk_string is not None:
            #         return unk_string
            #     else:
            #         return tgt_dict['<unk>']
            # else:
            return self.external_info['i2w'][i]

        if '<|startoftext|>' in  tgt_dict:
            extra_symbols_to_ignore.add(tgt_dict['<|startoftext|>'])

        sent = separator.join(
            token_string(i)
            for i in tensor
            if i.data not in extra_symbols_to_ignore
        )

        return sent.replace(bpe_symbol,'')


if __name__ == "__main__":
    t = task()
    t.run()
