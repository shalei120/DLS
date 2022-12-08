
import numpy as np
import nltk  # For tokenize
from nltk.probability import FreqDist
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random, json
import string, copy
from nltk.tokenize import word_tokenize
from Hyperparameters import args
import torch
import  tarfile
import clip

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.encoder_lens = []
        self.decoder_lens = []
        self.raw_source = []
        self.raw_target = []

        self.encoder_image = []

def make_batches(loader_iter):
    bat_iter = list(loader_iter)
    batches = []
    for b in bat_iter:
        batch = Batch()
        batch.encoder_image = b[0]
        batch.label = b[1]
        batches.append(batch)
    return batches


class TextData_COCO:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, taskID = 1):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        self.taskID = taskID

        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()

        if not os.path.exists(args['rootDir']):
            os.mkdir(args['rootDir'])

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2index = {}
        self.index2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)

        self.loadCorpus()



    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format('LMbenchmark', len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        maxlen_def = args['maxLengthEnco'] #if setname == 'train' else 511

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            src,  raw_src= samples[i]

            if len(src) > maxlen_def:
                src = src[:maxlen_def]

            src_list= src.tolist()[0]
            src_list = self.del_tail_0(src_list)

            batch.encoderSeqs.append(src)
            batch.decoderSeqs.append(src_list[:-1])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(src_list[1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            assert len(batch.decoderSeqs[i]) <= maxlen_def +1

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            batch.decoder_lens.append(len(batch.targetSeqs[i]))
            batch.raw_source.append(raw_src)

        maxlen_dec = max(batch.decoder_lens)
        maxlen_enc = max(batch.encoder_lens)


        for i in range(batchSize):
            # batch.encoderSeqs[i] = batch.encoderSeqs[i] + [0] * (maxlen_enc - len(batch.encoderSeqs[i]))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [0] * (maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [0] * (maxlen_dec - len(batch.targetSeqs[i]))

        batch.encoderSeqs = torch.cat(batch.encoderSeqs, dim = 0)
        batch.decoderSeqs = torch.LongTensor(batch.decoderSeqs)
        batch.targetSeqs = torch.LongTensor(batch.targetSeqs)

        # pre_sort_list = [(a, b, c) for a, b, c  in
        #                  zip( batch.decoderSeqs, batch.decoder_lens,
        #                      batch.targetSeqs)]
        #
        # post_sorted_list = sorted(pre_sort_list, key=lambda x: x[1], reverse=True)
        #
        # batch.decoderSeqs = [a[0] for a in post_sorted_list]
        # batch.decoder_lens = [a[1] for a in post_sorted_list]
        # batch.targetSeqs = [a[2] for a in post_sorted_list]

        return batch

    def getBatches(self,setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()


        batches = []
        batch_size = args['batchSize'] #if setname == 'train' else 32
        print(len(self.datasets[setname]), setname, batch_size)
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(setname), batch_size):
                yield self.datasets[setname][i:min(i + batch_size, self.getSampleSize(setname))]

        # TODO: Should replace that by generator (better: by tf.queue)

        for index, samples in enumerate(genNextSamples()):
            # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
            batch = self._createBatch(samples)
            batches.append(batch)

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return batches

    def getSampleSize(self,setname):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

    def extract(self, tar_url, extract_path='.'):
        print(tar_url)
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)

    def loadCorpus(self):
        """Load/create the conversations data
        """
        self.basedir = './data/'

        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        self.corpus_train = self.basedir + 'captions_train2014.json'
        self.corpus_val = self.basedir + 'captions_val2014.json'
        self.fullSamplesPath = args['rootDir'] + '/COCO.pkl'  # Full sentences length/vocab

        print(self.fullSamplesPath)
        datasetExist = os.path.isfile(self.fullSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            self.word2index = self.tokenizer.encoder
            self.index2word = self.tokenizer.decoder
            # wordnum = len(self.word2index)
            # self.word2index['<unk>'] = wordnum
            # self.index2word[wordnum] = '<unk>'

            data = []
            for file_path in [self.corpus_train,self.corpus_val]:
                with open(file_path , 'r') as handle:
                    json_structure = json.load(handle)
                    captions = [anno['caption'] for anno in json_structure['annotations']]
                    data.extend(captions)

            random.shuffle(data)
            datanum = len(data)
            dataset={}
            dataset['train'] = data[:int(0.8 * datanum)]
            dataset['valid'] = data[int(0.8 * datanum):int(0.9 * datanum)]
            dataset['test'] = data[int(0.9 * datanum):]
            for setname in ['train', 'valid', 'test']:
                dataset[setname] = [(clip.tokenize(src) , src) for src in tqdm(dataset[setname])]
            self.datasets = dataset

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.fullSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.fullSamplesPath)



    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {'word2index': self.word2index,
                    'index2word': self.index2word,
                    'datasets': self.datasets
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))

        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            self.datasets = data['datasets']

    def del_tail_0(self, indexlist):
        p = -1
        while indexlist[p]==0:
            p -= 1

        return indexlist[:p+1] if p < -1 else indexlist

    # def read_word2vec(self, vocfile ):
    #     word2index = dict()
    #     word2index['<s>'] = 0
    #     word2index['<pad>'] = 1
    #     word2index['</s>'] = 2
    #     word2index['<unk>'] = 3
    #     word2index['<sep>'] = 4
    #     cnt = 5
    #     with open(vocfile, "r") as v:
    #
    #         for line in v:
    #             word = line.strip().split()[0]
    #             word2index[word] = cnt
    #             print(word,cnt)
    #             cnt += 1
    #
    #     print(len(word2index),cnt)
    #     # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
    #     print ('Dictionary Got!')
    #     return word2index
    #
    # def TurnWordID(self, words):
    #     res = []
    #     for w in words:
    #         w = w.lower()
    #         if w in self.index2word_set:
    #             id = self.word2index[w]
    #             # if id > 20000:
    #             #     print('id>20000:', w,id)
    #             res.append(id)
    #         else:
    #             res.append(self.word2index['<unk>'])
    #     return res

if __name__ == '__main__':
    TextData_COCO()