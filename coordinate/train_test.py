import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torchtext import data
import pandas as pd
import numpy as np
import spacy
import sys
sys.path.append('../')
from data_util import config
from training_ptr_gen.model import Model
from torch.optim import Adagrad
from train import train_batch


if config.use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

SOS = '[START]'
EOS = '[STOP]'

def main():
    TEST_FRAC = 0.2
    DATA_SET_SAMPLE = 100
    SAMPLE = True
    print("Loading dataset")
    whole_dataset = pd.read_csv('../data/wikihowAll.csv')
    if SAMPLE:
        whole_dataset = whole_dataset.sample(n=DATA_SET_SAMPLE)
    mask = np.random.rand(len(whole_dataset)) < (1 - TEST_FRAC)
    train = whole_dataset[mask]
    test = whole_dataset[~mask]

    train.to_csv('../data/train.csv', index=False)
    test.to_csv('../data/test.csv', index=False)

    article_field = data.Field(sequential=True,
                               stop_words=['\n'],
                               init_token=SOS,
                               eos_token=EOS,
                               batch_first=True, # model uses batch first tensors for some reason
                               include_lengths=True,
                               use_vocab=True)

    headline_field = data.Field(sequential=True,
                                init_token=SOS,
                                eos_token=EOS,
                                stop_words=['\n'],
                                batch_first=True,
                                include_lengths=True,
                                use_vocab=True)
    
    fields = [
        ('headline', headline_field),
        ('title', None),
        ('text', article_field)
    ]
    print("Creating train and test sets")
    train_set, test_set = data.TabularDataset.splits(path='../data',
                                                     format='csv',
                                                     train='train.csv',
                                                     validation='test.csv',
                                                     fields=fields,
                                                     skip_header=True)
    print("building vocabs")
    article_field.build_vocab(train_set, test_set, max_size=config.vocab_size)
    headline_field.build_vocab(train_set, test_set, max_size=config.vocab_size)

    train_bch, test_bch = data.BucketIterator.splits(datasets=(train_set, test_set), # specify train and validation Tabulardataset
                                            batch_sizes=(config.batch_size,config.batch_size),  # batch size of train and validation
                                            device=torch.device('cpu'), # -1 mean cpu and 0 or None mean gpu
                                            repeat=False)

    # construct the model first
    model = Model()
    params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
             list(model.reduce_state.parameters())
    initial_lr = config.lr_coverage if config.is_coverage else config.lr
    optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
    training_batches = BatchGenerator(train_bch, 'text', 'headline')
    batch_counter = 0
    print("training")
    for ((article_t, article_lens_t), (headline_t, headline_lens_t)) in training_batches:
        batch_loss = train_batch(article_t, article_lens_t, headline_t, headline_lens_t,
                                 model, optimizer, article_field, headline_field)
        batch_counter += 1
        print("Batch %d done, loss: %f" % (batch_counter, batch_loss))
    

def clean(text):
    text = text.replace('\n', '')
    return text.strip()

def tokenizer(s):
    return [w.text.lower() for w in nlp(clean(s))]

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)



if __name__ == "__main__":
    main()
