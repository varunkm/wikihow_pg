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
                               tokenize=tokenizer,
                               batch_first=True, # model uses batch first tensors for some reason
                               include_lengths=True,
                               use_vocab=True)

    headline_field = data.Field(sequential=True,
                                init_token=SOS,
                                eos_token=EOS,
                                tokenize=tokenizer,
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
                                            device='cpu', # -1 mean cpu and 0 or None mean gpu
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
        batch_size = article_t.size()[0]
        assert article_lens_t.size()[0] == batch_size
        assert headline_t.size()[0] == batch_size
        assert headline_lens_t.size()[0] == batch_size
        
        # need to sort article_t by each examples respective length
        article_lens_t, new_indices = torch.sort(article_lens_t, 0, descending=True)
        article_t = article_t[new_indices, :]
        print("batch", batch_counter)
        optimizer.zero_grad()
        if config.use_gpu:
            article_t = article_t.cuda()
            article_lens_t = article_lens_t.cuda()
            headline_t = headline_t.cuda()
            headline_lens_t = headline_lens_t.cuda()

        # article_t :: (B x N) where N is max sequence length of batch
        encoder_outputs, encoder_feature, encoder_hidden = \
          model.encoder(article_t, article_lens_t)

        c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
        s_t_1 = model.reduce_state(encoder_hidden)
        max_article_len = torch.max(article_lens_t).item()
        assert article_t.size() == (batch_size, max_article_len)
        enc_padding_mask = np.zeros((batch_size, max_article_len), dtype=np.float32)
        # enc_padding_mask is 1 for all non padding words in article_t
        max_art_oov = 0 # max oov words in this batch
        for b in range(batch_size):
            art_oov = 0
            for j in range(article_lens_t[b]):
                numerical = article_t[b][j]
                if article_field.vocab.itos[numerical] == '<unk>':
                    art_oov += 1 # count oov words in each example in the batch
                enc_padding_mask[b][j] = 1 # also, fill in the enc_padding_mask in the same loop

            max_art_oov = max(art_oov, max_art_oov) # record max oov words in batch
        
        # extra zeros
        extra_zeros = None
        if config.pointer_gen and max_art_oov > 0:
            extra_zeros = Variable(torch.zeros(batch_size, max_art_oov))
        
        # coverage
        coverage = None
        if config.is_coverage:
            coverage = Variable(torch.zeros(article_t.size()))

        # dec_padding_mask
        dec_padding_mask = np.zeros((batch_size, max(headline_lens_t)), dtype=np.float32)
        for b in range(batch_size):
            for j in range(headline_lens_t[b]):
                dec_padding_mask[b][j] = 1
        
        enc_padding_mask = Variable(torch.from_numpy(enc_padding_mask)).float()
        dec_padding_mask = Variable(torch.from_numpy(dec_padding_mask)).float()
        
        # headline_t has B examples. Each has SOS and EOS
        # dec_batch has SOS but not EOS, target has EOS but not SOS.
        dec_batch = headline_t[:, :-1]
        target_batch = headline_t[:, 1:]
        dec_lens_var = Variable(headline_lens_t.float())
        
        if config.use_gpu:
            c_t_1 = c_t_1.cuda()
            s_t_1 = s_t_1.cuda()
            enc_padding_mask = enc_padding_mask.cuda()
            dec_padding_mask = dec_padding_mask.cuda()
            if extra_zeros:
                extra_zeros = extra_zeros.cuda()
            if coverage:
                coverage = coverage.cuda()
            dec_batch = dec_batch.cuda()
            target_batch = target.cuda()
            dec_lens_var = dec_lens_var.cuda()
        max_dec_len = max(headline_lens_t)
        
        step_losses = []
        for t in range(min(max_dec_len - 1, config.max_dec_steps)):
            y_t_1 = dec_batch[:, t] # get y_(t-1) for all members of batch
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
              model.decoder(y_t_1, s_t_1, encoder_outputs,
                                 encoder_feature, enc_padding_mask, c_t_1, extra_zeros,
                                 article_t,coverage, t)

            target = target_batch[:, t] # targets for all examples at time t
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, t]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        norm = clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.reduce_state.parameters(), config.max_grad_norm)

        optimizer.step()
        print("Loss: ", loss.item())
        batch_counter += 1

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

