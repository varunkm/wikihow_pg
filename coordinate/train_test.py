import torch
from torchtext import data
import pandas as pd
import numpy as np
import spacy
import sys
sys.path.append('../')
from data_util import config
if config.use_gpu:
    device = 0
else:
    device = -1

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])


def main():
    TEST_FRAC = 0.2
    whole_dataset = pd.read_csv('../data/wikihowAll.csv')
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
                                tokenize=tokenizer,
                                batch_first=True,
                                include_lengths=True,
                                use_vocab=True)

    fields = [
        ('headline', headline_field),
        ('title', None),
        ('text', article_field)
    ]

    train_set, test_set = data.TabularDataset.splits(path='../data',
                                                     format='csv',
                                                     train='train.csv',
                                                     validation='test.csv',
                                                     fields=fields,
                                                     skip_header=True)
    
    article_field.build_vocab(train_set, test_set, max_size=config.vocab_size)
    headline_field.build_vocab(train_set, test_set, max_size=config.vocab_size)

    train_bch, test_bch = data.BucketIterator.splits(datasets=(train_set, test_set), # specify train and validation Tabulardataset
                                            batch_sizes=(config.batch_size,config_batch_size),  # batch size of train and validation
                                            device=device, # -1 mean cpu and 0 or None mean gpu
                                            repeat=False)

    # construct the model first
    model = Model()
    optimizer = None
    training_batches = BatchGenerator(train_bch, 'text', 'headline')
    for ((article_t, article_lens_t), (headline_t, headline_lens_t)) in training_batches:
        # article_t :: (B x N) where N is max sequence length of batch
        encoder_outputs, encoder_feature, encoder_hidden = \
          self.model.encoder(article_t, article_lens_t)

        c_t_1 = Variable(torch.zeros((config.batch_size, 2 * config.hidden_dim)))
        s_t_1 = model.reduce_state(encoder_hidden)
        max_article_len = torch.max(article_lens_t).item()
        enc_padding_mask = np.zeros((config.batch_size, max_article_len), dtype=np.float32)
        # enc_padding_mask is 1 for all non padding words in article_t
        for b in config.batch_size:
            for j in xrange(article_lens_t[b]):
                enc_padding_mask[b][j] = 1
        
        for t in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = headline_t[:, t] # get y_(t-1) for all members of batch
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
              self.model.decoder(y_t_1, s_t_1, encoder_outputs,
                                 encoder_feature, enc_padding_mask, c_t_1, extra_zeros,
                                 enc_batch_extend_vocab,coverage, di)

            target = target_batch[:, t] # targets for all examples at time t
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

    
         
         
         
         

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





