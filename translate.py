''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator
from Dataset import get_dataloader
import numpy as np


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        46,
        46,
        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 

def prepare_dataloader(opt):
    """ loading data and preparing dataloader """

    def load_data(name):
        data = np.load(name+'.npy',allow_pickle=True)
        max_len =  max([len(elem[1]) for elem in data])
        return data, max_len

    print('[Info] Loading translate data...')
    translate_data, max_len = load_data(opt.data)

    dataloader = get_dataloader(translate_data, opt)
    return dataloader

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
#    parser.add_argument('-data_pkl', required=True,
#                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-data', required=True)
#    parser.add_argument('-seq', required = True, type = str)
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.batch_size = 1
#    print(opt.seq)
#    data = pickle.load(open(opt.data_pkl, 'rb'))
#    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
#    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
#    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
#    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
#    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    opt.vocab_dict = {'0':42, '1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'-1':10,'-2':11,'-3':12,'-4':13,'-5':14,'-6':15,'-7':16,'-8':17,'-9':18,')': 19, 'tan': 20, 'cos': 21, 'sin': 22, '**': 23, '*': 24, '/': 25, '+': 26, '-': 27, '(': 28, 'exp':29, 'log':30, 'sqrt':31, 'asin':32, 'acos':33, 'atan':34, 'sinh':35, 'cosh':36, 'tanh':37, 'asinh':38, 'acosh':39, 'atanh':40, 'x': 41, 'e': 43, ' ':0, 'pi':44,'s':45, '\s':46}
#    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    opt.device = device
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=0,
        trg_pad_idx=0,
        trg_bos_idx=45,
        trg_eos_idx=46).to(device)

#    unk_idx = SRC.vocab.stoi[SRC.unk_token]
#    with open(opt.output, 'w') as f:
#        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
#    src_seq = [opt.vocab_dict[str] for str in opt.seq.split()]
#    pred_seq =  translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
#    print(pred_seq)
#    pred_line = ' '.join(opt.vocab_dict[idx] for idx in pred_seq)
#    print(pred_line)
    dataloader = prepare_dataloader(opt)
    correct_n = 0
    total_n = 0
    for batch in tqdm(dataloader, mininterval=2, desc='  - (Test)', leave=False):
        #print(' '.join(example.src))
#        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
#        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
#        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
#        pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
#        #print(pred_line)
#        f.write(pred_line.strip() + '\n')
#            #print(pred_line)
#        f.write(pred_line.strip() + '\n')
        src_seq, trg_seq = map(lambda x: x.to(opt.device), batch)
        pred_seq = translator.translate_sentence(src_seq.long())
#        print('trg',trg_seq)
#        print('pred',pred_seq)
        trg_seq = trg_seq.long()
        for i in range(opt.beam_size):
            if trg_seq.size(1) == torch.tensor([pred_seq[i]]).size(1):
                if trg_seq.cpu().numpy().tolist() == [pred_seq[i]]:
                    correct_n += 1
        total_n += 1
    print(correct_n/total_n)

#    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
