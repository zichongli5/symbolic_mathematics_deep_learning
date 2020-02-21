import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    def __init__(self, data, opt):
        """
        Data should be a list of event time; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.dict = opt.vocab_dict
        self.src = [[45]+[self.dict[str] for str in inst[0]]+[46] for inst in data]
        self.trg = [[45]+[self.dict[str] for str in inst[1]]+[46] for inst in data]
        i = 0
        while i < len(self.trg):
            if len(self.trg[i])>300:
                self.trg.pop(i)
                self.src.pop(i)
            else:
                i += 1
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.length = len(self.trg)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ each returned element is a list, which represents an event stream """
        return self.src[idx], self.trg[idx]
        
        
        
def pad_src(insts):
    """ Pad the instance to the max seq length in batch """
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_trg(insts):
    """ Pad the instance to the max seq length in batch """
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
#
#    batch_pos = np.array([
#        [pos_i + 1 if w_i != Constants.PAD else 0
#            for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
            
    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    src, trg = list(zip(*insts))
    src = pad_src(src)
    trg = pad_trg(trg)
    return src, trg


def get_dataloader(data, opt):
    ds = EventData(data,opt)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    return dl
