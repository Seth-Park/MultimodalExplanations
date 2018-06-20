import os
import sys
import numpy as np

def reverse(dict):
    rev_dict = {}
    for k, v in dict.items():
        rev_dict[v] = k
    return rev_dict

def to_str(type, idxs, cont, r_adict, r_exp_vdict):
    if type == 'a':
        if idxs in r_adict:
            return r_adict[idxs]
        else:
            return '<UNK>'
    elif type == 'exp':
        words = []
        for idx in idxs:
            if idx == 0:
                break
            words.append(r_exp_vdict[idx])

    return ' '.join(words)

def batch_to_str(type, batch_idx, batch_cont, r_adict, r_exp_vdict):

    converted = []
    for idxs, cont in zip(batch_idx, batch_cont):
        converted.append(to_str(type, idxs, cont, r_adict, r_exp_vdict))
    return converted

def save_att_map(qid_list, att_maps, save_path):
    for qid, att_map in zip(qid_list, att_maps):
        path = os.path.join(save_path, qid)
        squeezed_map = np.squeeze(att_map)
        np.save(path, squeezed_map)
