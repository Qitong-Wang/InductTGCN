import torch
import numpy as np

def get_pretrain_emb(word_id_map,emb_size=300):
    emb =  np.zeros((len(word_id_map), emb_size)).astype(np.float32)
    pretrain_vector_map = {}

    with open("./hg2vec.txt") as f:
        f.readline()  # skip first line because it does not contains a vector
        for line in f:
            line = line.split()
            word, vals = line[0], list(map(float, line[1:]))
            # if number of vals is different from nb_dims, bad vector, drop it

            pretrain_vector_map[word] = np.array(vals)
    for key, value in word_id_map.items():
        if key in pretrain_vector_map.keys():
            emb[value,:] = pretrain_vector_map[key]
    emb = torch.FloatTensor(emb)
    return emb