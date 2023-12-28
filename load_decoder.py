from CVAE import CVAE
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from deep_speaker.audio import read_mfcc
# from deep_speaker.batcher import sample_from_mfcc
# from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
# from deep_speaker.conv_models import DeepSpeakerModel
# from deep_speaker.test import batch_cosine_similarity

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    # print(mul.shape)
    s = np.sum(mul, axis=1)
    import math
    l1 = np.sum(np.multiply(x1, x1), axis=1)
    l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s / (math.sqrt(l1) * math.sqrt(l2))


D = 192
if __name__ == '__main__':
    cvae = CVAE()
    cvae.load_state_dict(torch.load("./cvae_weights.pth", map_location=torch.device('cpu')))
    y_t = [0 for i in range(D)]
    y_t[10] = 1
    y_t = torch.Tensor(y_t).unsqueeze(0)  
    # ToDo: think about how to use sampler?
    rec_embedding_1 = cvae.sampler(y_t).reshape((1, D))
    rec_embedding_2 = cvae.sampler(y_t).reshape((1, D))
    cos_dis = nn.CosineSimilarity(dim=0, eps=1e-6)
    # cos_similarity = torch.cosine_similarity(rec_embedding_1, rec_embedding_2, dim=1)
    cos_sim = cos_dis(rec_embedding_1, rec_embedding_2)
    # print(cvae)
    # print(rec_embedding_1)
    # print(cos_sim)
    # print(torch.sum(cos_sim))
    rec_embedding_1 = rec_embedding_1.detach().numpy()
    rec_embedding_2 = rec_embedding_2.detach().numpy()
    # print(rec_embedding_1.shape)
    # print(rec_embedding_2.shape)
    similarity = batch_cosine_similarity(rec_embedding_1, rec_embedding_1)
    print("the same embeddings:", similarity) 
    similarity = batch_cosine_similarity(rec_embedding_1, rec_embedding_2)
    print("embeddings with same label:", similarity) 
    
    rec_embedding_2 = cvae.sampler(y_t).reshape((1, D))
    rec_embedding_2 = rec_embedding_2.detach().numpy()
    similarity = batch_cosine_similarity(rec_embedding_1, rec_embedding_2)
    print("embeddings with same label:", similarity) 
    
    y_t = [0 for i in range(D)]
    y_t[D-2] = 1
    y_t = torch.Tensor(y_t).unsqueeze(0)  
    rec_embedding_2 = cvae.sampler(y_t).reshape((1, D))
    rec_embedding_2 = rec_embedding_2.detach().numpy()
    similarity = batch_cosine_similarity(rec_embedding_1, rec_embedding_2)
    print("embeddings with diff label:", similarity) 
    similarity = batch_cosine_similarity(rec_embedding_2, rec_embedding_2)
    print("the same embedding:", similarity) 
    
    
    # r_e1 = F.normalize(rec_embedding_1, 2, dim=3).squeeze(0).squeeze(0).squeeze(0)
    # r_e2 = F.normalize(rec_embedding_2, 2, dim=3).squeeze(0).squeeze(0).squeeze(0)
    # similarity = torch.dot(r_e1, r_e2)
    # print(similarity)
    