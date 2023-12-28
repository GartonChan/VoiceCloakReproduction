import numpy as np
from CVAE import CVAE
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

signal, fs =torchaudio.load('./original_voice.flac')
embeddings = classifier.encode_batch(signal)


# def load_model(checkpoint_file):
#     model = DeepSpeakerModel()
#     model.m.load_weights(checkpoint_file)
#     return model

# def extract_embedding_from_file(model, voice_file):
#     mfcc = sample_from_mfcc(read_mfcc(voice_file, SAMPLE_RATE), NUM_FRAMES)
#     predict = model.m.predict(np.expand_dims(mfcc, axis=0))
#     return predict

# ------------- Get RIRs and normalize ----------------
# Normalize the RIR: h <- h/||h|| -> F.normalize(...)
# Initialize the perturation delta <- h
RIR_path = "./rir_female_0.2s.wav"
RIR_h, sample_rate = torchaudio.load(RIR_path)
# print(RIR_h.shape, sample_rate)
RIR_h = F.normalize(RIR_h, p=2, dim=1)
RIR_h.requires_grad_(False)

# --------------- Extact the original embedding -------------
# model_path = "./ResCNN_triplet_training_checkpoint_265.h5"
voice_path = "./original_voice.flac"
# model = load_model(model_path)
# original_embedding = extract_embedding_from_file(model, voice_path) 
# original_embedding = torch.Tensor(original_embedding)
signal, fs =torchaudio.load('./original_voice.flac')
original_embedding = classifier.encode_batch(signal)
# original_embedding.requires_grad_(False)
# -----------------------------------------------------------

D=192
# ----------------- Pseudo Target Sampler ------------------
import random
y_t = [0 for i in range(D)]
rand_num = random.randint(0, D-1)
y_t[rand_num] = 1  # generate one-hot label as target label
y_t = torch.Tensor(y_t).unsqueeze(0)
cvae = CVAE()
cvae.load_state_dict(torch.load("./cvae_weights.pth", 
                        map_location=torch.device('cpu')))
target_embedding = cvae.sampler(y_t)  # target embedding
target_embedding = target_embedding.reshape((1, D))
# ----------------------------------------------------------

# ----------------- hyperparameter setting -----------------
alpha = 5000
k1 = 0.2
k2 = 0.8
steps = 10
# ----------------------------------------------------------


Xs, sample_rate = torchaudio.load(voice_path)

# ------------------ Marginal Triplet Optimizer -----------------------
# -------------- Adversarial Perturbation Construction ----------------
def generate_adversarial_voice(Xs, delta):
    return torchaudio.functional.convolve(Xs, delta)

def extractor(Xs, delta):
    Xs_1 = generate_adversarial_voice(Xs, delta)
    # store Xs_1 into a wav file
    Xs_1_path = "./Xs_1.wav"
    torchaudio.save(Xs_1_path, Xs_1, sample_rate)
    # extract the adversarial embedding from its wav file.
    adversarial_embedding = classifier.encode_batch(Xs_1, sample_rate)
    return adversarial_embedding

# def extract_embedding_from_Xs(model, Xs):
#     Xs_path = "./tmp.wav"
#     torchaudio.save(Xs_path, Xs, sample_rate)
#     # extract the adversarial embedding from its wav file.
#     embedding = extract_embedding_from_file(model, Xs_path)
#     return torch.Tensor(embedding)

def perturb_loss(delta):
    loss = torch.norm((delta - RIR_h).reshape(-1), p=2)
    return loss

def cosDis(x1, x2):
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    x1 = x1.reshape((1, D))
    x2 = x2.reshape((1, D))
    ret = 1 - torch.sum(cos_similarity(x1, x2), dim=0)
    print("cosDis = %.8f" % ret)
    return ret

positive = target_embedding
negative = original_embedding
k1 = 0.2
k2 = 0.8

class TripletLossModule(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, delta):
        adversarial_voice = torchaudio.functional.convolve(Xs, delta)
        anchor = classifier.encode_batch(adversarial_voice)
        dis_1 = cosDis(anchor, positive)
        dis_2 = cosDis(anchor, negative)
        
        loss_1 = torch.max(torch.Tensor([dis_1 - k1, 0]))
        loss_2 = torch.max(torch.Tensor([-dis_2 + k2, 0]))
        loss = loss_1 + loss_2
        
        ctx.save_for_backward(delta)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        delta = ctx.saved_tensors
        return delta

delta = torch.Tensor(RIR_h)# + 0.000001*torch.rand_like(RIR_h)
delta.requires_grad_(True)

tri_loss = TripletLossModule()

opti = torch.optim.SGD([delta], lr=0.01)
for i in range(steps):
    print("Epoch %d: SGD" % i)

    t_loss = tri_loss.apply(delta)
    t_loss.requires_grad_(True)
    p_loss = perturb_loss(delta)
    loss = t_loss + alpha * p_loss

    opti.zero_grad()
    loss.backward(retain_graph=True)
    opti.step()

    grad = delta.grad.clone().detach().numpy()
    # before_lr = opti.param_groups[0]["lr"]
    # opti.param_groups[0]['lr'] = new_lr(last_grad, grad, before_lr)
    # after_lr = opti.param_groups[0]["lr"]
    # last_grad = grad

    print("detla:\t", delta.clone().detach().numpy()[:20])
    print("grad of delta:\t", grad[:20])
    print("norm of grad of delta:\t", np.linalg.norm(grad))
    print("t_loss:\t%.8f" % t_loss.clone().detach().numpy())
    print("alpha * p_loss:\t%.32f" % (alpha*p_loss).clone().detach().numpy()) 
    print("loss:\t%.8f" % loss.clone().detach().numpy())
    # print("lr %.32f -> %.32f" % (before_lr, after_lr))
    print("-" * 15, flush=True)

def save_result(delta):
    Xs_1 = torchaudio.functional.convolve(Xs, delta)
    # store Xs_1 into a wav file
    Xs_1_path = "./Xs_1.wav"
    torchaudio.save(Xs_1_path, Xs_1, sample_rate)

save_result(delta)