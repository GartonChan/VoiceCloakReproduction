from CVAE import CVAE
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
import os
import random
import soundfile as sf
from pystoi import stoi

# ------------ Initial Setting -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir="pretrained_models/spkrec-ecapa-voxceleb", 
                                           run_opts={"device":"cuda"})
classifier.device = device
classifier = classifier.to(device)
D = 192
cvae = CVAE().to(device)
cvae.load_state_dict(torch.load("./cvae_weights.pth",
                                map_location=torch.device('cpu')))

# ------------- Get RIRs and normalize ----------------
RIR_path = os.getcwd() + "/RIR.flac"
RIR_t, sample_rate = torchaudio.load(RIR_path)
RIR_t = RIR_t.to(device)
RIR_t = F.normalize(RIR_t, p=2, dim=1)
RIR_t.requires_grad_(False)


# ------------------ Marginal Triplet Optimizer -----------------------
# -------------- Adversarial Perturbation Construction ----------------
def convolution_injector(Xs, delta):
    return torchaudio.functional.convolve(Xs, delta)

def perturb_loss(delta):
    loss = torch.norm((delta - RIR_t).reshape(-1), p=2)
    return loss

def cosDis(x1, x2):
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    x1 = x1.reshape((1, D))
    x2 = x2.reshape((1, D))
    ret = 1 - torch.sum(cos_similarity(x1, x2), dim=0)
    return ret

# ------------------ Marginal Triplet Optimizer -----------------------
def triplet_loss(anchor, positive, negative, k1, k2):
    triplet_loss = max((cosDis(anchor, positive) - k1), 0)+ max((k2 - cosDis(anchor, negative)), 0)
    return triplet_loss

def save_result(file_path, delta, Xs):
    delta = delta.cpu()
    Xs = Xs.cpu()
    Xs_1 = torchaudio.functional.convolve(Xs, delta).detach().numpy()
    # store Xs_1 into a wav file
    saving_Xs_1 = torch.Tensor(Xs_1)
    torchaudio.save(file_path, saving_Xs_1, sample_rate)
    return Xs_1

def identify_similarity(embd1, embd2):
    similarity = torch.sum(torch.cosine_similarity(
        embd1.reshape(-1), embd2.reshape(-1), dim=0))
    return similarity


# ----------------- hyperparameter setting -----------------
alpha = 0.133
k1 = 0.2
k2 = 0.8
steps = 250
eta = 0.046
# ----------------------------------------------------------


class TripletOptimizer(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self):
        adversarial_voice = convolution_injector(Xs, self.delta)
        return adversarial_voice

def optimizer(delta, Xs, steps):
    print("\nworking on: " + voice_path)
    t_opt = TripletOptimizer(delta)
    opti = torch.optim.SGD([t_opt.delta], eta)
    for i in range(steps):
        opti.zero_grad()
        adversarial_voice = t_opt.forward()
        adversarial_voice.to(device)
        adversarial_embedding = classifier.encode_batch(
            adversarial_voice).reshape((-1)).to(device)
        loss = triplet_loss(adversarial_embedding, target_embedding,
                            original_embedding, k1, k2) + alpha * perturb_loss(t_opt.delta)
        loss.backward(retain_graph=True)
        opti.step()
        
    save_result(after_RIR_path, RIR_t, Xs)
    delta = torch.Tensor(t_opt.delta)
    save_result(after_optimizer_path, delta, Xs)

    origin_target_similarity = identify_similarity(original_embedding, target_embedding)
    advers_target_similarity = identify_similarity(adversarial_embedding, target_embedding)
    advers_origin_similarity = identify_similarity(adversarial_embedding, original_embedding)

    with open('selfSampling_results.txt', 'a') as f:
        print("working on:", voice_path, file = f)
        print("similarity between Anchor and Positive(Target): ", advers_target_similarity.data, file=f)
        print("similarity between Anchor and Negative(Origin): ", advers_origin_similarity.data, file=f)
        print("similarity between origin and target: ", origin_target_similarity.data, file=f)

    print("similarity between Anchor and Positive(Target): ", advers_target_similarity.data)
    print("similarity between Anchor and Negative(Origin): ", advers_origin_similarity.data)
    print("similarity between origin and target: ", origin_target_similarity.data)

path = os.getcwd() + "/test-clean/"
folder_path = []
file_name = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.flac'):
            folder_path.append(root + "/")
            file_name.append(file)

for i in range(len(file_name)):
    voice_path = folder_path[i] + file_name[i]
    after_RIR_path = os.getcwd() + "afterRIR-text-clean/afterRIR_" + file_name[i]
    after_optimizer_path = os.getcwd() + "processed-test-clean/processed_" + file_name[i]

    # --------------- Extact the original embedding -------------
    Xs, fs = torchaudio.load(voice_path)
    Xs = torch.Tensor(Xs).to(device)
    original_embedding = classifier.encode_batch(Xs).to(device)
    # -----------------------------------------------------------

    # ----------------- Pseudo Target Sampler ------------------
    y_t = [0 for i in range(D)]
    rand_num = random.randint(0, D-1)
    y_t[rand_num] = 1  # generate one-hot label as target label
    y_t = torch.Tensor(y_t).unsqueeze(0)
    y_t = y_t.to(device)

    target_embedding = cvae.sampler(y_t)  # target embedding
    target_embedding = target_embedding.to(device)
    target_embedding = target_embedding.reshape((1, D))
    positive = target_embedding.to(device)
    negative = original_embedding.to(device)
    RIR_h, sample_rate = torchaudio.load(RIR_path)
    RIR_h = RIR_h.to(device)
    RIR_h = F.normalize(RIR_h, p=2, dim=1)
    RIR_h.requires_grad_(False)
    delta = torch.Tensor(RIR_h).to(device)
    delta.requires_grad_(True)

    optimizer(delta, Xs, steps)
    original, fs = sf.read(voice_path)
    processed, fs = sf.read(after_optimizer_path)
    processed = processed[0:len(original)]
    d1 = stoi(original, processed, fs, extended=False)
    print("stoi: ", d1)
    with open('selfSampling_results.txt', 'a') as f:
        print("stoi:", d1, file=f)
        print("", file=f)