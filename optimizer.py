from CVAE import CVAE
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
import os

# ------------ Initial Setting -----------------------
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
classifier.device = device
classifier = classifier.to(device)
# print("type(classifier): ", type(classifier))
D=192
# ------------- Get RIRs and normalize ----------------
pwd = os.getcwd()
RIR_path = pwd+"/xz_rir_2.wav"
RIR_h, sample_rate = torchaudio.load(RIR_path)
RIR_h = RIR_h.to(device)
RIR_h = F.normalize(RIR_h, p=2, dim=1)
RIR_h.requires_grad_(False)
# RIR_h_np = RIR_h.detach().numpy()
# print(RIR_h_np[0])
# plt.plot(RIR_h_np[0])
# plt.show()
# --------------- Extact the original embedding -------------
# voice_path = "./original_voice.flac"
# voice_path = "./40744/5639-40744-0000.flac"
voice_path = "./media1.wav"
Xs, fs =torchaudio.load(voice_path)
Xs = torch.Tensor(Xs).to(device)
# print("Xs.type: ", Xs.type())
original_embedding = classifier.encode_batch(Xs).to(device)
# -----------------------------------------------------------

# ----------------- Pseudo Target Sampler ------------------
import random
y_t = [0 for i in range(D)]
rand_num = random.randint(0, D-1)
y_t[rand_num] = 1  # generate one-hot label as target label
y_t = torch.Tensor(y_t).unsqueeze(0)
y_t = y_t.to(device)
cvae = CVAE().to(device)
cvae.load_state_dict(torch.load("./cvae_weights.pth", 
                        map_location=torch.device('cpu')))
target_embedding = cvae.sampler(y_t)  # target embedding
target_embedding = target_embedding.to(device)
target_embedding = target_embedding.reshape((1, D))
# ----------------------------------------------------------

# ------------------ Marginal Triplet Optimizer -----------------------
# -------------- Adversarial Perturbation Construction ----------------
def convolution_injector(Xs, delta):
    return torchaudio.functional.convolve(Xs, delta)

def extractor(Xs, delta):
    Xs_1 = convolution_injector(Xs, delta)
    # store Xs_1 into a wav file
    Xs_1_path = "./Xs_1.wav"
    torchaudio.save(Xs_1_path, Xs_1, sample_rate)
    # extract the adversarial embedding from its wav file.
    Xs_1 = Xs_1.to(device)
    adversarial_embedding = classifier.encode_batch(Xs_1, sample_rate).to(device)
    return adversarial_embedding

def perturb_loss(delta):
    loss = torch.norm((delta - RIR_h).reshape(-1), p=2)
    return loss

def cosDis(x1, x2):
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    x1 = x1.reshape((1, D))
    x2 = x2.reshape((1, D))
    ret = 1 - torch.sum(cos_similarity(x1, x2), dim=0)
    # print("cosDis = %.8f" % ret)
    return ret

# ------------------ Marginal Triplet Optimizer -----------------------
def triplet_loss(anchor, positive, negative, k1, k2):
    triplet_loss = max((cosDis(anchor, positive) - k1), 0)    # k1 = 0.2  ->: <= 0.2
    + max((k2 - cosDis(anchor, negative)), 0)                 # k2 = 0.8  ->: >= 0.8
    return triplet_loss

def save_result(file_path, delta):
    delta = delta.cpu()
    Xs_1 = torchaudio.functional.convolve(Xs, delta).detach().numpy()
    # store Xs_1 into a wav file
    saving_Xs_1 = torch.Tensor(Xs_1)
    torchaudio.save(file_path, saving_Xs_1, sample_rate)
    return Xs_1

def identify_similarity(embd1, embd2):
    similarity = torch.sum(torch.cosine_similarity(embd1.reshape(-1), embd2.reshape(-1), dim=0))
    return similarity

# ----------------- hyperparameter setting -----------------
alpha = 5000
positive = target_embedding.to(device)
negative = original_embedding.to(device)
k1 = 0.2
k2 = 0.8
steps = 250
delta = torch.Tensor(RIR_h).to(device)
delta.requires_grad_(True)
eta = 0.006  # is set to 0.001 in the paper
# ----------------------------------------------------------

class TripletOptimizer(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
                
    def forward(self):
        adversarial_voice = convolution_injector(Xs, self.delta)
        return adversarial_voice

t_opt = TripletOptimizer(delta)
opti = torch.optim.SGD([t_opt.delta], eta)

for i in range(steps):
    print("----- Interation: {} -----".format(i))
    opti.zero_grad()
    adversarial_voice = t_opt.forward()
    adversarial_voice.to(device)
    adversarial_embedding = classifier.encode_batch(adversarial_voice).reshape((-1)).to(device)
    loss = triplet_loss(adversarial_embedding, target_embedding, 
                        original_embedding, k1, k2) + alpha * perturb_loss(t_opt.delta) 
    # print("p_loss = ", alpha * perturb_loss(t_opt.delta))
    # print("t_loss = ", triplet_loss(adversarial_embedding, target_embedding, 
    #                     original_embedding, k1, k2))
    print("CosDistance between Anchor and Positive = ", cosDis(adversarial_embedding, target_embedding))
    print("CosDistance between Anchor and Negative = ", cosDis(adversarial_embedding, original_embedding))
    loss.backward(retain_graph=True)
    # print("grad of delta = ", delta.grad)
    if torch.sum(delta.grad) == 0:
        break
    opti.step()
# --------------------------------------------------------------
Xs = Xs.cpu()
torchaudio.save("origin_voice.wav", Xs, sample_rate)
save_result("./before_optimizer.wav", RIR_h)
delta = torch.Tensor(t_opt.delta)
save_result("./after_optimizer.wav", delta)

# voiceCloak input: original voice, RIR, y_t(target_label)
# voiceCloak output: adversarial voice (with diff embedding from original voice)
origin_target_similarity =  identify_similarity(original_embedding, target_embedding)
advers_target_similarity = identify_similarity(adversarial_embedding, target_embedding)
advers_origin_similarity = identify_similarity(adversarial_embedding, original_embedding)

print("similarity between Anchor and Positive(Target): ", advers_target_similarity.data)
print("similarity between Anchor and Negative(Origin): ", advers_origin_similarity.data)
print("similarity between origin and target: ", origin_target_similarity.data)
