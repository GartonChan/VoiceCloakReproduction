import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
from data import *

D = 192
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder (nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------------------
        # Down-sample Block 1
        # ----------------------------
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 6), stride=(1,2), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.lRelu1 = nn.LeakyReLU()
        # ----------------------------
        # Down-sample Block 1
        # ----------------------------
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1,2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.lRelu2 = nn.LeakyReLU()
        # ----------------------------
        # Full-connnected Layer
        # ----------------------------
        self.full_conn = nn.Linear(in_features=16*D, out_features=D)
        # ----------------------------
        # Mean & Covariance Vector
        # ----------------------------
        self.mean = nn.Linear(in_features=D, out_features=int(D/2))
        self.cova = nn.Linear(in_features=D, out_features=int(D/2))
        
    def forward(self, x):
        # extract y_t here for generating laten var 'z'
        y_t = torch.Tensor.chunk(x, 2, dim=2)[1].squeeze(1).squeeze(1)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.lRelu1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = self.lRelu2(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], 16*D))
        x = self.full_conn(x)
        mean = self.mean(x)
        log_cova = self.cova(x)
        # print("mean_shape = ", mean.shape)
        # print("cova_shape = ", cova.shape)
        return mean, log_cova, y_t


class Decoder (nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------------------
        # Latent Vector
        # ----------------------------
        self.latent = nn.Linear(in_features=(3*D//2), out_features=D)
        # ----------------------------
        # Full-connnected Layer
        # ----------------------------
        self.full_conn = nn.Linear(in_features=D, out_features=16*D)
        # ----------------------------
        # up-sample Block
        # ----------------------------
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1,2), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.lRelu1 = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=(1, 6), stride=(1,2), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(1)
        self.lRelu2 = nn.LeakyReLU()
    
    def forward(self, x):
        # print("in decoder forward")
        x = self.latent(x)
        # print(x.shape) 
        x = self.full_conn(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], 64, 1, D//4))
        # print(x.shape)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.lRelu1(x)
        # print(x.shape)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.lRelu2(x)
        # print(x.shape)
        return x

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mean, log_cova):
        std = torch.exp(0.5 * log_cova)
        # eplison = torch.randn(size=(mean.shape))
        eplison = torch.randn_like(std)
        z = mean + std * eplison
        return z

    def forward(self, x):
        # print("cvae input_x.shape = ", x.shape)
        mean, log_cova, y_t = self.encoder(x)
        # print("extract y_t.shape = ", y_t.shape)
        z = self.reparameterize(mean, log_cova)
        # print("unconcatenated z.shape = ", z.shape)
        z = torch.cat((z, y_t), 1)
        # print("latent variable z.shape = ", z.shape)
        # z = concatenate(z, yt)
        rec_embedding = self.decoder(z)
        return mean, log_cova, rec_embedding
    
    def sampler(self, y_t):
        # mean = 0, cova = 0
        mean = torch.randn((1, D//2)).to(device)
        cova = torch.randn((1, D//2)).to(device)
        z = self.reparameterize(mean, cova)
        # z = torch.normal(0, 1, (1, D//2))
        
        z = torch.cat((z, y_t), 1)
        # print("sampler input z:\n", z)
        rec_embedding = self.decoder(z)
        return rec_embedding

def loss_fn(recon_x, x, mean, log_var, beta):
    # kl_loss= -0.5* sum(1+ log_var - mean ** 2 - exp(log_var))
    # print("rec_x.shape = ", recon_x)
    # print("x.shape = ", x)
    MSECriterion = nn.MSELoss()
    recon_error = MSECriterion(recon_x, x)
    KLD = torch.sum(-0.5 * (1 + log_var - mean.pow(2) - torch.exp(log_var)))
    # print(MSE)
    # print(KLD)
    loss = recon_error + beta * KLD
    return loss

if __name__ == '__main__':
    # data preprocessing (json -> e_y -> e, y_t)
    training_data = []
    test_data = []
    # with open("./train_set_all.json", "r") as f:
    with open("./train_set.json", "r") as f:
        str_data = f.read()
        training_data = json.loads(str_data)
    with open("./test_set.json", "r") as f:
        str_data = f.read()
        test_data = json.loads(str_data)
    import random
    random.shuffle(training_data)
    random.shuffle(test_data)
    e_y = torch.Tensor(training_data).unsqueeze(0).permute(1, 0, 2, 3)
    e_y_test = torch.Tensor(test_data).unsqueeze(0).permute(1, 0, 2, 3)
    
    # print("e_y.shape = ", e_y.shape)
    D = e_y.shape[3]
    data_size = e_y.shape[0]
    batch_data = data_loader(e_y, 32)
    # print(batch_data.shape)


    # ----------- initial setting -----------
    cvae = CVAE()
    cvae = cvae.to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001)
    beta = 2
    num_epochs = 30
    batchSize = 256
    iteration = num_epochs * data_size // batchSize
    # ----------- End of Initialization ---------
    
    # ----------- Training  -----------
    def train(train_set, test_set, batchSize=64, iteration=1000):
        cvae.train()  # set train mode
        for i in range(iteration):
            epoch = i * batchSize // data_size
            total_loss = 0.00
            optimizer.zero_grad()
            batch_data = data_loader(train_set, batchSize)
            # print("batch_data.shape = ", batch_data.shape)
            embeddings = torch.Tensor.chunk(batch_data,
                    2, dim=2)[1].squeeze(1).squeeze(1)
            # print(embeddings.shape)
            if torch.cuda.is_available():
                batch_data = batch_data.to('cuda')
                embeddings = embeddings.to('cuda')
            # forward propogation
            mean, log_cova, rec_e = cvae(batch_data)
            # print(mean.shape)
            # print(log_cova.shape)
            rec_e = rec_e.reshape((-1, 192))
            # print(rec_e.shape)
            # calculate loss
            loss = loss_fn(rec_e, embeddings, mean, log_cova, beta)
            total_loss += loss.item()
            # print("loss: ", loss)
            
            # performance on test_data
            test_batch = data_loader(test_set, batchSize)
            # print(test_batch.shape)
            test_embeddings = torch.Tensor.chunk(test_batch,
                    2, dim=2)[1].squeeze(1).squeeze(1)
            # print(test_batch.shape)
            # print("test_embeddings.shape", test_embeddings.shape)
            if torch.cuda.is_available():
                test_batch = test_batch.to('cuda')
                test_embeddings = test_embeddings.to('cuda')
            test_m, test_c, test_r_e = cvae(test_batch)
            test_r_e = test_r_e.reshape((-1, D))
            # print(test_m.shape)
            # print(test_c.shape)
            # print(test_r_e.shape)
            
            test_loss = loss_fn(test_r_e, test_embeddings, test_m, test_c, beta)
            
            # backward propogation
            loss.backward()
            optimizer.step()
            
            # log
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / batchSize, test_loss.item()/batchSize))
            # print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / batchSize ))

    train(e_y, e_y_test, batchSize, iteration)
    torch.save(cvae.state_dict(), './cvae_weights.pth')
    