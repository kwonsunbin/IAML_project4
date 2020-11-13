import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import PianoDataset, save_pianoroll_to_midi
from torch.utils.data import DataLoader
import pdb
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torchsummary import summary

writer = SummaryWriter()

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'model.pth'
model_save_dir = './model_save' 
if not os.path.exists(model_save_dir)
    os.mkdir(model_save_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths & team number
audio_dir = 'audio'
sample_dir = 'sample'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
team = 0

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
n_epochs = 300
batch_size = 32
num_samples = 3
hidden_dim = 256


# TODO: Build your loss here
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 256*128), reduction='sum')
    KLD = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# TODO : Build your model here
class YourModel(nn.Module):
    def __init__(self, hidden_dim):
        super(YourModel, self).__init__()

        # Encoder part
        self.fc_e1 = nn.Linear(256*128, 256*32)
        self.fc_e2 = nn.Linear(256*32, 256*4)
        self.fc_e31 = nn.Linear(256*4, hidden_dim)
        self.fc_e32 = nn.Linear(256*4, hidden_dim)

        # Decoder part
        self.fc4 = nn.Linear(hidden_dim, 256*4)
        self.fc5 = nn.Linear(256*4, 256*32)
        self.fc6 = nn.Linear(256*32, 256*128)

    def encoder(self, x):
        h = F.relu(self.fc_e1(x))
        h = F.relu(self.fc_e2(h))
        mu = self.fc_e31(h)
        log_var = self.fc_e32(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = torch.sigmoid(self.fc6(h))
        return h

    def forward(self, x):
        x = x.view(-1, 256*128)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var


# # TODO : Modify scaling and refining method if you want
def generate_samples(model, filename, num_samples, hidden_dim):
    # Notice model in evaluation mode
    # This is so all layers run in inference mode.
    random_seeds = torch.randn(num_samples, hidden_dim).to(device)
    samples = model.decoder(random_seeds)
    samples = samples.view(-1, 256, 128)
    # TODO : Modify scaling and refining method if you want
    samples = samples * 127
    # samples dtype should be integer and sample values should be in 0...127
    samples = samples.cpu().numpy().astype('int32')
    samples[samples < 0] = 0
    samples[samples > 127] = 127
    for i in range(samples.shape[0]):
        save_pianoroll_to_midi(samples[i], filename + '%d.midi' % i)

dummy_model = YourModel(hidden_dim=hidden_dim)
dummy_input_shape = (batch_size, 256, 128)
dummy = torch.rand(dummy_input_shape)
writer.add_graph(dummy_model, dummy)
summary(dummy_model, dummy_input_shape)

if not is_test_mode:

    # Load Dataset and Dataloader
    dataset = PianoDataset(audio_dir=audio_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = YourModel(hidden_dim=hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=3,)

    for epoch in range(n_epochs):

        model.train()
        train_loss = 0

        start = time.time()
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(random.randint(1,50))
        lr = optimizer.param_groups[0]['lr']

        if lr != lr_before :
            print(f'lr changed! : {lr_before} -> {lr}')

        for idx, pianoroll in enumerate(data_loader):

            optimizer.zero_grad()

            # TODO : Modify scaling method if you want
            pr = (pianoroll.float() / 127.).to(device)

            recon_pr, mu, log_var = model(pr)
            loss = loss_function(recon_pr, pr, mu, log_var)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        writer.add_scalars('lr / train_loss ', {'lr':lr, 'train_loss' : train_loss}, epoch)
        
        print(f"elapsed time for epoch{epoch} : {time.time()-start}")
        print("==== Epoch: %d, Train Loss: %.2f" % (epoch, train_loss / (batch_size*len(data_loader))))

        if epoch % 5 == 0 :
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch{epoch}_model_state_dict.pt"))
    
    torch.save(model,restore_path)
        


# TODO: sampling
elif is_test_mode:
    # restore model
    model = torch.load(restore_path).to(device)
    print('==== Model restored : %s' % restore_path)
    model.eval()
    generate_samples(model, os.path.join(sample_dir, "test_%02d_" % team), num_samples, hidden_dim)
