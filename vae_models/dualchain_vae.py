import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow.Flow import *

# define DCVAE
class DCVAE(nn.Module):
    def __init__(self, filters, channels, features, data_type,flow):
        super(DCVAE, self).__init__()
 
        self.filters = filters
        self.channels = channels
        self.features = features
        self.data_type = data_type

        self.is_flow = flow

        if flow == True:
            self.flow = Flow(features, 'radial', 8)
            self.flow_2 = Flow(features, 'radial',8)

        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=filters*2,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.flatten =  nn.Flatten()

        if data_type == 'eeg':
            dense_var = 156
            dense_var_2 = 156
        else:
            dense_var = 236
            dense_var_2 = 240

        self.dense = nn.Linear(dense_var*filters,32)
        self.dense2 = nn.Linear(features,32)
        self.dense3 = nn.Linear(32,dense_var_2*filters)
        self.dense_features = nn.Linear(32,features)
        # decoder

        self.decode_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
        )

        if data_type == 'eeg':
            self.decode_layer2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )
        else:
            self.decode_layer2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,25),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )

        self.output = nn.ConvTranspose2d(in_channels=filters*2,out_channels=1,kernel_size=5,padding=(2,2))

        ## Second Encoder/Decoder

        self.layer1_2 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(in_channels=filters*2,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.flatten_2 =  nn.Flatten()
        self.dense_2 = nn.Linear(dense_var*filters,32)
        self.dense2_2 = nn.Linear(features,32)
        self.dense3_2 = nn.Linear(32,dense_var_2*filters)
        self.dense_features_2 = nn.Linear(32,features)
        # decoder

        self.decode_layer1_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
        )

        if data_type == 'eeg':
            self.decode_layer2_2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )
        else:
            self.decode_layer2_2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,25),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )

        self.output_2 = nn.ConvTranspose2d(in_channels=filters*2,out_channels=1,kernel_size=5,padding=(2,2))

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space

        return sample



    def forward(self, x):
        # encoding
        # print(x.shape)
        original = x
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.dense(x))
        # print(x.shape)

        # get `mu` and `log_var`
        mu = self.dense_features(x)
        log_var = self.dense_features(x) + 1e-8
        # print(mu.shape)
        # print(log_var.shape)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        if self.is_flow == True:
            z , log_det = self.flow(z)
        # print(z.shape)

        # decoding
        x = F.relu(self.dense2(z))
        x = F.relu(self.dense3(x))
        # print(x.shape)
        batch_size = len(x)
        if self.data_type == 'eeg':
            view_var = 39
        else:
            view_var = 60 * batch_size
        x = x.view(-1, self.filters*4, 1, view_var)
        # print(x.shape)

        x = self.decode_layer1(x)
        # print(x.shape)
        x = self.decode_layer2(x)
        # print(x.shape)
        reconstruction = self.output(x)
        # print(reconstruction.shape)
        if self.data_type != 'eeg':
            reconstruction = reconstruction.view(batch_size,1,4,1500)
 

        new_inputs = original - reconstruction
        ## Second encoder decoder
        # print(new_inputs.shape)
        x1 = self.layer1_2(new_inputs)
        x1 = self.layer2_2(x1)
        x1 = self.flatten_2(x1)
        x1 = F.relu(self.dense_2(x1))

        # get `mu` and `log_var`
        mu_2 = self.dense_features_2(x1)
        log_var_2 = self.dense_features_2(x1) + 1e-8
        # get the latent vector through reparameterization
        z_2 = self.reparameterize(mu_2, log_var_2)
        if self.is_flow == True:
            z_2 , log_det_2 = self.flow_2(z_2)

        # decoding
        x1 = F.relu(self.dense2_2(z_2))
        x1 = F.relu(self.dense3_2(x1))
        # print(x1.shape)
        if self.data_type == 'eeg':
            view_var = 39
        else:
            view_var = 60 * len(x)
        x1 = x1.view(-1, self.filters*4, 1, view_var)

        x1 = self.decode_layer1_2(x1)
        x1 = self.decode_layer2_2(x1)
        reconstruction_2 = self.output_2(x1)

        # decoding
        if self.is_flow == True:
            return reconstruction, mu, log_var, new_inputs, reconstruction_2, mu_2, log_var_2,log_det,log_det_2
        else:          
            return reconstruction, mu, log_var, new_inputs, reconstruction_2, mu_2, log_var_2