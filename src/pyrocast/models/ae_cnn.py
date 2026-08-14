import utils.data.dataprep  as dp
from utils import metrics
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, num_channels_in):
        super().__init__()
        '''
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=12, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 6, 5)
        #self.conv3 = nn.Conv2d(6, 16, 5)
        #self.conv4 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 1)
        self.fc3 = nn.Linear(84, 2)
        
        '''        
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=6, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 16)
        #self.fc3 = nn.Linear(16, 2)
        
    def forward(self, x):
        #print("input",x.shape)
        #print(torch.sum(torch.isnan(x)))
        x, indx1 = self.pool(F.relu(self.conv1(x)))
        #print("Conv1: ", x.shape)
        x, indx2 = self.pool(F.relu(self.conv2(x)))
        #print("Conv2: ", x.shape)
        x, indx3 = self.pool(F.relu(self.conv3(x)))
        #print("Conv3: ", x.shape)
        x, indx4 = self.pool(F.relu(self.conv4(x)))
        #print("Conv4: ", x.shape)
        x, indx5 = self.pool(F.relu(self.conv5(x)))
        #print("Conv5: ", x.shape)
        #x, indx6 = self.pool(F.relu(self.conv6(x)))
        x= F.relu(self.conv6(x))
        #print("Conv6: ", x.shape)
        
        x = x.reshape(x.size(0), -1)  # flatten all dimensions except batch
        #print("Flattening: ", x.shape)
        x = F.relu(self.fc1(x))
        #print("fc1: ", x.shape)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #print("fc2: ", x.shape)
        #x = torch.sigmoid(self.fc3(x)) 
        #x = self.fc3(x)
        #x = torch.reshape(x, (x.size(0), 2,8))
       # #print("fc3: ", x.shape)
        
        return x, indx1 , indx2, indx3, indx4, indx5


class Decoder(nn.Module):
    def __init__(self, num_channels_in):
        super().__init__()
        '''
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=12, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 6, 5)
        #self.conv3 = nn.Conv2d(6, 16, 5)
        #self.conv4 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 1)
        self.fc3 = nn.Linear(84, 2)
        
        ''' 
        self.fc2 = nn.Linear(16,128)
        self.fc1 = nn.Linear(128,512)
        
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding=0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding=-2)
        
                
        self.deconv6 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=6)
        self.deconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=3)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=num_channels_in, kernel_size=3, padding=1)
        
        
        
    #def forward(self, x, indx1):
    def forward(self, x, indx1, indx2, indx3, indx4, indx5):    
        #print("input",x.shape)
        #print(torch.sum(torch.isnan(x)))
        
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        
        x = x.reshape( (x.shape[0],512,1,1), -1)  # unflatten all dimensions except batch
        #print(x.shape)
        #
        
        #Encoder : pool(relu(conv(x)))
        #Decoder:  deconv(relu(unpool(x)))
        
        x = F.relu(self.deconv6(x))
        #print("decconv6",x.shape)
        
        
        x = F.relu(self.deconv5(self.unpool1(x, indx5)))
        #print("decconv5",x.shape)
        x = self.unpool2(x, indx4)
        #print(x.shape)
        x = F.relu(self.deconv4(x))
        #print("decconv4",x.shape)
        x = F.relu(self.deconv3(self.unpool1(x, indx3)))
        #print("decconv3",x.shape)
        x = F.relu(self.deconv2(self.unpool1(x, indx2)))
        #print("decconv2",x.shape)
        x = F.relu(self.deconv1(self.unpool1(x, indx1)))
        #print("decconv1",x.shape)
        
    
        #print("Conv4: ", x.shape)    
        #print("Flattening: ", x.shape)
        #print("fc1: ", x.shape)
        
        return x


if __name__ in "__main__":

    encoder = Encoder(num_channels_in)
    encoder = encoder.double()
    decoder = Decoder(num_channels_in+1)
    decoder = decoder.double()
