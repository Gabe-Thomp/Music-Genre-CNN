#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import librosa
import matplotlib.pyplot as plt
#%matplotlib inline


# In[93]:


'''
DATA_PATH = "/content/drive/Shareddrives/DL_Project/Data"
IMAGES_PATH = "/content/drive/Shareddrives/DL_Project/Data/images_original"
CROPPED_PATH = "/content/drive/Shareddrives/DL_Project/Data/images_cropped"
AUDIO_PATH = "/content/drive/Shareddrives/DL_Project/Data/genres_original"
CREATED_PATH = "/content/drive/Shareddrives/DL_Project/Data/created_images"
'''


# ## My local path

# In[94]:


DATA_PATH = "./Data"
IMAGES_PATH = "./Data/created_images"


# ### 1. Load and Prepare Data
# 
# This should illustrate your code for loading the dataset and the split into training, validation and testing. You can add steps like pre-processing if needed.

# In[95]:


### YOUR CODE HERE
df_three = pd.read_csv(f"{DATA_PATH}/features_3_sec.csv")
df_thirty =  pd.read_csv(f"{DATA_PATH}/features_30_sec.csv")


# https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

# In[96]:


'''
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Function to convert audio files to spectrograms
def audio_to_spectrogram(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each audio file in the input folder
    for audio_file in os.listdir(input_folder):
        # Check if the file is an audio file
        if audio_file.endswith(".wav"):
            # Load audio file
            audio_path = os.path.join(input_folder, audio_file)
            y, sr = librosa.load(audio_path)

            # Generate spectrogram
            plt.figure(figsize=(10, 4))
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
            plt.axis('off')  # Turn off axis labels
            plt.gca().set_position([0, 0, 1, 1])  # Remove surrounding whitespace
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            # Save spectrogram
            spectrogram_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.png')
            plt.savefig(spectrogram_path)
            plt.close()

            print(f"Spectrogram saved: {spectrogram_path}")

# Paths to input and output folders
genre_names = ["hiphop", "metal", "reggae", "blues", "disco", "jazz", "pop", "classical", "rock", "country"]

input_folders = [os.path.join(AUDIO_PATH, genre) for genre in genre_names]
output_folders = [os.path.join(CREATED_PATH, genre) for genre in genre_names]


# Convert audio files to spectrograms for each pair of input and output folders
for input_folder, output_folder in zip(input_folders, output_folders):
    audio_to_spectrogram(input_folder, output_folder)
    '''


# ### Three second audio clips

# In[97]:


df_three


# In[98]:


df_three.info()


# ### Thirty Second Audio Clips

# In[99]:


df_thirty.info()


# In[100]:


df_thirty.describe()


# ### Loading Images as training data

# Images are read into RGBA, where A is transparency. As seen below, all transparency values are 255. We can get rid of them.

# In[101]:


import os
import numpy as np
from PIL import Image

# Define genres
genres = df_three.label.unique()

# Function to load images from directory
def load_images(root_folder, genres):
    images = []
    y = []
    for genre in genres:
      img_temp, y_temp = load_images_from_genre(root_folder, genre)
      images.extend(img_temp)
      y.extend(y_temp)
    return np.array(images), np.array(y)

def load_images_from_genre(root_folder, genre):
    images = []
    folder = root_folder+f"/{genre}"
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))

        # Get rid of 'A' in 'RGBA'
        img = img.convert("RGB")
        if img is not None:
            img = np.array(img)
            images.append(img)
    images = np.stack(images)
    return np.array(images), np.array([genre]*images.shape[0]).reshape(-1, 1)

# Load images that we created

X, y = load_images(IMAGES_PATH, genres)


# In[102]:


plt.imshow(X[np.random.randint(0, X.shape[0])])


# ### One Hot Encode, Normalize, Flatten

# In[103]:


image_dimensions = X.shape[1:3]
image_dimensions


# In[104]:


X = ((X / 255.0)*2.0)-1
X.shape


# Making sure images are normalized to [-1.0, 1.0]

# In[105]:


plt.hist(X[0].flatten())
plt.show()


# ## Train Test Split

# In[106]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
# Use stratify to evenly split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[107]:


# Checking for ~ evenly distributed test, training classes
print(np.unique(y_train, return_counts=True)[1])
print(np.unique(y_test, return_counts=True)[1])


# In[108]:


genres


# 

# # Implementing the CNN

# In[109]:


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# Define custom dataset in order to have a dataset + dataloader
class CustomDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None, target_transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.x_train[idx]
        label = self.y_train[idx]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Converting data to tensors
X_train = torch.from_numpy(X_train).float()
# Change shape of data to be in the format (batch_size, channels, height, width)
X_train = X_train.permute(0,3,1,2)
X_train = X_train.type(torch.float32)

X_test = torch.from_numpy(X_test).float()
# Change shape of data to be in the format (batch_size, channels, height, width)
X_test = X_test.permute(0,3,1,2)
X_test = X_test.type(torch.float32)

# Encoding labels
label_encoder = LabelEncoder()
label_encoder.fit(genres)

BATCH_SIZE = 50

target_transform = Lambda(lambda y: label_encoder.transform(y))

train_dataset = CustomDataset(X_train, y_train, transform=None, target_transform=target_transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomDataset(X_test, y_test, transform=None,  target_transform=target_transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# From tutorial https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# In[110]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class Net(nn.Module):
    def __init__(self, image_input_shape,dropout_p=0.5):
        super().__init__()
        if image_input_shape:
            self.row_length, self.col_length = image_input_shape
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                                kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                                kernel_size=3 , stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, 
                                kernel_size=5, stride=1, padding="same")
        
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, 
                                kernel_size=5, stride=1, padding="same")
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(17856, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
    

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc(x)
        return x


# # Send to device, remove from colab

# In[111]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


# In[112]:


def train(temp_model, criterion_temp, optimizer_temp,epochs=90, val_dataloader=None, train_dataloader=None, verbose = True):
    for epoch in range(epochs):
            temp_model.train()
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer_temp.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze(1)
                outputs = temp_model(inputs)
                loss = criterion_temp(outputs, labels)
                loss.backward()
                optimizer_temp.step()
                train_loss += loss.item()
            if verbose:
                 print(f'Epoch: {epoch+1}, Loss: {train_loss/(i+1.0)}')

            # validation
            temp_model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_losses = []
                i=0
                for i, (inputs, labels) in enumerate(val_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device).squeeze(1)
                    outputs = temp_model(inputs)
                    loss = criterion_temp(outputs, labels)
                    val_loss += loss.item()
                    val_losses.append(loss.item())
                if verbose: 
                     print(f'Epoch: {epoch+1}, Validation Loss: {val_loss/(i+1.0)}')
    return val_losses


# In[113]:


temp_net = Net(image_dimensions, dropout_p=0.1).to(device)


# In[114]:


import torch.optim as optim
criterion = nn.CrossEntropyLoss()


#optimizer = optim.SGD(temp_net.parameters(), lr=0.001, momentum=0.999)
optimizer = optim.Adam(temp_net.parameters(), lr=0.001, weight_decay=0.01)


train(temp_net, criterion, optimizer, epochs=20, val_dataloader=test_dataloader, train_dataloader=train_dataloader)


# In[115]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with torch.no_grad():
    temp_net.eval()
    y_hat = temp_net(X_test.to(device)).cpu()
    y_hat = torch.argmax(y_hat, dim=1)
    #y_test = label_encoder.transform(y_test)
    print(accuracy_score(y_test, y_hat))

    print(classification_report(target_transform(y_test), y_hat.cpu().numpy(), target_names=genres))


# In[116]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score

def cm(y, y_hat, genres=genres):
    cm = confusion_matrix(y, y_hat, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
    disp.plot()
    print(f"Accuracy: {accuracy_score(y,y_hat)}\n Bal Acc: {balanced_accuracy_score(y,y_hat)}\n")
cm(target_transform(y_test), y_hat.cpu().numpy())


# ## Hyperparameter Search

# In[88]:


import optuna
import torch.nn.init as init

class Net_to_opt(nn.Module):
    def __init__(self,dropout_p=0.5, n_fc_layers = 1, hidden_weights = [], kernel3=5, kernel4=5, initializer=False):
        super().__init__()
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                                kernel_size=3, stride=2, padding=1)
        if initializer: init.kaiming_normal_(self.conv1.weight)  # He initialization
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                                kernel_size=3 , stride=2, padding=1)
        if initializer: init.kaiming_normal_(self.conv2.weight)  # He initialization
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, 
                                kernel_size=kernel3, stride=1, padding="same")
        if initializer: init.kaiming_normal_(self.conv3.weight)  # He initialization

        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, 
                                kernel_size=kernel4, stride=1, padding="same")
        if initializer: init.kaiming_normal_(self.conv4.weight)  # He initialization

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc_layer = nn.ModuleList()
        
        self.n_fc_layers = n_fc_layers
        self.hidden_weights = hidden_weights
        self.kernel3 = kernel3
        self.kernel4 = kernel4

        if n_fc_layers == 1:
            self.fc_layer.append(nn.Linear(17856, 10).to(device))
            if initializer: init.xavier_uniform_(self.fc_layer[0].weight)  # Xavier initialization
        elif n_fc_layers>1:
            self.fc_layer.append(nn.Linear(17856, hidden_weights[0]).to(device))
            if initializer: init.xavier_uniform_(self.fc_layer[0].weight)  # Xavier initialization
            for i in range(1, n_fc_layers-1):
                self.fc_layer.append(nn.Linear(hidden_weights[i-1], hidden_weights[i]).to(device))
                if initializer: init.xavier_uniform_(self.fc_layer[0].weight)  # Xavier initialization
            self.fc_layer.append(nn.Linear(hidden_weights[-1], 10).to(device).to(device))
            if initializer: init.xavier_uniform_(self.fc_layer[0].weight)  # Xavier initialization
        
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
    

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        #print(x.shape)
        if self.n_fc_layers ==1:
            x = self.fc_layer[0](x)
        elif self.n_fc_layers>1:
            for layer in self.fc_layer[:-1]:
                x = self.relu(layer(x))
            x = self.fc_layer[-1](x)
        return x


def objective(trial, epochs=15):
    # Hyperparameters
    dropout_p = trial.suggest_float('dropout_p', 0, 0.5)
    kernel3 = trial.suggest_int('kernel1', 3, 5)
    kernel4 = trial.suggest_int('kernel2', 3, 5)
    
    n_fc_layers = trial.suggest_int('n_fc_layers', 1, 3)
    hidden_weights = []
    for i in range(n_fc_layers-1):
        if len(hidden_weights) == 0:
            hidden_weights.append(trial.suggest_int(f'hidden_weights{i}', 100, 1000))
        else:
            hidden_weights.append(trial.suggest_int(f'hidden_weights{i}', 100, hidden_weights[-1]))

    weight_decays = trial.suggest_float('weight_decays', 1e-5, 1e-2, log=True)

    # For criterion
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    initializers = trial.suggest_categorical('initializers', [True, False])

    temp_model = Net_to_opt(dropout_p=dropout_p, n_fc_layers=n_fc_layers, initializer=initializers,
                            hidden_weights=hidden_weights, kernel3=kernel3, kernel4=kernel4)
    temp_model =temp_model.to(device)

    opt_cat = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])


    if opt_cat == 'Adam':
        optimizer_temp = optim.Adam(temp_model.parameters(), lr=learning_rate, weight_decay=weight_decays)
    elif opt_cat == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.999)
        optimizer_temp = optim.SGD(temp_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decays)
    
    criterion_temp = nn.CrossEntropyLoss()
    
    val_losses = train(temp_model, criterion_temp, optimizer_temp, epochs=epochs, 
                       val_dataloader=test_dataloader, train_dataloader=train_dataloader, verbose=False)  
    # Return mean of two lowest val losses
    val_loss_final = np.mean(sorted(val_losses)[:2])
    trial.report(val_loss_final, step=trial.number)
    return val_loss_final


# In[91]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Retrieve best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

trials_df = study.trials_dataframe()
trials_df.to_csv('optuna_results.csv', index=False)


# In[ ]:




