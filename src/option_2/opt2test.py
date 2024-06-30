import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from tensorflow.keras.models import Model, Sequential

train_path = (r"/Users/olivialiau/Documents/gr-WPI-UMASS-TOD-Project/data/train_test_split_data/full_data/full_60_log_test.csv")
# test_path = (r"/Users/olivialiau/Documents/gr-WPI-UMASS-TOD-Project/data/train_test_split_data/full_data/full_60_log_test.csv")
columns_to_drop = ['TOD', 'Age', 'Sex']
df = pd.read_csv(train_path)
# df2 = pd.read_csv(test_path)

x_train = df.drop(columns=columns_to_drop)
# y_train = df['TOD'].values
# x_test = df2.drop(columns=columns_to_drop)  
# y_test = df2['TOD']


window_size = 3
start_index = window_size
end_index = len(df)- window_size

def get_window(dataset, obs, gene):
    i = 0

    df_window = pd.DataFrame()
    col = dataset.iloc[:, gene]
    
    position = window_size + obs
    start = position - window_size
    end = position + window_size + 1
    
    window = col.iloc[start:end]
    
    for value in window:
        df_window[i] = [value]
        i+=1
    return (df_window)


encoding_dim = (1,1)  
original_dim = (7,1)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = Sequential([
           # Flatten(),
            #LSTM(3),
            Dense(1, activation='tanh')
        ])
        self.decoder = Sequential([
            Dense(7, activation='relu')
        ])

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
 

final_df = pd.DataFrame() 
gene_names = x_train.columns.tolist()

def addGene(index, values):
    gene = gene_names[index]
    final_df[gene] = values


for gene in range(235):
    encoded_values = []
    for window in range(len(df)- 2*window_size):  
        tf.keras.backend.clear_session()
        autoencoder = Autoencoder()
        autoencoder.compile(optimizer='adam', loss='mse') 

        x = get_window(x_train, window, gene)
        #y = get_window(x_test, window, gene)
        autoencoder.fit(x, x, epochs=50, batch_size=256, shuffle=True)
        
        encoded_data = autoencoder.encoder(x).numpy()
        encoded_values.append(encoded_data)
    addGene(gene, encoded_values)



final_df['TOD'] = df['TOD'].iloc[start_index:end_index].to_list()
final_df['Age'] = df['Age'].iloc[start_index:end_index].to_list()
final_df['Sex'] = df['Sex'].iloc[start_index:end_index].to_list()

print(final_df)