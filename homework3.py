# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# %%
# load data
# 使用 try-except 来处理可能的 FileNotFoundError
try:
    df = pd.read_csv('data/household_power_consumption.txt', sep = ";", low_memory=False)
except FileNotFoundError:
    print("FileNotFoundError: Please make sure 'household_power_consumption.txt' is in a 'data' subfolder.")
    # 如果文件不存在，可以添加代码从网上下载，或直接退出
    # For this example, we'll assume the file exists.
    exit()
    
df.head()

# %%
# check the data
# 将所有可能包含'?'的列转换为数值型，无法转换的设为NaN
# 这是因为原始数据中的缺失值是用 '?' 表示的
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col].str.replace('?', 'nan'))
        except:
            pass
df.info()

# %%
# 创建 datetime 列并设置为索引，方便后续处理
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'], dayfirst=True)
df.drop(['Date', 'Time'], axis = 1, inplace = True)
df.set_index('datetime', inplace=True)

# 处理缺失值
# .dropna() 会删除任何包含 NaN 值的行
df.dropna(inplace = True)

# %%
print("Start Date: ", df.index.min())
print("End Date: ", df.index.max())

# %%
# 为了简化问题，我们只使用 'Global_active_power' 进行单变量预测
power_df = df[['Global_active_power']]

# split training and test sets
# a chronological split is crucial for time series data
train_size = int(len(power_df) * 0.8)
train, test = power_df.iloc[0:train_size], power_df.iloc[train_size:len(power_df)]

print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")

# %%
# data normalization
# Fit the scaler ONLY on the training data to prevent data leakage
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# %%
# split X and y
# We need a function to create sequences of data
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQUENCE_LENGTH = 60 # Use 60 minutes of data to predict the next minute

X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH)

print(f"Shape of X_train: {X_train.shape}") # Should be (num_samples, seq_length, num_features)
print(f"Shape of y_train: {y_train.shape}")

# %%
# creat dataloaders
# Convert numpy arrays to torch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# build a LSTM model
class PowerLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # LSTM layer
        # batch_first=True makes the input/output tensors have shape (batch_size, seq_length, features)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # Fully connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # The LSTM returns the output and the final hidden & cell states
        # We only need the output of the last time step
        lstm_out, _ = self.lstm(input_seq)
        # Slicing `lstm_out[:, -1, :]` gets the last time step's output for each sequence in the batch
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PowerLSTM().to(device)
criterion = nn.MSELoss() # Mean Squared Error is suitable for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# %%
# train the model
NUM_EPOCHS = 5 # A small number for demonstration purposes
for epoch in range(NUM_EPOCHS):
    model.train()
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(seq)
        
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} Loss: {loss.item():.4f}')

print("Training finished.")

# %%
# evaluate the model on the test set
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for seq, labels in test_loader:
        seq, labels = seq.to(device), labels.to(device)
        y_pred = model(seq)
        # Move predictions and labels to CPU and convert to numpy for inverse scaling and plotting
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# Inverse transform the predictions and actuals to their original scale
predictions_unscaled = scaler.inverse_transform(np.array(predictions))
actuals_unscaled = scaler.inverse_transform(np.array(actuals))

# Calculate Mean Squared Error on the unscaled data
mse = np.mean((predictions_unscaled - actuals_unscaled)**2)
print(f"Test Set MSE (unscaled): {mse:.4f}")

# %%
# plotting the predictions against the ground truth
# To make the plot readable, we'll only plot a subset of the test data
plot_range = 300 

plt.figure(figsize=(15, 6))
plt.plot(actuals_unscaled[:plot_range], label='Actual Power Consumption')
plt.plot(predictions_unscaled[:plot_range], label='Predicted Power Consumption', linestyle='--')
plt.title('Household Power Consumption Prediction')
plt.xlabel('Time (minutes)')
plt.ylabel('Global Active Power (kilowatt)')
plt.legend()
plt.grid(True)
plt.show()
