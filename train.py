import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import string
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

# Load names of males and females, and concatenate them.
with open("D:/Data/names-en/male.txt", 'r') as f:
    names = f.read().splitlines()
    
with open("D:/Data/names-en/female.txt", 'r') as f:
    names += f.read().splitlines()

# Convert all names to lowercase for consistency
names = [name.lower() for name in names if name.strip() != '']

print(f"Total names loaded: {len(names)}")


# Define special tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

# Create a sorted list of unique characters
all_chars = sorted(list(set(''.join(names))))
chars = [PAD_TOKEN, START_TOKEN, END_TOKEN] + all_chars

# Create mappings from characters to indices and vice versa
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary Size: {vocab_size}")
print(f"Characters: {chars}")


# Encode names
encoded_names = []
for name in names:
    encoded = [char_to_idx[START_TOKEN]] + [char_to_idx[c] for c in name] + [char_to_idx[END_TOKEN]]
    encoded_names.append(encoded)

# Determine the maximum name length
max_length = max(len(name) for name in encoded_names)
print(f"Maximum name length (including start and end tokens): {max_length}")


# Pad sequences
padded_names = []
for name in encoded_names:
    padded = name + [char_to_idx[PAD_TOKEN]] * (max_length - len(name))
    padded_names.append(padded)

padded_names = np.array(padded_names)
print(f"Padded Names Shape: {padded_names.shape}")


class NamesDataset(Dataset):
    def __init__(self, padded_names):
        self.padded_names = torch.tensor(padded_names, dtype=torch.long)
        self.X = self.padded_names[:, :-1]  # Input sequence
        self.y = self.padded_names[:, 1:]   # Target sequence

    def __len__(self):
        return len(self.padded_names)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset and DataLoader
dataset = NamesDataset(padded_names)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Number of batches per epoch: {len(dataloader)}")


class NameGeneratorRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(NameGeneratorRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[PAD_TOKEN])
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return output, hidden

# Hyperparameters
embed_size = 64
hidden_size = 128
num_layers = 2
dropout = 0.3

# Initialize the model
model = NameGeneratorRNN(vocab_size, embed_size, hidden_size, num_layers, dropout)
print(model)


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training parameters
num_epochs = 30
validation_split = 0.2
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        # Reshape outputs and targets to (batch_size * seq_length, vocab_size)
        outputs = outputs.view(-1, vocab_size)
        y_batch = y_batch.view(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        non_pad = y_batch != char_to_idx[PAD_TOKEN]
        correct += (predicted == y_batch).masked_select(non_pad).sum().item()
        total += non_pad.sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs, _ = model(X_val)
            outputs = outputs.view(-1, vocab_size)
            y_val = y_val.view(-1)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            non_pad = y_val != char_to_idx[PAD_TOKEN]
            correct += (predicted == y_val).masked_select(non_pad).sum().item()
            total += non_pad.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs +1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs +1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs +1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs +1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


def generate_name(model, seed, char_to_idx, idx_to_char, max_length, device, top_k=5):
    model.eval()
    name_indices = [char_to_idx[START_TOKEN]] + [char_to_idx.get(c, char_to_idx[PAD_TOKEN]) for c in seed.lower()]
    generated = seed
    predictions_over_time = []

    with torch.no_grad():
        input_seq = torch.tensor(name_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
        hidden = None
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)  # output: (1, seq_len, vocab_size)
            last_logits = output[0, -1, :]  # (vocab_size)
            probs = torch.softmax(last_logits, dim=0).cpu().numpy()

            # Get top k predictions
            top_indices = probs.argsort()[-top_k:][::-1]
            top_chars = [idx_to_char[idx] for idx in top_indices]
            top_probs = [probs[idx] for idx in top_indices]
            predictions_over_time.append((top_chars, top_probs))

            # Choose the character with the highest probability
            next_char_idx = top_indices[0]
            if next_char_idx == char_to_idx[END_TOKEN]:
                break
            next_char = idx_to_char[next_char_idx]
            generated += next_char

            # Append the new character to the input sequence
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)

    return generated.capitalize(), predictions_over_time


# Example seed
seed = "ai"
generated_name, preds_time = generate_name(model, seed, char_to_idx, idx_to_char, max_length, device)
print(f"Generated Name: {generated_name}")


def visualize_predictions(preds_time, seed):
    plt.figure(figsize=(10, len(preds_time) * 2))
    for i, (chars, probs) in enumerate(preds_time):
        plt.bar(chars, probs, color='skyblue')
        current_prefix = seed + ''.join([chars[0] for _, chars in preds_time[:i]])
        plt.title(f"Step {i+1}: After '{current_prefix}'")
        plt.xlabel('Characters')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.show()
        
visualize_predictions(preds_time, seed)
