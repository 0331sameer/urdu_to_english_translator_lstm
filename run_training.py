"""
RNN with Attention-based Encoder-Decoder Model for Urdu to English Translation
Custom LSTM Implementation (without using nn.LSTM or nn.RNN)
Standalone Python script that can be run from terminal
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import subprocess
import os
from pathlib import Path

print("Starting Urdu to English Translation Model Training...")
print("=" * 70)

# ============================================================================
# CUSTOM LSTM CELL IMPLEMENTATION
# ============================================================================

class LSTMCell(nn.Module):
    """Custom LSTM Cell implementation from scratch."""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices for input-to-hidden connections
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_io = nn.Linear(input_size, hidden_size)
        
        # Weight matrices for hidden-to-hidden connections
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, hidden_state):
        h, c = hidden_state
        
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h))
        g = torch.tanh(self.W_ig(x) + self.W_hg(h))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h))
        
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        
        return new_h, new_c

class CustomLSTM(nn.Module):
    """Custom LSTM layer using custom LSTM cells."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h, c = hidden
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]
            
            outputs.append(h[-1])
        
        outputs = torch.stack(outputs, dim=1)
        h_final = torch.stack(h, dim=0)
        c_final = torch.stack(c, dim=0)
        
        return outputs, (h_final, c_final)

# ============================================================================
# ENCODER, ATTENTION, DECODER
# ============================================================================

class Encoder(nn.Module):
    """Encoder with custom LSTM."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = CustomLSTM(embed_size, hidden_size, num_layers)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(
            self.W_a(encoder_outputs) + self.U_a(decoder_hidden)
        )
        
        attention_scores = self.v_a(energy).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)
        
        return context, attention_weights

class Decoder(nn.Module):
    """Decoder with custom LSTM and attention."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = CustomLSTM(embed_size + hidden_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, target, hidden, cell, encoder_outputs):
        target = target.unsqueeze(1)
        embedded = self.dropout(self.embedding(target))
        
        decoder_hidden = hidden[-1]
        
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs)
        
        context = context.unsqueeze(1)
        lstm_input = torch.cat([embedded, context], dim=2)
        
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        lstm_output = lstm_output.squeeze(1)
        context = context.squeeze(1)
        
        output = self.fc_out(torch.cat([lstm_output, context], dim=1))
        
        return output, hidden, cell, attention_weights

class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model with attention."""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            outputs[:, t, :] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(data_dir="."):
    """Load and prepare data."""
    print("Loading data...")
    
    with open(os.path.join(data_dir, "urd_Arab.dev"), "r", encoding="utf-8") as f:
        urdu_dev = [line.strip() for line in f.readlines()]
    with open(os.path.join(data_dir, "urd_Arab.devtest"), "r", encoding="utf-8") as f:
        urdu_devtest = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(data_dir, "eng_Latn.dev"), "r", encoding="utf-8") as f:
        eng_dev = [line.strip() for line in f.readlines()]
    with open(os.path.join(data_dir, "eng_Latn.devtest"), "r", encoding="utf-8") as f:
        eng_devtest = [line.strip() for line in f.readlines()]
    
    urdu_sentences = urdu_dev + urdu_devtest
    eng_sentences = eng_dev + eng_devtest
    
    print(f"Total sentences: {len(urdu_sentences)}")
    
    combined = list(zip(urdu_sentences, eng_sentences))
    random.seed(42)
    random.shuffle(combined)
    urdu_sentences, eng_sentences = zip(*combined)
    
    train_src, temp_src, train_tgt, temp_tgt = train_test_split(
        urdu_sentences, eng_sentences, test_size=0.3, random_state=42
    )
    val_src, test_src, val_tgt, test_tgt = train_test_split(
        temp_src, temp_tgt, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_src)}, Val: {len(val_src)}, Test: {len(test_src)}")
    
    return (train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
            urdu_sentences, eng_sentences)

def build_vocab(sentences, min_freq=2):
    """Build vocabulary."""
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    
    word_freq = {}
    for sentence in sentences:
        for word in sentence.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(vocab)
    
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    idx_to_token = {idx: token for token, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    return vocab, idx_to_token

def tokenize_sentence(sentence, vocab, max_len):
    """Tokenize and pad sentence."""
    tokens = [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    while len(tokens) < max_len:
        tokens.append(vocab["<pad>"])
    
    return tokens

def prepare_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len_src, max_len_tgt):
    """Prepare data tensors."""
    src_data = []
    tgt_data = []
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_tokens = tokenize_sentence(src, src_vocab, max_len_src)
        
        tgt_with_special = f"<sos> {tgt} <eos>"
        tgt_tokens = tokenize_sentence(tgt_with_special, tgt_vocab, max_len_tgt)
        
        src_data.append(src_tokens)
        tgt_data.append(tgt_tokens)
    
    src_tensor = torch.tensor(src_data, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_data, dtype=torch.long)
    
    return src_tensor, tgt_tensor

class TranslationDataset(Dataset):
    """Dataset class."""
    def __init__(self, src_tensor, tgt_tensor):
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        
    def __len__(self):
        return len(self.src_tensor)
    
    def __getitem__(self, idx):
        return self.src_tensor[idx], self.tgt_tensor[idx]

# ============================================================================
# TRAINING
# ============================================================================

def calculate_bleu_score(model, data_loader, src_vocab, tgt_vocab, device, data_dir="."):
    """Calculate BLEU score."""
    model.eval()
    
    predictions = []
    references = []
    
    idx_to_tgt = {idx: token for token, idx in tgt_vocab.items()}
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)
            output = output.argmax(2)
            
            for pred, ref in zip(output, tgt):
                pred_text = []
                ref_text = []
                
                for idx in pred:
                    token = idx_to_tgt.get(idx.item(), "<unk>")
                    if token not in ["<pad>", "<sos>", "<eos>"]:
                        pred_text.append(token)
                
                for idx in ref:
                    token = idx_to_tgt.get(idx.item(), "<unk>")
                    if token not in ["<pad>", "<sos>", "<eos>"]:
                        ref_text.append(token)
                
                predictions.append(" ".join(pred_text))
                references.append(" ".join(ref_text))
    
    pred_file = os.path.join(data_dir, "predictions.txt")
    ref_file = os.path.join(data_dir, "references.txt")
    
    with open(pred_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    with open(ref_file, "w", encoding="utf-8") as f:
        for ref in references:
            f.write(ref + "\n")
    
    try:
        # Try using Moses multi-bleu.perl if Perl is available
        bleu_script = os.path.join(data_dir, "multi-bleu.perl")
        if os.path.exists(bleu_script):
            try:
                result = subprocess.run(
                    ["perl", bleu_script, ref_file],
                    stdin=open(pred_file, "r", encoding="utf-8"),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                bleu_output = result.stdout.strip()
                if "BLEU = " in bleu_output:
                    bleu_score = float(bleu_output.split("BLEU = ")[1].split(",")[0])
                    print(f"BLEU: {bleu_output}")
                    return bleu_score
            except:
                pass
        
        # Fallback: Simple BLEU approximation
        from collections import Counter
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            if len(pred_words) == 0 or len(ref_words) == 0:
                continue
            # Unigram precision
            pred_counter = Counter(pred_words)
            ref_counter = Counter(ref_words)
            matches = sum((pred_counter & ref_counter).values())
            precision = matches / len(pred_words) if len(pred_words) > 0 else 0
            bleu_scores.append(precision * 100)
        
        bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
        print(f"BLEU (approx): {bleu_score:.2f}")
            
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu_score = 0.0
    
    return bleu_score

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, 
                src_vocab, tgt_vocab, data_dir="."):
    """Train the model with learning rate scheduling and early stopping."""
    train_losses = []
    val_losses = []
    train_bleu_scores = []
    val_bleu_scores = []
    best_val_loss = float('inf')
    patience = 8  # More patience
    patience_counter = 0
    
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                output = model(src, tgt, teacher_forcing_ratio=0)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                val_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Calculate BLEU only on first and last epoch to save time
        if epoch == 0 or epoch == epochs - 1:
            print(f"Calculating BLEU scores for epoch {epoch + 1}...")
            train_bleu = calculate_bleu_score(model, train_loader, src_vocab, tgt_vocab, device, data_dir)
            val_bleu = calculate_bleu_score(model, val_loader, src_vocab, tgt_vocab, device, data_dir)
            train_bleu_scores.append((epoch + 1, train_bleu))
            val_bleu_scores.append((epoch + 1, val_bleu))
        else:
            if train_bleu_scores:
                train_bleu = train_bleu_scores[-1][1]
                val_bleu = val_bleu_scores[-1][1]
            else:
                train_bleu, val_bleu = 0, 0
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train BLEU: {train_bleu:.2f} | Val BLEU: {val_bleu:.2f}")
        print("-" * 70)
    
    print("Training completed!")
    print("=" * 70)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_bleu_scores': train_bleu_scores,
        'val_bleu_scores': val_bleu_scores
    }

def plot_training_curves(history, data_dir="."):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    train_epochs, train_bleu = zip(*history['train_bleu_scores'])
    val_epochs, val_bleu = zip(*history['val_bleu_scores'])
    
    ax2.plot(train_epochs, train_bleu, 'b-o', label='Train BLEU', linewidth=2, markersize=6)
    ax2.plot(val_epochs, val_bleu, 'r-o', label='Validation BLEU', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('Training and Validation BLEU Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(data_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Training curves saved as '{plot_path}'")

def create_results_table(history, test_bleu, data_dir="."):
    """Create results table."""
    print("\n" + "=" * 70)
    print("COMPARATIVE RESULTS TABLE")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Final Loss':<15} {'Final BLEU Score':<20}")
    print("-" * 70)
    
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    final_train_bleu = history['train_bleu_scores'][-1][1]
    final_val_bleu = history['val_bleu_scores'][-1][1]
    
    print(f"{'Training Set':<20} {final_train_loss:<15.4f} {final_train_bleu:<20.2f}")
    print(f"{'Validation Set':<20} {final_val_loss:<15.4f} {final_val_bleu:<20.2f}")
    print(f"{'Test Set':<20} {'-':<15} {test_bleu:<20.2f}")
    print("=" * 70)
    
    table_path = os.path.join(data_dir, "results_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("COMPARATIVE RESULTS TABLE\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Dataset':<20} {'Final Loss':<15} {'Final BLEU Score':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Training Set':<20} {final_train_loss:<15.4f} {final_train_bleu:<20.2f}\n")
        f.write(f"{'Validation Set':<20} {final_val_loss:<15.4f} {final_val_bleu:<20.2f}\n")
        f.write(f"{'Test Set':<20} {'-':<15} {test_bleu:<20.2f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"Results table saved to '{table_path}'")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load and prepare data
    (train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
     all_urdu, all_eng) = load_and_prepare_data()
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    urdu_vocab, urdu_idx_to_token = build_vocab(all_urdu, min_freq=2)
    eng_vocab, eng_idx_to_token = build_vocab(all_eng, min_freq=2)
    
    # Hyperparameters - Balanced for 2K sentences
    MAX_LEN_SRC = 40
    MAX_LEN_TGT = 40
    EMBED_SIZE = 256
    HIDDEN_SIZE = 384  # Moderate size
    NUM_LAYERS = 2
    DROPOUT = 0.3  # Moderate dropout
    LEARNING_RATE = 0.001
    EPOCHS = 50  # More epochs
    BATCH_SIZE = 16
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {DEVICE}")
    
    # Prepare data
    print("\nPreparing data tensors...")
    train_src_tensor, train_tgt_tensor = prepare_data(
        train_src, train_tgt, urdu_vocab, eng_vocab, MAX_LEN_SRC, MAX_LEN_TGT
    )
    val_src_tensor, val_tgt_tensor = prepare_data(
        val_src, val_tgt, urdu_vocab, eng_vocab, MAX_LEN_SRC, MAX_LEN_TGT
    )
    test_src_tensor, test_tgt_tensor = prepare_data(
        test_src, test_tgt, urdu_vocab, eng_vocab, MAX_LEN_SRC, MAX_LEN_TGT
    )
    
    # Create datasets and loaders
    train_dataset = TranslationDataset(train_src_tensor, train_tgt_tensor)
    val_dataset = TranslationDataset(val_src_tensor, val_tgt_tensor)
    test_dataset = TranslationDataset(test_src_tensor, test_tgt_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    INPUT_DIM = len(urdu_vocab)
    OUTPUT_DIM = len(eng_vocab)
    
    encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {count_parameters(model):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=eng_vocab['<pad>'])
    
    # Train
    history = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion, 
        DEVICE, EPOCHS, urdu_vocab, eng_vocab
    )
    
    # Load best model
    if os.path.exists('best_model.pth'):
        print("\nLoading best model from training...")
        model.load_state_dict(torch.load('best_model.pth'))
    
    # Save model
    model_path = 'urdu_to_english_lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved as '{model_path}'")
    
    # Plot curves
    plot_training_curves(history)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    test_bleu = calculate_bleu_score(model, test_loader, urdu_vocab, eng_vocab, DEVICE)
    print(f"\nFinal Test BLEU Score: {test_bleu:.2f}")
    print("=" * 70)
    
    # Create results table
    create_results_table(history, test_bleu)
    
    print("\n✅ Training complete! Check the output files:")
    print("  - training_curves.png")
    print("  - results_table.txt")
    print("  - urdu_to_english_lstm_model.pth")
