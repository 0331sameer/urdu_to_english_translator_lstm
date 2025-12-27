"""
Test the trained Urdu-to-English translation model on sample sentences
"""

import torch
import torch.nn as nn
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# MODEL CLASSES (same as training)
# ============================================================================

class LSTMCell(nn.Module):
    """Custom LSTM Cell implementation from scratch."""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_io = nn.Linear(input_size, hidden_size)
        
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
# HELPER FUNCTIONS
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
    
    return vocab, idx_to_token

def tokenize_sentence(sentence, vocab, max_len):
    """Tokenize and pad sentence."""
    tokens = [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    while len(tokens) < max_len:
        tokens.append(vocab["<pad>"])
    
    return tokens

def translate_sentence(model, sentence, src_vocab, tgt_vocab, idx_to_tgt, max_len, device):
    """Translate a single sentence."""
    model.eval()
    
    # Tokenize source sentence
    src_tokens = tokenize_sentence(sentence, src_vocab, max_len)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Start with <sos> token
        decoder_input = torch.tensor([tgt_vocab['<sos>']], device=device)
        
        translated_tokens = []
        
        # Decode
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            predicted_token = output.argmax(1).item()
            
            # Stop if <eos> token
            if predicted_token == tgt_vocab['<eos>']:
                break
            
            translated_tokens.append(predicted_token)
            decoder_input = torch.tensor([predicted_token], device=device)
        
        # Convert tokens to words
        translated_words = [idx_to_tgt.get(token, "<unk>") for token in translated_tokens]
        translation = " ".join([word for word in translated_words if word not in ["<pad>", "<unk>"]])
    
    return translation

def beam_search_translate(model, sentence, src_vocab, tgt_vocab, idx_to_tgt, max_len, device, beam_width=5):
    """Translate using beam search for better quality."""
    model.eval()
    
    # Tokenize source
    src_tokens = tokenize_sentence(sentence, src_vocab, max_len)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Initialize beam
        # Each beam item: (sequence, score, hidden, cell)
        beams = [([tgt_vocab['<sos>']], 0.0, hidden, cell)]
        
        for _ in range(max_len):
            new_beams = []
            
            for seq, score, h, c in beams:
                # Stop if this beam ended
                if seq[-1] == tgt_vocab['<eos>']:
                    new_beams.append((seq, score, h, c))
                    continue
                
                # Get predictions
                decoder_input = torch.tensor([seq[-1]], device=device)
                output, new_h, new_c, _ = model.decoder(decoder_input, h, c, encoder_outputs)
                
                # Get top k predictions
                log_probs = torch.log_softmax(output, dim=1)
                top_probs, top_indices = torch.topk(log_probs[0], beam_width)
                
                # Add to beam
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score, new_h, new_c))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams ended
            if all(seq[-1] == tgt_vocab['<eos>'] for seq, _, _, _ in beams):
                break
        
        # Get best sequence
        best_seq = beams[0][0][1:]  # Skip <sos>
        
        # Remove <eos> if present
        if best_seq and best_seq[-1] == tgt_vocab['<eos>']:
            best_seq = best_seq[:-1]
        
        # Convert to words
        translated_words = [idx_to_tgt.get(token, "<unk>") for token in best_seq]
        translation = " ".join([word for word in translated_words if word not in ["<pad>", "<unk>"]])
    
    return translation

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("URDU TO ENGLISH TRANSLATION - TESTING")
    print("=" * 70)
    
    # Load data and build vocabularies
    (train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
     all_urdu, all_eng) = load_and_prepare_data()
    
    print("\nBuilding vocabularies...")
    urdu_vocab, urdu_idx_to_token = build_vocab(all_urdu, min_freq=2)
    eng_vocab, eng_idx_to_token = build_vocab(all_eng, min_freq=2)
    
    # Model hyperparameters (must match training)
    MAX_LEN_SRC = 40
    MAX_LEN_TGT = 40
    EMBED_SIZE = 256
    HIDDEN_SIZE = 384
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Initialize model
    INPUT_DIM = len(urdu_vocab)
    OUTPUT_DIM = len(eng_vocab)
    
    encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    # Load trained model
    model_path = 'urdu_to_english_lstm_model.pth'
    if os.path.exists(model_path):
        print(f"\nLoading trained model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file '{model_path}' not found!")
        exit(1)
    
    # Test translations
    print("\n" + "=" * 70)
    print("SAMPLE TRANSLATIONS FROM TEST SET")
    print("=" * 70)
    
    # Select random samples from test set
    num_samples = 10
    random.seed(42)
    sample_indices = random.sample(range(len(test_src)), min(num_samples, len(test_src)))
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'-' * 70}")
        print(f"Example {i}:")
        print(f"{'-' * 70}")
        print(f"Source (Urdu):")
        print(f"  {test_src[idx]}")
        print(f"\nReference (English):")
        print(f"  {test_tgt[idx]}")
        
        translation = beam_search_translate(
            model, test_src[idx], urdu_vocab, eng_vocab, 
            eng_idx_to_token, MAX_LEN_TGT, DEVICE, beam_width=5
        )
        
        print(f"\nModel Translation (Beam Search):")
        print(f"  {translation}")
    
    # Additional custom sentences to translate
    print("\n" + "=" * 70)
    print("ADDITIONAL CUSTOM SENTENCE TRANSLATIONS")
    print("=" * 70)
    
    custom_sentences = [
        "میں اردو سیکھ رہا ہوں۔",
        "یہ کتاب بہت اچھی ہے۔",
        "آج موسم خوشگوار ہے۔",
        "میں پاکستان سے ہوں۔",
        "کیا آپ انگریزی بولتے ہیں؟",
        "مجھے کھانا پسند ہے۔",
        "وہ سکول جاتا ہے۔",
        "یہ شہر بہت خوبصورت ہے۔",
        "میں کل جاؤں گا۔",
        "آپ کا نام کیا ہے؟",
        "میں آپ سے ملنا چاہتا ہوں۔",
        "یہ میری کتاب ہے۔",
        "پانی پینا صحت کے لیے اچھا ہے۔",
        "سورج آسمان میں چمک رہا ہے۔",
        "بچے پارک میں کھیل رہے ہیں۔"
    ]
    
    for i, sentence in enumerate(custom_sentences, 1):
        print(f"\n{'-' * 70}")
        print(f"Custom Example {i}:")
        print(f"Urdu: {sentence}")
        
        translation = beam_search_translate(
            model, sentence, urdu_vocab, eng_vocab,
            eng_idx_to_token, MAX_LEN_TGT, DEVICE, beam_width=5
        )
        
        print(f"Translation (Beam Search): {translation}")
    
    print("\n" + "=" * 70)
    print("TRANSLATION TESTING COMPLETE")
    print("=" * 70)
