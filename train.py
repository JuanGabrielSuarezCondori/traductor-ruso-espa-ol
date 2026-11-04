import os
import sys
import json
import random
import time
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Constantes de tokens especiales
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocab:
    """Clase para manejar el vocabulario de tokens palabra por palabra."""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX
        }
        self.idx2word: Dict[int, str] = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts: Dict[str, int] = {}

    def build_vocab(self, sentences: List[str]):
        """Construye el vocabulario a partir de una lista de oraciones."""
        for sentence in sentences:
            for word in sentence.strip().split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        idx = len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def sentence_to_indices(self, sentence: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Convierte una oración en una lista de IDs de tokens."""
        tokens = sentence.strip().split()
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for token in tokens:
            indices.append(self.word2idx.get(token, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def indices_to_tokens(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """Convierte una lista de IDs de tokens a palabras legibles."""
        words = []
        for idx in indices:
            if idx == EOS_IDX and remove_special:
                break
            if remove_special and idx in (PAD_IDX, SOS_IDX):
                continue
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return words

    def __len__(self):
        return len(self.word2idx)


class ParallelTextDataset(Dataset):
    """Dataset de pares de oraciones paralelas Ruso - Español."""
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Vocab, trg_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, trg_text = self.pairs[idx]
        src_indices = self.src_vocab.sentence_to_indices(src_text, add_sos=False, add_eos=True)
        trg_indices = self.trg_vocab.sentence_to_indices(trg_text, add_sos=True, add_eos=True)
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)


def collate_fn(batch):
    """
    Agrupa secuencias de longitud variable y genera src_lens para pack_padded_sequence.
    Esto evita que los tokens de relleno (<pad>) corrompan el estado oculto del encoder.
    """
    src_batch, trg_batch = zip(*batch)
    
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    trg_lens = [len(y) for y in trg_batch]
    
    max_src_len = max(src_lens).item()
    max_trg_len = max(trg_lens)
    
    padded_src = torch.full((len(src_batch), max_src_len), PAD_IDX, dtype=torch.long)
    padded_trg = torch.full((len(trg_batch), max_trg_len), PAD_IDX, dtype=torch.long)
    
    for i, (src, trg) in enumerate(zip(src_batch, trg_batch)):
        padded_src[i, :len(src)] = src
        padded_trg[i, :len(trg)] = trg
        
    return padded_src, padded_trg, src_lens


# ==============================================================================
# ARQUITECTURA MANY-TO-MANY ASÍNCRONO (ENCODER-DECODER RECURRENTE CLÁSICO)
# ==============================================================================

class Encoder(nn.Module):
    """
    Fase 1: Lectura secuencial de las entradas rusas (x_1, ..., x_n).
    Una celda GRU procesa cada token actualizando el estado oculto h sin emitir predicciones.
    Usa pack_padded_sequence para extraer el estado oculto exacto tras el último token real.
    """
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src, src_lens):
        # src: [batch_size, src_len]
        # src_lens: [batch_size]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        
        # Empaquetado dinámico: ignora los pasos de <pad>
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        # hidden: [1, batch_size, hidden_dim] con el estado exacto en el último token real
        return hidden


class Decoder(nn.Module):
    """
    Fase 2: Generación secuencial de las salidas en español (y_1, ..., y_m).
    Inicia con el estado oculto del encoder y va emitiendo token a token.
    """
    def __init__(self, output_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, input_token, hidden):
        # input_token: [batch_size]
        # hidden: [1, batch_size, hidden_dim]
        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))   # [batch_size, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Modelo de Traducción Many-to-Many Asíncrono (Esquema clásico de Karpathy).
    1. Encoder procesa toda la entrada x_1...x_T -> h_T
    2. Traspaso: h_0(decoder) = h_T(encoder)
    3. Decoder genera paso a paso y_1...y_T'
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, trg, teacher_forcing_ratio: float = 0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        # FASE 1: Procesamiento completo de la secuencia de entrada en ruso
        hidden = self.encoder(src, src_lens)

        # FASE 2: Traspaso directo del estado oculto hacia el decoder
        # El primer token de entrada siempre es <sos>
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden = self.decoder.forward_step(input_token, hidden)
            outputs[:, t, :] = prediction

            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if use_teacher_force else top1

        return outputs


def translate_sentence(
    model: Seq2Seq,
    ru_sentence: str,
    src_vocab: Vocab,
    trg_vocab: Vocab,
    max_len: int = 15,
    device: torch.device = torch.device("cpu")
) -> Tuple[str, List[Dict]]:
    """
    Función de inferencia paso a paso:
    Procesa la frase en ruso con el Encoder y genera secuencialmente los tokens en español.
    Retorna la traducción final y una traza detallada de cada paso temporal.
    """
    model.eval()
    
    import re
    clean_ru = re.sub(r"[^а-яё0-9\s]", " ", ru_sentence.lower()).strip()
    words = clean_ru.split()
    
    if not words:
        return "", []

    src_indices = [src_vocab.word2idx.get(w, UNK_IDX) for w in words] + [EOS_IDX]
    src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([len(src_indices)], dtype=torch.long)

    step_details = []
    
    with torch.no_grad():
        # FASE 1: Encoder procesa los tokens de entrada
        hidden = model.encoder(src_tensor, src_lens)

        # FASE 2: Decoder inicia con <sos> y el hidden del encoder
        input_token = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
        generated_indices = []

        for step in range(1, max_len + 1):
            prediction, hidden = model.decoder.forward_step(input_token, hidden)
            
            probs = torch.softmax(prediction, dim=-1)
            token_idx = prediction.argmax(dim=-1).item()
            token_prob = probs[0, token_idx].item()
            token_word = trg_vocab.idx2word.get(token_idx, UNK_TOKEN)

            step_details.append({
                "step": step,
                "token_id": token_idx,
                "token_word": token_word,
                "probability": float(token_prob)
            })

            if token_idx == EOS_IDX:
                break

            generated_indices.append(token_idx)
            input_token = torch.tensor([token_idx], dtype=torch.long, device=device)

    out_tokens = trg_vocab.indices_to_tokens(generated_indices, remove_special=True)
    translation = " ".join(out_tokens)
    return translation, step_details


# ==============================================================================
# BUCLE DE ENTRENAMIENTO Y GUARDADO
# ==============================================================================

def train_epoch(model, dataloader, optimizer, criterion, clip: float = 1.0, teacher_forcing: float = 0.5):
    model.train()
    epoch_loss = 0.0

    for src, trg, src_lens in dataloader:
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()

        output = model(src, src_lens, trg, teacher_forcing_ratio=teacher_forcing)
        output_dim = output.shape[-1]
        
        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output_flat, trg_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg, src_lens in dataloader:
            src, trg = src.to(model.device), trg.to(model.device)
            output = model(src, src_lens, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(
    data_file: str = "data/ru_es.tsv",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 18,
    batch_size: int = 64,
    emb_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 0.001
):
    """Carga datos, entrena el modelo RNN many-to-many y guarda el checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")

    if not os.path.exists(data_file):
        print(f"Archivo {data_file} no encontrado. Ejecutando data_prep.py...")
        from data_prep import prepare_dataset
        prepare_dataset()

    pairs = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    print(f"Total de pares cargados: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    src_vocab = Vocab(min_freq=1)
    trg_vocab = Vocab(min_freq=1)
    src_vocab.build_vocab([p[0] for p in train_pairs])
    trg_vocab.build_vocab([p[1] for p in train_pairs])

    print(f"Vocabulario Ruso (Encoder): {len(src_vocab)} tokens")
    print(f"Vocabulario Español (Decoder): {len(trg_vocab)} tokens")

    train_dataset = ParallelTextDataset(train_pairs, src_vocab, trg_vocab)
    val_dataset = ParallelTextDataset(val_pairs, src_vocab, trg_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(input_dim=len(src_vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    decoder = Decoder(output_dim=len(trg_vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    print("\n--- Iniciando Entrenamiento Many-to-Many Asíncrono ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # Decaimiento suave de Teacher Forcing de 0.8 a 0.4
        tf_ratio = max(0.4, 0.8 - (epoch / epochs) * 0.4)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, teacher_forcing=tf_ratio)
        val_loss = evaluate(model, val_loader, criterion)
        epoch_dur = time.time() - epoch_start

        print(f"Época {epoch:02d}/{epochs:02d} | Duración: {epoch_dur:.1f}s | TF: {tf_ratio:.2f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "src_vocab_word2idx": src_vocab.word2idx,
                "src_vocab_idx2word": src_vocab.idx2word,
                "trg_vocab_word2idx": trg_vocab.word2idx,
                "trg_vocab_idx2word": trg_vocab.idx2word,
                "emb_dim": emb_dim,
                "hidden_dim": hidden_dim,
                "epoch": epoch,
                "val_loss": val_loss
            }
            save_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(checkpoint, save_path)
            torch.save(checkpoint, "model.pt")

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.1f} segundos.")
    print("Mejor modelo guardado en 'model.pt' y 'checkpoints/model.pt'")

    # Cargar el mejor modelo para evaluar
    best_ckpt = torch.load("model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_phrases = [
        "привет",
        "спасибо",
        "да",
        "нет",
        "хорошо",
        "где ты",
        "что ты делаешь",
        "мне пора идти спать",
        "я говорю по русски",
        "где ты живешь",
        "я студент",
        "кот",
        "собака",
        "книга"
    ]
    print("\n--- Evaluando traducciones de prueba (incluyendo palabras solas) ---")
    for phrase in test_phrases:
        trans, steps = translate_sentence(model, phrase, src_vocab, trg_vocab, device=device)
        step_words = [s['token_word'] for s in steps]
        print(f"RU: {phrase:25} -> ES: {trans:30} | Pasos: {step_words}")


if __name__ == "__main__":
    train_model(epochs=18, batch_size=64)
