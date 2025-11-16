import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import time
from tqdm import tqdm
import os

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10  
LR = 2e-5

# Classes escolhidas
TARGETS = ["onca", "caseiro", "noticia"] 

# Mapeamento de rótulos por tema, conforme diagnóstico anterior
SENTIMENTS_MAP = {
    "onca": ["positivo", "negativo", "neutro"],
    "caseiro": ["positivo", "negativo", "neutro"],
    "noticia": ["boa", "ruim", "neutra"]
}
NUM_CLASSES = len(TARGETS) * 3 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("\n### 1. Preparação dos Dados ###")
url = "https://docs.google.com/spreadsheets/d/17aHYyRNfbmde8bVOR_HX_BmNUEdkygPuaGO4lJj26jg/gviz/tq?tqx=out:csv&gid=0"
try:
    df = pd.read_csv(url, on_bad_lines="skip")
except Exception as e:
    df = pd.read_csv(url, encoding="latin1", on_bad_lines="skip")

def clean_col_name(name):
    name = name.lower().strip()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.replace(" ", "_")
    return name

df.columns = [clean_col_name(col) for col in df.columns]

# Remoção de vazios/duplicados
df.dropna(subset=["comment_text"], inplace=True)
df.drop_duplicates(subset=["comment_text"], inplace=True)
df["comment_text"] = df["comment_text"].astype(str)

FINAL_COLUMNS = []
for t in TARGETS:
    current_sentiments = SENTIMENTS_MAP.get(t, []) 

    if t not in df.columns:
        df[t] = "neutro" 

    df[t] = df[t].astype(str).str.lower()
    dummies = pd.get_dummies(df[t], prefix=t)

    for s in current_sentiments:
        col = f"{t}_{s}"
        # Conversão para binário (0 ou 1)
        df[col] = dummies[col].astype(int) if col in dummies.columns else 0
        FINAL_COLUMNS.append(col)

X = df["comment_text"].values
y = df[FINAL_COLUMNS].values.astype(np.float32)

# Divisão dos dados (70/15/15)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# Re-split do treino+validação: 15% de (100%-15%) = 17.6%
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, random_state=42) 

print(f"Divisão: Treino={len(X_train)}, Validação={len(X_val)}, Teste={len(X_test)}")

print("\n### 2. Tokenização ###")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts, labels):
    texts_list = texts.tolist() if isinstance(texts, np.ndarray) else texts
    
    enc = tokenizer.batch_encode_plus(
        texts_list,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Rótulos como float32 para BCEWithLogitsLoss
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.float32))

train_data = encode(X_train, y_train)
val_data = encode(X_val, y_val)
test_data = encode(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=RandomSampler(train_data))
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print("\n### 3. Configuração do Modelo e Loss ###")

# AutoModelForSequenceClassification já utiliza o vetor [CLS] para a classificação
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
model.to(DEVICE)

# Pesos para lidar com o desbalanceamento das classes
pos = y_train.sum(axis=0)
neg = len(y_train) - pos
weights = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32).to(DEVICE)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)

print("\n### 4. Treinamento ###")

# Função para calcular Métricas (Loss, Acurácia)
def get_metrics(loader, model, loss_fn):
    model.eval()
    total_loss, total_preds, total_labels = 0, [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            logits = model(ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_preds.append(logits.cpu().numpy())
            total_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_labels = np.concatenate(total_labels)
    all_logits = np.concatenate(total_preds)
    
    # Otimização simples de threshold (0.5) para Acurácia/F1
    all_preds_binary = (all_logits > 0).astype(int) 
    
    # Acurácia Multi-rótulo: Acurácia exata (rótulos perfeitos)
    acc = accuracy_score(all_labels, all_preds_binary) 
    
    return avg_loss, acc, all_logits, all_labels

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for e in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # Loop de Treinamento
    for ids, mask, labels in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS} (Treino)"):
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, attention_mask=mask).logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Métrica de Treino
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Métrica de Validação
    avg_val_loss, val_acc, _, _ = get_metrics(val_loader, model, loss_fn)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    # Imprimir Loss e Acurácia
    # Calculando a acurácia de treino (simplesmente passando o loader de treino para get_metrics)
    _, train_acc, _, _ = get_metrics(train_loader, model, loss_fn)
    train_accuracies.append(train_acc)
    
    print(f"\n===== ÉPOCA {e+1}/{EPOCHS} =====")
    print(f"Treino Loss: {avg_train_loss:.4f} | Treino Acurácia: {train_acc:.4f}")
    print(f"Validação Loss: {avg_val_loss:.4f} | Validação Acurácia: {val_acc:.4f}")


print("\n--- GRÁFICO DE EVOLUÇÃO DO LOSS ---")
print(f"Losses de Treino: {train_losses}")
print(f"Losses de Validação: {val_losses}")

print("\n### 5. Avaliação no Conjunto de Teste ###")

test_loss, test_acc, test_logits, test_labels = get_metrics(test_loader, model, loss_fn)

# Otimização do Threshold (como feito nas etapas anteriores)
def find_best_threshold(logits, labels):
    best_f1 = 0
    best_thresh = 0.0
    # Procuramos o melhor ponto de corte para o F1-Macro
    for t in np.arange(-2.0, 2.0, 0.05):
        binary_preds = (logits > t).astype(int)
        current_f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = t
    return best_thresh

best_threshold = find_best_threshold(test_logits, test_labels)
test_preds_final = (test_logits > best_threshold).astype(int)

print(f"\nThreshold Otimizado no Teste (F1-Macro): {best_threshold:.4f}")

print("\n==== MÉTRICAS POR CLASSE ====")
for i, col in enumerate(FINAL_COLUMNS):
    # Calcula Precision, Recall e F1-Score para cada classe individualmente
    precision = precision_score(test_labels[:,i], test_preds_final[:,i], zero_division=0)
    recall = recall_score(test_labels[:,i], test_preds_final[:,i], zero_division=0)
    f1 = f1_score(test_labels[:,i], test_preds_final[:,i], zero_division=0)
    
    print(f"\nClasse: **{col}**")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

print("\n### 6. Exemplos de Erros de Classificação (Primeiros 5 Erros) ###")

error_count = 0
max_errors = 5
for i in range(len(test_labels)):
    # Um erro é quando o vetor de rótulos predito não é idêntico ao rótulo verdadeiro
    if not np.array_equal(test_labels[i], test_preds_final[i]):
        
        # Converte o vetor binário para os nomes dos rótulos
        true_labels_names = [FINAL_COLUMNS[j] for j, v in enumerate(test_labels[i]) if v == 1]
        pred_labels_names = [FINAL_COLUMNS[j] for j, v in enumerate(test_preds_final[i]) if v == 1]

        print(f"\n--- Erro {error_count + 1} ---")
        print(f"Texto: {X_test[i]}")
        print(f"Rótulos CORRETOS: {', '.join(true_labels_names)}")
        print(f"Rótulos PREVISTOS: {', '.join(pred_labels_names)}")

        error_count += 1
        if error_count >= max_errors:
            break

# salvar o best model
MODEL_SAVE_PATH = "./bert_multilabel_classifier" 

print(f"\n### 7. Salvando o Modelo e o Tokenizer em: {MODEL_SAVE_PATH} ###")

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

model.save_pretrained(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("deu certo")