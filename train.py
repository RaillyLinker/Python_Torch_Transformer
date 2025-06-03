import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from transformer import Transformer

# 하이퍼파라미터 설정
BATCH_SIZE = 64
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1
MAX_LEN = 128
EPOCHS = 10
PAD_TOKEN = "<pad>"

# 데이터셋 로드 및 전처리
dataset = load_dataset("wmt14", "de-en", split={"train": "train[:1%]", "valid": "validation[:1%]"})

# tokenizer는 HuggingFace의 pre-trained tokenizer 사용 (예: t5-small)
tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-uncased")

# <pad> 토큰 id
pad_token_id = tokenizer_tgt.pad_token_id


# 토크나이즈 함수
def tokenize_fn(example):
    src_text = example["translation"]["de"]
    tgt_text = example["translation"]["en"]
    src = tokenizer_src(src_text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    tgt = tokenizer_tgt(tgt_text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return {
        "src_input": src.input_ids.squeeze(0),
        "tgt_input": tgt.input_ids.squeeze(0)[:-1],
        "tgt_output": tgt.input_ids.squeeze(0)[1:]
    }


# 전처리 적용
dataset["train"] = dataset["train"].map(tokenize_fn)
dataset["valid"] = dataset["valid"].map(tokenize_fn)


# PyTorch Dataset으로 변환
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["src_input"], dtype=torch.long),
            torch.tensor(item["tgt_input"], dtype=torch.long),
            torch.tensor(item["tgt_output"], dtype=torch.long),
        )


train_loader = DataLoader(TranslationDataset(dataset["train"]), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(TranslationDataset(dataset["valid"]), batch_size=BATCH_SIZE)

# 모델 초기화
model = Transformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    vocab_size_src=tokenizer_src.vocab_size,
    vocab_size_tgt=tokenizer_tgt.vocab_size,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    dim_feedforward=FFN_HID_DIM,
    dropout=DROPOUT,
    max_len=MAX_LEN,
    pad_token_id=pad_token_id
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 옵티마이저 및 손실 함수
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt_in, tgt_out in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        optimizer.zero_grad()
        output = model(src, tgt_in)
        loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Training Loss: {total_loss / len(train_loader):.4f}")

    # 검증 루프
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in valid_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            output = model(src, tgt_in)
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            val_loss += loss.item()
    print(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(valid_loader):.4f}")
    torch.save(model, "model_full.pth")
