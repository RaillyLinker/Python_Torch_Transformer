import torch
from transformers import AutoTokenizer
from transformer import Transformer

# 하이퍼파라미터 - 학습 시와 동일하게 유지
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로드
tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-uncased")
pad_token_id = tokenizer_tgt.pad_token_id

# 모델 로드
model = torch.load("model_full.pth", map_location=DEVICE, weights_only=False)
model.eval()

# 번역 함수
def translate_sentence(sentence_de):
    # 1. 독일어 문장을 토크나이즈
    tokens_src = tokenizer_src(sentence_de, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    src_input = tokens_src["input_ids"].to(DEVICE)

    # 2. 인코더에 입력
    src_mask = model._make_src_mask(src_input)
    memory = model.encode(src_input, src_mask)

    # 3. 디코더 초기 입력 (<pad> or <bos> 토큰으로 시작)
    tgt_tokens = torch.full((1, 1), tokenizer_tgt.cls_token_id or tokenizer_tgt.pad_token_id, dtype=torch.long, device=DEVICE)

    # 4. 토큰을 하나씩 생성
    for _ in range(MAX_LEN):
        tgt_mask = model._make_tgt_mask(tgt_tokens)
        out = model.decode(tgt_tokens, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        logits = model.generator(out[:, -1])  # 마지막 위치
        next_token = logits.argmax(dim=-1).unsqueeze(1)  # Greedy decoding
        tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

        # 종료 조건: [SEP] 또는 [PAD] 토큰이면 중단
        if next_token.item() in [tokenizer_tgt.sep_token_id, tokenizer_tgt.pad_token_id]:
            break

    # 5. 디코더 출력에서 첫 토큰 제거하고 복호화
    output_tokens = tgt_tokens.squeeze(0).tolist()[1:]  # 첫 토큰 제거
    translated_text = tokenizer_tgt.decode(output_tokens, skip_special_tokens=True)
    return translated_text

# 예시 문장
german_sentence = "Guten Morgen, wie geht es dir?"
translated = translate_sentence(german_sentence)
print(f"German: {german_sentence}")
print(f"English: {translated}")
