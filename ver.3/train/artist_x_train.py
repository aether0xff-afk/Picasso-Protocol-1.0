# --- 필요한 라이브러리들을 가져옵니다 ---
from pathlib import Path
import sys

import torch
from torch import nn # PyTorch의 신경망 모듈
from torch.utils.data import DataLoader # 데이터를 배치 단위로 묶어주는 유틸리티
from transformers import BertTokenizer, DataCollatorWithPadding # Hugging Face 라이브러리
from datasets import load_dataset # Hugging Face의 데이터셋 라이브러리
from tqdm import tqdm # 학습 진행률을 보여주는 라이브러리
import warnings
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from picasso_protocol.models import ArtistX  # noqa: E402

# transformers 라이브러리의 일부 경고 메시지는 학습에 지장을 주지 않으므로 끕니다.
warnings.filterwarnings("ignore")

# --- 2. 메인 실행 블록 ---
# 이 스크립트 파일이 직접 실행될 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # --- 학습에 필요한 주요 설정값 (하이퍼파라미터) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # GPU가 있으면 cuda, 없으면 cpu 사용
    MODEL_NAME = 'bert-base-uncased' # 사용할 사전학습 모델 이름
    LEARNING_RATE = 2e-5 # 학습률: 모델이 얼마나 큰 보폭으로 정답을 찾아갈지 결정
    BATCH_SIZE = 16    # 배치 사이즈: 한 번에 몇 개의 데이터를 보고 학습할지 결정
    NUM_EPOCHS = 20    # 에포크: 전체 데이터셋을 총 몇 번 반복해서 학습할지 결정

    print(f"🚀 ArtistX 학습을 시작합니다. Device: {DEVICE}")

    # --- 데이터 준비 단계 ---
    print("1. 데이터셋 로드 및 전처리 중...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    raw_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1') # Hugging Face Hub에서 데이터셋 다운로드
    
    # 텍스트 데이터를 토큰 ID로 변환하는 함수
    def tokenize_function(examples):
        # 텍스트를 토크나이징하고, 최대 길이를 넘으면 자르고, 모자라면 채웁니다.
        return tokenizer(examples["text"], truncation=True, max_length=128)

    # .map()을 사용해 데이터셋 전체에 토크나이징 함수를 빠르게 적용
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch") # 데이터셋 형식을 PyTorch 텐서로 지정

    # Data Collator: 배치 내에서 가장 긴 문장에 맞춰 동적으로 패딩을 추가해주는 역할
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # DataLoader: 데이터셋을 배치 단위로 묶어주는 역할
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)
    print("✅ 데이터 준비 완료.")

    # --- 모델 및 학습 도구 설정 ---
    print("2. 모델 및 옵티마이저 설정 중...")
    model = ArtistX(model_name=MODEL_NAME, local_files_only=True).to(DEVICE) # 모델을 생성하고 지정된 디바이스로 보냄
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # 옵티마이저(최적화 도구) 설정
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # 손실 함수 설정 (패딩 토큰은 무시)
    print("✅ 모델 설정 완료.")

    # --- 본격적인 학습 루프 ---
    print("3. 학습을 시작합니다...")
    best_val_loss = float('inf') # 가장 낮은 검증 손실 값을 기록하기 위한 변수. 무한대로 초기화.

    # 정해진 에포크 수만큼 전체 데이터셋을 반복
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        # -- 훈련(Training) 단계 --
        model.train() # 모델을 '훈련 모드'로 전환
        train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in train_progress_bar:
            optimizer.zero_grad() # 이전 배치의 기울기(gradient)를 초기화
            batch = {k: v.to(DEVICE) for k, v in batch.items()} # 데이터를 디바이스로 이동
            output_logits = model(batch['input_ids'], batch['attention_mask']) # 모델 예측
            loss = loss_fn(output_logits.view(-1, tokenizer.vocab_size), batch['input_ids'].view(-1)) # 손실 계산
            loss.backward() # 역전파: 손실을 기반으로 각 파라미터의 기울기를 계산
            optimizer.step() # 옵티마이저가 계산된 기울기를 바탕으로 모델의 파라미터를 업데이트
            train_loss += loss.item() # 현재 배치의 손실 값을 누적
            train_progress_bar.set_postfix(loss=loss.item()) # 진행률 표시줄에 현재 손실 값 표시
        avg_train_loss = train_loss / len(train_dataloader)

        # -- 검증(Validation) 단계 --
        model.eval() # 모델을 '평가 모드'로 전환
        val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} [Val]")
        with torch.no_grad(): # 기울기 계산 비활성화
            for batch in val_progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                output_logits = model(batch['input_ids'], batch['attention_mask'])
                loss = loss_fn(output_logits.view(-1, tokenizer.vocab_size), batch['input_ids'].view(-1))
                val_loss += loss.item()
                val_progress_bar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # -- 최적 모델 저장 --
        # 현재 검증 손실이 이전에 기록된 최저 손실보다 낮으면, 모델을 저장
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "artist_x_best_model.pth")

    print("\n✅ 학습이 최종 완료되었습니다.")
    print(f"최적 모델의 가중치가 'artist_x_best_model.pth'에 저장되었습니다.")
