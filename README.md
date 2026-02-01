# Picasso Protocol v1

피카소 프로토콜(Picasso Protocol)은 **파블로 피카소의 입체파 미학**에서 영감을 받은 비대칭 스테가노그래피(텍스트 → 이미지 → 텍스트) 시스템입니다. 프로젝트의 핵심은 **하나의 인코더 + 두 개의 디코더** 구조로, 동일한 텍스트를 **보안 복원용(Artist X)** 또는 **창의적 재해석용(Artist Y)** 결과로 분기해내는 데 있습니다. 이 저장소는 연구/시연용 모델 스크립트, 데스크톱 GUI, 그리고 웹 데모(Flask + HTML)를 함께 제공합니다.

---

## ✨ 주요 개념 요약

- **공유 인코더(Encoder)**: 텍스트를 잠재 벡터(latent vector)로 변환합니다.
- **Artist X (Private / 복원용)**: 잠재 벡터를 원문 텍스트로 최대한 정확히 복원합니다.
- **Artist Y (Public / 재해석용)**: 동일한 잠재 벡터를 새로운 텍스트로 창의적으로 재해석합니다.
- **이미지 기반 스테가노그래피**: 잠재 벡터와 원문 길이 정보를 RGBA PNG 픽셀 데이터에 **바이트 단위로 숨겨** 전송합니다.

---

## 📦 레포 구조

```
Picasso-Protocol-1.0/
├─ README.md
├─ toWebPage/
│  ├─ index.html          # 웹 데모 UI (Tailwind 기반)
│  └─ server.py           # Flask API 서버 (encode/decode)
├─ ver.3/
│  ├─ app_x.py             # Artist X 데스크톱 GUI
│  ├─ app_y.py             # Artist Y 데스크톱 GUI
│  └─ train/
│     ├─ artist_x_train.py # Artist X 학습 스크립트
│     ├─ artist_y_train.py # Artist Y 학습 개념 스크립트
│     ├─ download_models.py# Hugging Face 모델 다운로드
│     ├─ image_utils.py    # PNG 인코딩/디코딩 유틸리티
│     └─ y_bundler.py       # X 인코더를 Y에 포함해 단일 모델 생성
├─ LICENSE
└─ (이미지/문서 파일들)
```

---

## ✅ 요구사항

### 공통
- **Python 3.11+**
- **PyTorch 2.7.1**
- **Transformers 4.54.0**
- **Datasets 4.0.0**
- **Pillow 11.2.1**
- **NumPy 2.3.1**
- **Matplotlib 3.10.3**
- **Tkinter** (GUI 실행 시 필요)

### 설치 예시 (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 transformers==4.54.0 datasets==4.0.0 pillow==11.2.1 numpy==2.3.1 matplotlib==3.10.3 flask flask-cors
```

> ⚠️ 환경에 따라 PyTorch 설치 방식이 다를 수 있습니다. GPU 사용 시에는 CUDA 버전에 맞는 설치 명령을 사용하세요.

---

## 🚀 웹 데모 실행 (Flask + HTML)

### 1) 모델 파일 준비
웹 서버는 다음 모델 파일을 **`toWebPage/` 폴더 안**에서 로드합니다.

- `artist_x_best_model.pth`
- `artist_y_standalone.pth`

모델 파일은 아래 링크에서 받을 수 있습니다:
- **Model Data**: https://drive.google.com/drive/folders/1p2EyQxCJMCiGHDhB0LjuIHfLjvf5jvvE?usp=sharing

> 파일을 `toWebPage/` 안에 직접 넣어주세요.

### 2) 서버 실행
```bash
cd toWebPage
python server.py
```

서버가 실행되면 다음과 같은 로그가 출력됩니다:
```
서버 시작 중... 모델을 불러옵니다.
✅ 모델 로딩 완료. (Device: cpu 또는 cuda)
```

### 3) 웹 UI 열기
`toWebPage/index.html` 파일을 브라우저로 직접 열면 됩니다.

웹 UI에서 제공하는 기능:
- **모드 선택**: Artist X / Artist Y
- **텍스트 → 이미지 인코딩**
- **이미지 → 텍스트 디코딩**
- 결과 PNG 다운로드

---

## 🧩 Flask API 요약

### POST `/encode`
텍스트를 이미지(베이스64 PNG)로 변환합니다.

**Request JSON**
```json
{ "text": "비밀 메시지", "mode": "x" }
```

- `mode`: `x` 또는 `y` (기본값 `y`)

**Response JSON**
```json
{ "image": "<base64>" }
```

### POST `/decode`
이미지(베이스64 PNG)를 텍스트로 복원합니다.

**Request JSON**
```json
{ "image": "<base64>", "mode": "x" }
```

**Response JSON**
```json
{ "text": "복원 결과" }
```

---

## 🖥 데스크톱 GUI 실행 (Artist X / Y)

### Artist X (복원용)
```bash
cd ver.3
python app_x.py
```

기능:
1. **모델 불러오기** (`artist_x_best_model.pth` 선택)
2. 텍스트 → PNG 이미지 생성
3. PNG → 텍스트 복원

### Artist Y (재해석용)
```bash
cd ver.3
python app_y.py
```

기능:
1. **모델 불러오기** (`artist_y_standalone.pth` 선택)
2. 텍스트 → PNG 이미지 생성
3. PNG → 텍스트 재해석

> ✅ GUI는 내부적으로 **same encoder + different decoder** 구조를 사용합니다.

---

## 🧠 모델 학습 흐름 (ver.3/train)

### 1) 사전 모델 다운로드 (오프라인 캐시용)
```bash
cd ver.3/train
python download_models.py
```
- Hugging Face에서 `bert-base-uncased` 모델과 토크나이저를 내려받아 로컬 캐시에 저장합니다.
- 이후 학습 및 실행을 **오프라인**으로 진행할 수 있습니다.

### 2) Artist X 학습
```bash
python artist_x_train.py
```
- 데이터셋: `wikitext-2-raw-v1`
- 목적: **원문 복원용 디코더** 학습
- 출력: `artist_x_best_model.pth`

### 3) Artist Y 구성 (X의 인코더 복사)
```bash
python y_bundler.py
```
- Artist X의 인코더를 Artist Y에 이식
- 출력: `artist_y_standalone.pth`

### 4) Artist Y 학습 (개념 스크립트)
`artist_y_train.py`는 **개념적 구조**만 제공하며, 실제 학습을 위해서는
창의적 재해석 데이터셋(원문 → 새 텍스트 페어)이 필요합니다.

---

## 🔍 PNG 인코딩 방식 요약

1. 텍스트 → BERT 인코더 → 잠재 벡터 (float32)
2. 잠재 벡터 + 원문 길이를 바이트 스트림으로 결합
3. 1280×1280 RGBA PNG 픽셀 배열에 바이트 그대로 삽입
4. 중앙에는 **픽اسو 스타일 추상화 시각화**를 생성해 시각적 커버를 제공
5. 복원 시 이미지에서 바이트 추출 → 텐서 재구성 → 디코더 실행

---

## 📚 참고 문헌

- RFNNS: Robust Fixed Neural Network Steganography with Popular Deep Generative Models  
  https://arxiv.org/pdf/2505.04116
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
  https://arxiv.org/abs/1810.04805

---

## 📂 데이터셋

- Wikitext-2 (raw): https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1

---

## 📝 라이선스

이 프로젝트는 `LICENSE` 파일에 명시된 조건을 따릅니다.

---

## 🙌 기여 안내

- 이슈/PR 환영합니다.
- 학습용 창의적 데이터셋이나 보안성 검증 관련 실험 결과를 공유해주시면 큰 도움이 됩니다.

---

## ✅ 빠른 실행 체크리스트

- [ ] `artist_x_best_model.pth` / `artist_y_standalone.pth` 확보
- [ ] `toWebPage/`에 모델 파일 배치
- [ ] `python toWebPage/server.py` 실행
- [ ] `toWebPage/index.html` 열어 데모 확인

---

## ⚠️ 주의 사항

- **모델 파일이 없으면 서버가 실행되지 않습니다.**
- 실행 환경에 따라 **GPU/CPU 성능 차이가 큽니다.**
- `artist_y_train.py`는 실제 학습 로직이 아닌 **개념 흐름**을 설명합니다.

---

즐거운 실험과 창작을 바랍니다! 🎨
