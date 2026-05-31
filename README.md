# Picasso Protocol v1

피카소 프로토콜은 텍스트를 PNG 이미지에 숨기고, 다시 텍스트로 복원하거나 재해석하는 프로젝트입니다.

- `Artist X`: 원문 복원용
- `Artist Y`: 재해석용
- 제공 형태: 웹 데모(`toWebPage/`), 데스크톱 GUI(`ver.3/`), 학습 스크립트(`ver.3/train/`)

이 README는 개념 설명보다 실제 실행 방법에 집중합니다.

## 빠른 시작

가장 빨리 써보는 방법은 아래 순서입니다.

1. Python 가상환경을 만듭니다.
2. 필요한 패키지를 설치합니다.
3. `artist_x_best_model.pth`, `artist_y_standalone.pth`를 준비합니다.
4. 두 모델 파일을 `toWebPage/` 폴더에 넣습니다.
5. `toWebPage/`로 이동한 뒤 `python server.py`로 서버를 실행합니다.
6. `toWebPage/index.html`을 브라우저에서 엽니다.

## 1. 모델 학습 / 모델 파일 준비

이 프로젝트는 결국 아래 두 모델 파일이 있어야 제대로 쓸 수 있습니다.

- `artist_x_best_model.pth`
- `artist_y_standalone.pth`

### 이미 학습된 모델을 받아서 바로 쓰는 경우

현재 저장소에는 모델 파일이 포함되어 있지 않습니다.

다운로드 링크:

- [Model Data](https://drive.google.com/drive/folders/1p2EyQxCJMCiGHDhB0LjuIHfLjvf5jvvE?usp=sharing)

받은 뒤 웹 데모를 쓸 경우 두 파일을 `toWebPage/` 안에 넣으면 됩니다.

### 직접 학습해서 모델을 만드는 경우

순서는 아래와 같습니다.

1. `ver.3/train/download_models.py`로 BERT 기본 모델 캐시
2. `ver.3/train/artist_x_train.py`로 X 학습
3. `ver.3/train/y_bundler.py`로 Y 독립 모델 생성

실행 예시:

```powershell
cd ver.3\train
python download_models.py
python artist_x_train.py
python y_bundler.py
```

생성 결과:

- `artist_x_best_model.pth`
- `artist_y_standalone.pth`

중요:

- `artist_x_train.py`는 `wikitext-2-raw-v1`로 X를 학습합니다.
- `y_bundler.py`는 X의 인코더를 Y에 복사해 단일 모델 파일을 만듭니다.
- [`ver.3/train/artist_y_train.py`](./ver.3/train/artist_y_train.py)는 아직 개념 설명에 가깝습니다.
- 그래서 현재 구조에서는 X는 학습 가능하지만, Y는 "품질 좋은 창의적 재해석 모델"까지 완전히 학습되는 상태는 아닙니다.

### 웹 데모용 모델 배치 위치

`server.py`는 현재 작업 디렉터리 기준으로 모델을 읽습니다. 따라서 웹 데모를 쓸 때는 두 파일을 반드시 `toWebPage/` 안에 둬야 합니다.

```text
toWebPage/
├─ server.py
├─ index.html
├─ artist_x_best_model.pth
└─ artist_y_standalone.pth
```

### 데스크톱 GUI용 모델 배치 위치

GUI에서는 실행 후 파일 선택 창이 뜨므로, 모델 파일이 꼭 `ver.3/` 안에 있을 필요는 없습니다. 다만 찾기 쉽게 한 폴더에 두는 것을 권장합니다.

## 2. 준비물

### 공통 요구사항

- Python 3.11 이상
- pip
- 인터넷 연결
  - 처음 실행 시 Hugging Face의 `bert-base-uncased` 관련 파일을 받을 수 있습니다.
- Tkinter
  - 데스크톱 GUI를 쓸 경우 필요합니다.

### 권장 설치

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.7.1 transformers==4.54.0 datasets==4.0.0 pillow==11.2.1 numpy==2.3.1 matplotlib==3.10.3 flask flask-cors tqdm
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 transformers==4.54.0 datasets==4.0.0 pillow==11.2.1 numpy==2.3.1 matplotlib==3.10.3 flask flask-cors tqdm
```

주의:

- GPU 환경이면 PyTorch는 CUDA 버전에 맞게 별도 설치하는 편이 안전합니다.
- `requirements.txt`는 없어서 현재는 수동 설치가 필요합니다.

## 3. 웹 데모 사용법

웹 데모가 가장 간단합니다.

### 1) 서버 실행

```powershell
cd toWebPage
python server.py
```

정상 실행되면 콘솔에 아래와 비슷한 로그가 나옵니다.

```text
서버 시작 중... 모델을 불러옵니다.
✅ 모델 로딩 완료. (Device: cpu)
```

기본 주소는 `http://127.0.0.1:5000`입니다.

### 2) 웹 화면 열기

아래 파일을 브라우저에서 직접 엽니다.

- [`toWebPage/index.html`](./toWebPage/index.html)

또는 파일 탐색기에서 `toWebPage/index.html`을 더블클릭해도 됩니다.

### 3) 사용 흐름

1. 모드를 고릅니다.
   - `Artist Y`: 공개용, 디코딩 시 재해석 결과
   - `Artist X`: 개인용, 디코딩 시 원문 복원 결과
2. 텍스트를 입력합니다.
3. `PNG 이미지 생성`을 누릅니다.
4. 생성된 이미지를 저장합니다.
5. 다시 PNG를 업로드하면 텍스트가 복원되거나 재해석됩니다.

### 4) 웹 데모에서 자주 막히는 부분

- 서버를 안 켜고 `index.html`만 열면 동작하지 않습니다.
- 모델 파일이 `toWebPage/` 안에 없으면 서버가 시작되지 않습니다.
- `index.html`은 `http://127.0.0.1:5000`으로 요청을 보내므로, 서버 포트를 바꾸면 프론트도 같이 수정해야 합니다.

## 4. 데스크톱 GUI 사용법

웹 대신 로컬 GUI로도 사용할 수 있습니다.

### Artist X 실행

```powershell
cd ver.3
python app_x.py
```

실행 후 순서:

1. `Artist X 모델 불러오기 (.pth)` 클릭
2. `artist_x_best_model.pth` 선택
3. 텍스트를 입력하고 `텍스트 → PNG 이미지` 실행
4. 또는 PNG 파일을 열어 `PNG 이미지 → 텍스트` 실행

### Artist Y 실행

```powershell
cd ver.3
python app_y.py
```

실행 후 순서:

1. `Artist Y 모델 불러오기 (.pth)` 클릭
2. `artist_y_standalone.pth` 선택
3. 텍스트를 입력하고 `텍스트 → PNG 이미지` 실행
4. 또는 PNG 파일을 열어 `PNG 이미지 → 텍스트 재해석` 실행

### GUI를 쓸 때 알아둘 점

- GUI는 모델 파일 경로를 직접 선택하는 방식입니다.
- X는 복원용, Y는 생성형 재해석용이라 같은 PNG라도 결과가 다를 수 있습니다.
- Tkinter가 없는 환경에서는 GUI가 뜨지 않습니다.

## 5. API로 직접 쓰는 방법

서버를 실행한 뒤 `POST /encode`, `POST /decode`를 호출할 수 있습니다.

### `POST /encode`

요청:

```json
{
  "text": "비밀 메시지",
  "mode": "x"
}
```

- `mode`: `x` 또는 `y`
- 응답: base64 PNG 문자열

응답 예시:

```json
{
  "image": "<base64 png>"
}
```

### `POST /decode`

요청:

```json
{
  "image": "<base64 png>",
  "mode": "x"
}
```

응답 예시:

```json
{
  "text": "복원 결과"
}
```

### 간단한 호출 예시

```bash
curl -X POST http://127.0.0.1:5000/encode \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"secret message\",\"mode\":\"x\"}"
```

## 6. 폴더별 용도

```text
toWebPage/
  웹 데모 UI와 Flask 서버

ver.3/
  Artist X, Artist Y 데스크톱 GUI

ver.3/train/
  X 학습, Y 개념 학습, 모델 다운로드, Y 번들링 스크립트
```

## 7. 문제 해결

### `artist_x_best_model.pth` 또는 `artist_y_standalone.pth`를 찾을 수 없다고 나올 때

- 웹 서버: 두 파일이 `toWebPage/` 안에 있는지 확인
- GUI: 파일 선택 창에서 올바른 `.pth` 파일을 선택했는지 확인

### Hugging Face 관련 다운로드 오류가 날 때

- 인터넷 연결 확인
- 사내망/프록시 환경이면 Hugging Face 접속 가능 여부 확인
- 먼저 `python ver.3/train/download_models.py`로 기본 모델 캐시 시도

### GUI가 실행되지 않을 때

- Python에 Tkinter가 포함되어 있는지 확인
- Linux는 별도 패키지 설치가 필요할 수 있음

### 웹 화면은 열리는데 버튼이 실패할 때

- `server.py`가 실행 중인지 확인
- 브라우저 개발자 도구에서 `127.0.0.1:5000` 요청 실패 여부 확인

## 8. 참고

- 라이선스: [LICENSE](./LICENSE)
- 데이터셋: [Wikitext-2 (raw)](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1)

## 한 줄 요약

바로 써보려면:

1. 패키지 설치
2. 모델 2개 다운로드
3. `toWebPage/`에 모델 복사
4. `cd toWebPage` 후 `python server.py`
5. `toWebPage/index.html` 열기
