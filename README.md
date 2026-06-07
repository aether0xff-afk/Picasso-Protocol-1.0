# Picasso Protocol v1

Einstein-Picasso Odyssey 프로젝트의 공식 저장소입니다.

Picasso Protocol은 텍스트를 latent JSON으로 변환하는 공개 `Artist Y`와,
latent JSON을 다시 텍스트로 복원하는 비공개 `Artist X`를 분리하는 신경망
프로토타입입니다. 암호학적으로 안전한 시스템이 아니라, 공개 인코더/비공개
디코더 구조와 inversion attack 가능성을 실험하고 발표하기 위한 데모입니다.

## 폴더 구조

```text
.
├── *.cmd                       # Windows 실행용 래퍼 명령
├── picasso_protocol/           # 공통 모델, brain fragment, latent 입출력 코드
├── tools/
│   ├── cli/                    # 내부 CLI 파이썬 스크립트
│   └── attacks/                # 공격/취약성 실험 스크립트
├── toWebPage/                  # Flask API 서버와 브라우저 데모
├── ver.3/
│   ├── app_x.py, app_y.py      # 구버전 Tkinter 앱
│   └── train/                  # 학습, export, checkpoint metric 스크립트
├── docs/                       # 발표 시연 대본과 공격 분석 문서
├── assets/                     # 포스터, 이미지 등 발표/시각 자료
├── samples/                    # 샘플 latent JSON
├── attack_results/             # 측정된 공격 결과와 그래프
└── models/                     # 다운로드/생성한 모델 파일, Git에는 포함하지 않음
```

루트에는 사용자가 바로 실행하는 `.cmd` 파일만 남겨 두었습니다. 파이썬 파일을
직접 실행해야 할 때는 `tools/cli` 또는 `tools/attacks` 아래 파일을 사용하면
됩니다.

## 필요 환경

- Windows PowerShell 또는 CMD
- Python 3.11+
- Conda 환경 이름 예시: `picasso-protocol`
- PyTorch 2.7.1
- Transformers 4.54.0
- Datasets 4.0.0
- Pillow 11.2.1
- NumPy 2.3.1
- Matplotlib 3.10.3
- Tkinter

이 저장소의 `.cmd` 파일은 현재 활성화된 Conda 환경의 Python을 우선 사용합니다.

```powershell
conda activate picasso-protocol
.\picasso.cmd status
```

Conda 환경을 활성화하지 않은 경우에는 PATH에 잡힌 `python`을 사용합니다. CPU
환경에서도 PyTorch CPU 버전과 모델 파일이 있으면 실행할 수 있습니다. 학습과
공격 실험은 CPU에서 매우 느릴 수 있습니다.

## 모델 파일 준비

모델 데이터는 아래 Google Drive에서 내려받습니다.

https://drive.google.com/drive/folders/1p2EyQxCJMCiGHDhB0LjuIHfLjvf5jvvE?usp=sharing

기본 CLI가 기대하는 모델 위치는 다음과 같습니다.

```text
models/public/artist_y_public_encoder.pt
models/private/artist_x_private_decoder.pt
models/public_v4/artist_y_public_encoder.pt
models/private_v4/artist_x_private_decoder.pt
models/public_v5b/artist_y_public_brain_fragment.pt
models/private_v5b/artist_x_private_brain_fragment.pt
models/public_v6_ultradrop/artist_y_public_brain_fragment.pt
models/private_v6_ultradrop/artist_x_private_brain_fragment.pt
```

`models/`는 Git에 올리지 않습니다. 공개 배포가 필요한 경우에도 private decoder
파일은 공유하지 마세요.

## 빠른 시작

모델과 GPU 상태를 확인합니다.

```powershell
.\picasso.cmd status
.\picasso.cmd verify-public
```

텍스트를 latent JSON으로 인코딩합니다.

```powershell
.\picasso.cmd encode "My Secret"
```

생성된 latent를 private decoder로 복원합니다.

```powershell
.\picasso.cmd decode picasso_latent.json
```

출력 파일명을 직접 지정할 수도 있습니다.

```powershell
.\picasso.cmd encode "My Secret" -o message.json
.\picasso.cmd decode message.json
```

제출/발표 직전의 최소 검증 절차만 필요하면
[docs/FINAL_TEST_GUIDE_KO.md](./docs/FINAL_TEST_GUIDE_KO.md)를 따라가세요.

Conda 래퍼 없이 직접 실행하려면 다음 형태를 사용합니다.

```powershell
python tools\cli\picasso.py status
python tools\cli\picasso.py encode "My Secret" -o message.json
python tools\cli\picasso.py decode message.json
```

## 역할 분리 CLI

발표나 배포 설명에서는 `Artist Y`와 `Artist X`를 분리해서 보여주는 명령을
사용하는 것이 가장 명확합니다.

공개 인코더 쪽:

```powershell
.\artist-y-public.cmd status
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json
```

비공개 디코더 쪽:

```powershell
.\artist-x-private.cmd status
.\artist-x-private.cmd decode captured_latent.json
```

의미는 다음과 같습니다.

- `artist-y-public.cmd`: 공개해도 되는 encoder만 사용합니다.
- `artist-x-private.cmd`: 수신자가 보관해야 하는 decoder를 사용합니다.
- `captured_latent.json`: 네트워크로 전달되거나 공격자가 볼 수 있는 latent 예시입니다.

## Web 데모

공개 인코딩만 가능한 서버를 실행합니다.

```powershell
.\server-public.cmd
```

브라우저에서 다음 파일을 엽니다.

```text
toWebPage/index.html
```

private decode까지 허용하는 로컬 수신자 서버는 다음 명령을 사용합니다.

```powershell
.\server-private.cmd
```

shielded public encoder 서버는 다음 명령을 사용합니다.

```powershell
.\server-public-shielded.cmd
```

서버 API는 다음 엔드포인트를 제공합니다.

```text
GET  /health
POST /encode-latent
POST /decode-latent
```

`server-public.cmd`와 `server-public-shielded.cmd`에서는 private decode가 꺼져
있습니다. `/decode-latent`는 `server-private.cmd`에서만 사용하세요.

## 공격 데모

발표용 기본 공격 시나리오는 다음 순서로 실행합니다.

```powershell
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json
.\attacker-demo.cmd captured_latent.json --known-text "My Secret"
.\artist-x-private.cmd decode captured_latent.json
```

전체 발표 흐름은 [docs/DEMO_SCRIPT_KO.md](./docs/DEMO_SCRIPT_KO.md)에 정리되어
있습니다.

decoder inversion 공격 실험은 다음 명령으로 실행합니다.

```powershell
.\attacker-inversion-demo.cmd --sizes 100 500 1000 --epochs 100
```

결과와 해석은 [docs/ATTACK_REPORT_KO.md](./docs/ATTACK_REPORT_KO.md)를 참고하세요.

## Shielded V4 데모

V4 shielded public encoder는 latent에 dropout/noise를 추가합니다.

```powershell
.\artist-y-shielded.cmd "My Secret" -o shielded_latent.json
.\artist-x-shielded.cmd shielded_latent.json
```

같은 조건으로 inversion 공격을 측정하려면 다음처럼 실행합니다.

```powershell
.\attacker-inversion-demo.cmd --public-model models\public_v4\artist_y_public_encoder.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --public-dropout 0.65 --public-noise 1.0 --output attack_results\decoder_inversion_v4_dropout65_noise1_results.json
```

V4는 보안 문제를 완전히 해결하지는 않지만, external cryptography 없이 공격
비용을 높이는 실험입니다.

## V5 Split Brain

V5는 하나의 `PicassoBrain`을 학습한 뒤 export 단계에서 공개 fragment와 비공개
fragment로 나눕니다.

```text
Artist Y public brain fragment -> ambiguous latent
Artist X private brain fragment -> text reconstruction
```

실행:

```powershell
.\artist-y-brain-v5.cmd encode "My Secret" -o brain_v5_latent.json
.\artist-x-brain-v5.cmd decode brain_v5_latent.json
```

측정된 V5b 결과:

```text
normal exact match: 0.9577
1000-message encode+decode speed: ~0.138 ms/message
1000-pair inversion exact match: 0.7525
```

## V6 UltraDrop Strong Mode

V6는 split-brain latent 폭과 runtime masking을 키워 inversion attack 저항성을 더
강하게 보여주는 모드입니다. 정상 복원 정확도와 공격 저항성 사이의 trade-off가
있습니다.

```powershell
.\artist-y-brain-v6.cmd encode "My Secret" -o brain_v6_latent.json
.\artist-x-brain-v6.cmd decode brain_v6_latent.json
```

측정된 V6 UltraDrop 결과:

```text
normal validation exact match: 0.9415
1000-message encode+decode speed: ~0.140 ms/message
1000-pair inversion exact match: 0.6750
500-pair inversion exact match: 0.4725
```

정상 복원 안정성을 강조할 때는 V5b, 공격 저항성을 강조할 때는 V6를 사용하세요.

## 3단계 Surrogate Brain 공격

V6 public fragment를 대상으로 chosen plaintext pair를 모으고 surrogate private
fragment를 학습한 뒤 평가하는 분리형 실험입니다.

```powershell
python tools\attacks\attack_brain_step1_collect_pairs.py --pairs 1000 --output attack_results\brain_attack_pairs.pt
python tools\attacks\attack_brain_step2_train_surrogate.py --input attack_results\brain_attack_pairs.pt --epochs 80
python tools\attacks\attack_brain_step3_evaluate_surrogate.py --pairs-file attack_results\brain_attack_pairs.pt
```

중간 산출물 `.pt` 파일은 재생성 가능한 실험 파일이므로 Git에 포함하지 않습니다.

## Legacy Latent CLI

초기 latent encoder/decoder 스크립트는 `tools/cli` 아래에 보관되어 있습니다.

```powershell
python tools\cli\encode_latent.py --text "My Secret" --output latent_demo.json
python tools\cli\decode_latent.py --input latent_demo.json
```

일반 사용은 `picasso.cmd` 또는 역할 분리 `.cmd` 사용을 권장합니다.

## 학습과 Export

학습 스크립트는 `ver.3/train` 아래에 있습니다.

기본 public/private compact model 분리:

```powershell
python ver.3\train\split_public_private_models.py
```

V5 brain fragment export:

```powershell
python ver.3\train\export_brain_v5.py
```

V6 brain fragment export:

```powershell
python ver.3\train\export_brain_v6.py
```

학습 checkpoint 디렉터리는 private decoder를 재구성할 수 있으므로 공유하지 마세요.

```text
models/private/
models/private_v4/
models/private_v5b/
models/private_v6_ultradrop/
ver.3/train/wikitext_checkpoints_final/
```

## 샘플과 결과 파일

- `samples/`: 예시 latent JSON입니다. 실행법 확인이나 파일 구조 설명에 사용합니다.
- `attack_results/`: 논문/발표에 사용한 측정 결과와 그래프입니다.
- `assets/`: 포스터, 이미지 자료입니다.
- `docs/`: 한국어 발표 대본과 공격 보고서입니다.

새로 생성되는 `picasso_latent.json`, `captured_latent.json`, `brain_v5_latent.json`,
`brain_v6_latent.json`, `shielded_latent.json` 등은 `.gitignore`에 등록되어 있습니다.

## 문제 해결

`Conda executable not found`가 나오면 `.cmd` 파일의 Conda 경로를 확인하세요.

모델 파일을 찾지 못하면 `models/` 아래 경로가 위의 "모델 파일 준비" 섹션과
일치하는지 확인하세요.

CUDA가 없으면 CPU로 실행됩니다. 학습과 inversion attack 실험은 CPU에서 매우
느릴 수 있습니다.

PowerShell에서 한글 파일명 표시가 깨져도 Git 내부 상태에는 영향이 없을 수
있습니다. 실제 파일 존재 여부는 `Get-ChildItem assets`, `Get-ChildItem docs`,
`Get-ChildItem samples`로 확인하세요.

## 참고 자료

- RFNNS: Robust Fixed Neural Network Steganography with Popular Deep Generative Models (https://arxiv.org/pdf/2505.04116)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/abs/1810.04805)
- Dataset: https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1
