# Picasso Protocol 시연 스크립트

## 목표

비대칭 암호화 느낌의 역할 분리를 보여준다.

- Artist Y: 공개키처럼 쓰는 public encoder
- Artist X: 개인키처럼 보관하는 private decoder

보안성은 완성형 암호가 아니라 prototype 수준이며, 영상에서는 취약점 발견과 완화까지 함께 보여주는 구성이 좋다.

## 1. 공개 모델 확인

```powershell
.\artist-y-public.cmd status
```

말할 내용:

> Artist Y는 누구나 받을 수 있는 공개 encoder입니다. 이 파일에는 decoder가 없기 때문에 단독으로 원문을 복원할 수 없습니다.

## 2. 공개 encoder로 latent 생성

```powershell
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json
```

말할 내용:

> 송신자는 공개 Artist Y만 사용해서 메시지를 latent JSON으로 변환합니다. 현재 단계에서는 PNG 저장 전 latent 파일을 직접 사용합니다.

## 3. 기본 공격 차단 시연

공개 서버 실행:

```powershell
.\server-public.cmd
```

다른 PowerShell에서:

```powershell
.\attacker-demo.cmd captured_latent.json --known-text "My Secret"
```

보여줄 포인트:

- latent JSON 안에 평문 문자열은 직접 보이지 않음
- public Artist Y 파일을 private decoder처럼 쓰면 차단됨
- public server의 decode 요청은 HTTP 403으로 차단됨
- 임의 decoder 가중치로는 읽을 수 있는 문장이 나오지 않음

## 4. 수신자 복호화

```powershell
.\artist-x-private.cmd status
.\artist-x-private.cmd decode captured_latent.json
```

말할 내용:

> 수신자만 가진 Artist X private decoder가 같은 latent에서 원문을 복원합니다.

## 5. 진짜 취약점: Decoder Inversion Attack

```powershell
.\attacker-inversion-demo.cmd --sizes 100 500 1000 --epochs 100
```

말할 내용:

> 단순 차단만으로 안전하다고 볼 수는 없습니다. 공격자가 공개 encoder에 자신이 고른 평문을 반복 입력하면, 평문-latent 쌍을 모아 대체 decoder를 학습할 수 있습니다.

기존 모델 결과:

```text
100 pairs  -> exact match 0.8950
500 pairs  -> exact match 0.9750
1000 pairs -> exact match 0.9875
```

## 6. 완화 버전 시연: V4 Stochastic Latent Shielding

```powershell
.\artist-y-shielded.cmd "My Secret" -o shielded_latent.json
.\artist-x-shielded.cmd shielded_latent.json
```

말할 내용:

> 외부 암호화 알고리즘을 붙이지 않고, 기존 neural latent 과정 안에서 방어를 추가했습니다. Artist X는 dropout/noise가 섞인 latent를 복원하도록 학습됐고, Artist Y는 latent 일부를 확률적으로 가립니다.

## 7. 완화 후 공격 재시도

```powershell
.\attacker-inversion-demo.cmd --public-model models\public_v4\artist_y_public_encoder.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --public-dropout 0.65 --public-noise 1.0 --output attack_results\decoder_inversion_v4_dropout65_noise1_results.json
```

비교:

```text
100 pairs  exact match: 0.8950 -> 0.4000
500 pairs  exact match: 0.9750 -> 0.8050
1000 pairs exact match: 0.9875 -> 0.9000
```

마무리 멘트:

> 완전한 암호 수준은 아니지만, 같은 알고리즘 계열 안에서 정상 복호화 속도를 유지하면서 공격 비용을 높였습니다. 이 prototype은 역할 분리, 취약점 발견, 완화 방향까지 보여주는 단계입니다.

## 8. V5 Split Brain Fragment

```powershell
.\artist-y-brain-v5.cmd status
.\artist-x-brain-v5.cmd status
.\artist-y-brain-v5.cmd encode "My Secret" -o brain_v5_latent.json
.\artist-x-brain-v5.cmd decode brain_v5_latent.json
```

말할 내용:

> V5는 X와 Y를 따로 만든 decoder/encoder가 아니라, 하나의 PicassoBrain을 학습한 뒤 앞쪽 public fragment와 뒤쪽 private fragment로 절단한 구조입니다. 즉 X와 Y는 같은 뇌의 다른 조각을 가진다는 컨셉입니다.

공격 재시도:

```powershell
.\attacker-brain-v5-inversion-demo.cmd --public-model models\public_v5b\artist_y_public_brain_fragment.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --output attack_results\brain_v5b_inversion_results.json
```

비교:

```text
기존 1000쌍 공격 exact: 0.9875
V4   1000쌍 공격 exact: 0.9000
V5b  1000쌍 공격 exact: 0.7525
```

마무리 멘트:

> 아직 완전한 암호학적 차단은 아니지만, 공격자가 대체 decoder를 학습하는 과정이 훨씬 불안정해졌습니다. V5b는 현재 구현 중 Picasso Protocol의 컨셉과 방어 성능이 가장 잘 맞는 버전입니다.

## 9. V6 UltraDrop 강력 모드

```powershell
.\artist-y-brain-v6.cmd encode "My Secret" -o brain_v6_latent.json
.\artist-x-brain-v6.cmd decode brain_v6_latent.json
```

공격 재시도:

```powershell
.\attacker-brain-v5-inversion-demo.cmd --public-model models\public_v6_ultradrop\artist_y_public_brain_fragment.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --output attack_results\brain_v6_ultradrop_inversion_results.json
```

비교:

```text
기존 1000쌍 공격 exact: 0.9875
V5b  1000쌍 공격 exact: 0.7525
V6   1000쌍 공격 exact: 0.6750

기존 500쌍 공격 exact: 0.9750
V5b  500쌍 공격 exact: 0.6175
V6   500쌍 공격 exact: 0.4725
```

말할 내용:

> V6는 정상 정확도를 조금 희생해서 공격자 학습을 더 불안정하게 만든 강력 모드입니다. 외부 암호화 알고리즘을 붙이지 않고, 같은 split brain 계열 안에서 방어 강도를 올렸습니다.

## 배포 규칙

공개 가능:

```text
artist-y-public.cmd
artist_y_public.py
models/public/
models/public_v4/
models/public_v5b/
models/public_v6_ultradrop/
```

비공개 유지:

```text
artist-x-private.cmd
artist-x-shielded.cmd
artist_x_private.py
models/private/
models/private_v4/
models/private_v5b/
models/private_v6_ultradrop/
ver.3/train/wikitext_checkpoints_final/
ver.3/train/wikitext_checkpoints_shielded_v4/
ver.3/train/brain_v5b/
ver.3/train/brain_v6_ultradrop/
```
