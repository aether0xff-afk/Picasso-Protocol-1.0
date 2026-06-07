# 취약점 및 완화 실험 보고서

## 1. 발견된 취약점

현재 Picasso Protocol은 역할을 분리한다.

- Artist Y: 공개 encoder
- Artist X: 비공개 decoder

하지만 공개 Artist Y가 임의의 평문을 latent vector로 바꿔주기 때문에, 공격자는 `평문-latent` 쌍을 많이 모아 대체 decoder를 학습할 수 있다. 이 공격은 private Artist X 파일을 직접 읽지 않아도 가능하다.

실험 명령:

```powershell
.\attacker-inversion-demo.cmd --sizes 100 500 1000 --epochs 100
```

기존 모델 공격 결과:

| 데이터 수 | Train Loss | Test Loss | Token Accuracy | Exact Match Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.011171 | 0.028510 | 0.9987 | 0.8950 |
| 500 | 0.002499 | 0.009441 | 0.9997 | 0.9750 |
| 1000 | 0.001137 | 0.005539 | 0.9998 | 0.9875 |

즉, latent가 그대로 노출되면 공격자가 private decoder와 유사한 기능을 학습할 수 있다.

## 2. 알고리즘 내부 완화: Stochastic Latent Shielding V4

외부 암호화 알고리즘을 붙이지 않고, 기존 neural protocol 내부에서만 완화했다.

적용한 방법:

- 학습 시 Artist X decoder가 latent dropout/noise에 견디도록 훈련
- 공개 Artist Y 출력 시 latent 값 일부를 확률적으로 masking
- Gaussian noise 추가
- 모델 구조는 기존 sequence encoder/decoder를 유지

채택한 shield 조건:

```text
latent_dropout = 0.65
latent_noise = 1.0
```

정상 복호화 성능:

| 모델 | 조건 | Exact Match | 속도 |
| --- | --- | ---: | ---: |
| V4 shielded | dropout 0.65 + noise 1.0 | 약 0.976 | 약 0.12 ms/문장 |

## 3. 완화 후 공격 결과

실험 명령:

```powershell
.\attacker-inversion-demo.cmd --public-model models\public_v4\artist_y_public_encoder.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --public-dropout 0.65 --public-noise 1.0 --output attack_results\decoder_inversion_v4_dropout65_noise1_results.json
```

결과:

| 데이터 수 | 기존 Exact Match | V4 Shielded Exact Match | 감소 |
| ---: | ---: | ---: | ---: |
| 100 | 0.8950 | 0.4000 | -0.4950 |
| 500 | 0.9750 | 0.8050 | -0.1700 |
| 1000 | 0.9875 | 0.9000 | -0.0875 |

## 4. 결론

V4 shielded 방식은 기존 알고리즘을 외부 암호화로 대체하지 않고, 내부 neural latent 생성 방식을 강화한다. 속도 저하는 거의 없고, 특히 적은 공격 데이터에서는 inversion 성공률을 크게 낮춘다.

단, 충분한 chosen plaintext 데이터가 있으면 공격자는 여전히 대체 decoder를 학습할 수 있다. 따라서 현재 단계의 결론은 다음과 같다.

```text
완전한 암호 시스템은 아니지만,
기존 prototype보다 안전성 공격 시연과 완화 시연을 모두 보여줄 수 있는 개선 버전이다.
```

시연용 명령:

```powershell
.\artist-y-shielded.cmd "My Secret" -o shielded_latent.json
.\artist-x-shielded.cmd shielded_latent.json
```

## 5. V5: Split Brain Fragment

V5는 X와 Y를 따로 학습한 encoder/decoder처럼 두지 않고, 하나의 `PicassoBrain`을 먼저 학습한 뒤 public fragment와 private fragment로 절단한다.

구조:

```text
PicassoBrain
 ├─ Artist Y public brain fragment
 │   └─ text -> ambiguous brain latent
 └─ Artist X private brain fragment
     └─ ambiguous brain latent -> text
```

V5b 채택 조건:

```text
latent_dim = 256
private_dim = 128
latent_dropout = 0.85
latent_noise = 1.0
```

정상 복호화:

| 모델 | Exact Match | Token Accuracy |
| --- | ---: | ---: |
| V5b Split Brain | 0.9577 | 0.9995 |

추론 속도:

| 모델 | 1000문장 encode+decode | 문장당 시간 |
| --- | ---: | ---: |
| V5b Split Brain | 0.1381초 | 약 0.138 ms |

공격 실험:

```powershell
.\attacker-brain-v5-inversion-demo.cmd --public-model models\public_v5b\artist_y_public_brain_fragment.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --output attack_results\brain_v5b_inversion_results.json
```

결과:

| 데이터 수 | 기존 Exact Match | V4 Shielded | V5b Split Brain |
| ---: | ---: | ---: | ---: |
| 100 | 0.8950 | 0.4000 | 0.1700 |
| 500 | 0.9750 | 0.8050 | 0.6175 |
| 1000 | 0.9875 | 0.9000 | 0.7525 |

해석:

- V5b는 X와 Y가 같은 brain의 절단 조각이라는 컨셉에 더 맞다.
- 공격자가 public fragment만으로 대체 private fragment를 학습할 수는 있지만, 학습 안정성이 크게 떨어진다.
- 1000쌍 공격 기준으로 V4보다 더 낮은 복원률을 보였다.
- latent 폭은 커졌지만 추론 속도는 여전히 매우 빠르다.
- 완전한 암호학적 차단은 아니지만, 현재 레포의 neural protocol 내부 개선 중 가장 강한 버전이다.

V5b 시연:

```powershell
.\artist-y-brain-v5.cmd encode "My Secret" -o brain_v5_latent.json
.\artist-x-brain-v5.cmd decode brain_v5_latent.json
```

## 6. V6: UltraDrop Split Brain

V6에서는 V5b보다 public latent 폭을 넓히고, dropout을 더 강하게 올렸다. 목적은 Artist X가 복원할 여유는 남기되, 공격자가 모으는 `평문-latent` 쌍의 안정성을 더 낮추는 것이다.

V6 ultradrop 조건:

```text
latent_dim = 512
private_dim = 192
latent_dropout = 0.93
latent_noise = 1.0
```

정상 복호화:

| 모델 | Validation Exact | 1000문장 샘플 Exact | 문장당 시간 |
| --- | ---: | ---: | ---: |
| V6 UltraDrop | 0.9415 | 0.9320 | 약 0.140 ms |

공격 실험:

```powershell
.\attacker-brain-v5-inversion-demo.cmd --public-model models\public_v6_ultradrop\artist_y_public_brain_fragment.pt --sizes 100 500 1000 --epochs 60 --test-size 400 --output attack_results\brain_v6_ultradrop_inversion_results.json
```

결과:

| 데이터 수 | 기존 Exact Match | V5b Split Brain | V6 UltraDrop |
| ---: | ---: | ---: | ---: |
| 100 | 0.8950 | 0.1700 | 0.0500 |
| 500 | 0.9750 | 0.6175 | 0.4725 |
| 1000 | 0.9875 | 0.7525 | 0.6750 |

해석:

- V6는 현재 레포에서 가장 강한 neural-only 방어 버전이다.
- 정상 복호화 정확도는 V5b보다 낮아졌지만, 속도는 거의 유지됐다.
- 공격자의 surrogate private fragment 학습은 더 불안정해졌다.
- 특히 500쌍 이하 공격에서는 복원 성공률이 크게 떨어진다.

V6 시연:

```powershell
.\artist-y-brain-v6.cmd encode "My Secret" -o brain_v6_latent.json
.\artist-x-brain-v6.cmd decode brain_v6_latent.json
```
