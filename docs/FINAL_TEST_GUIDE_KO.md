# 최종 테스트 가이드

이 문서는 프로젝트 제출/발표 전에 핵심 기능만 빠르게 확인하기 위한 최종 점검용
가이드입니다. 전체 사용법은 [README.md](../README.md)를 참고하세요.

## 0. 테스트 전 준비

프로젝트 루트에서 PowerShell을 엽니다.

```powershell
cd X:\Dev\GitHub\Picasso-Protocol-1.0
conda activate picasso-protocol
```

CPU 환경에서도 테스트할 수 있습니다. 단, 현재 환경에 PyTorch CPU 버전과 필요한
라이브러리가 설치되어 있어야 합니다.

모델 파일이 다음 위치에 있는지 확인합니다.

```powershell
Test-Path models\public\artist_y_public_encoder.pt
Test-Path models\private\artist_x_private_decoder.pt
Test-Path models\public_v5b\artist_y_public_brain_fragment.pt
Test-Path models\private_v5b\artist_x_private_brain_fragment.pt
Test-Path models\public_v6_ultradrop\artist_y_public_brain_fragment.pt
Test-Path models\private_v6_ultradrop\artist_x_private_brain_fragment.pt
```

모든 명령이 `True`를 출력해야 합니다. `False`가 있으면 모델 파일을 먼저
다운로드하거나 올바른 폴더로 옮기세요.

## 1. 코드 구문 검사

```powershell
python -m compileall -q picasso_protocol tools toWebPage ver.3\train
```

기대 결과:

- 아무 출력 없이 종료되면 통과입니다.
- 에러가 나오면 해당 파일의 Python 문법 또는 import 경로를 확인하세요.

## 2. 기본 CLI 도움말 확인

```powershell
python tools\cli\picasso.py --help
```

기대 결과:

```text
{encode,decode,status,verify-public}
```

위 subcommand 목록이 보이면 CLI 진입점이 정상입니다.

## 3. 모델 상태 확인

```powershell
.\picasso.cmd status
.\picasso.cmd verify-public
```

기대 결과:

- `status`에서 device, public encoder, private decoder, latent shape가 출력됩니다.
- `verify-public`에서 공개 Artist Y 파일에 decoder가 없다는 안전 검사 통과 메시지가 출력됩니다.

## 4. 기본 Encode/Decode 왕복 테스트

```powershell
.\picasso.cmd encode "My Secret" -o final_test_latent.json
.\picasso.cmd decode final_test_latent.json
```

기대 결과:

- 첫 번째 명령은 `final_test_latent.json`을 생성합니다.
- 두 번째 명령은 `My Secret` 또는 정상 복원된 문장을 출력합니다.

테스트 후 임시 파일을 지웁니다.

```powershell
Remove-Item final_test_latent.json
```

## 5. 역할 분리 테스트

공개 Artist Y로 latent를 만들고, 비공개 Artist X로 복원합니다.

```powershell
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json
.\artist-x-private.cmd decode captured_latent.json
```

기대 결과:

- `artist-y-public.cmd`는 latent shape와 저장 경로를 출력합니다.
- `artist-x-private.cmd`는 복원된 텍스트를 출력합니다.

정리:

```powershell
Remove-Item captured_latent.json
```

## 6. 공격 데모 최소 테스트

```powershell
.\artist-y-public.cmd encode "My Secret" -o captured_latent.json
.\attacker-demo.cmd captured_latent.json --known-text "My Secret"
```

기대 결과:

- latent JSON 내부에 원문이 그대로 들어 있지 않다는 확인이 출력됩니다.
- public server decode 시도가 막히거나 서버 미실행 상태로 표시됩니다.
- public model을 private decoder처럼 쓰려는 시도가 차단됩니다.

정리:

```powershell
Remove-Item captured_latent.json
```

## 7. V5 Brain Fragment 테스트

```powershell
.\artist-y-brain-v5.cmd encode "My Secret" -o brain_v5_latent.json
.\artist-x-brain-v5.cmd decode brain_v5_latent.json
```

기대 결과:

- V5 public brain fragment가 latent를 생성합니다.
- V5 private brain fragment가 텍스트를 복원합니다.

정리:

```powershell
Remove-Item brain_v5_latent.json
```

## 8. V6 UltraDrop 테스트

```powershell
.\artist-y-brain-v6.cmd encode "My Secret" -o brain_v6_latent.json
.\artist-x-brain-v6.cmd decode brain_v6_latent.json
```

기대 결과:

- V6 UltraDrop public fragment가 latent를 생성합니다.
- V6 private fragment가 텍스트를 복원합니다.

정리:

```powershell
Remove-Item brain_v6_latent.json
```

## 9. Web API 서버 테스트

공개 서버를 실행합니다.

```powershell
.\server-public.cmd
```

다른 PowerShell 창에서 health check를 실행합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

기대 결과:

- 서버 상태 JSON이 출력됩니다.
- public 서버에서는 private decode가 비활성화되어 있어야 합니다.

브라우저 데모는 다음 파일을 엽니다.

```text
toWebPage/index.html
```

서버 창은 테스트가 끝나면 `Ctrl+C`로 종료합니다.

## 10. 최종 제출 전 확인

Git 상태를 확인합니다.

```powershell
git status --short
```

확인할 점:

- `__pycache__` 또는 `*.pyc`가 없어야 합니다.
- `models/` 안의 `.pt` 모델 파일은 Git에 올라가지 않아야 합니다.
- 임시 테스트 파일인 `final_test_latent.json`, `captured_latent.json`,
  `brain_v5_latent.json`, `brain_v6_latent.json`이 남아 있지 않아야 합니다.

최종 테스트에서 최소한 아래 항목은 통과해야 합니다.

```text
[ ] Python compileall 통과
[ ] picasso.py --help 출력 확인
[ ] picasso.cmd status 통과
[ ] picasso.cmd verify-public 통과
[ ] 기본 encode/decode 왕복 통과
[ ] 역할 분리 encode/decode 통과
[ ] attacker-demo 최소 테스트 통과
[ ] V5 brain fragment 왕복 통과
[ ] V6 UltraDrop 왕복 통과
[ ] Web API health check 통과
```
