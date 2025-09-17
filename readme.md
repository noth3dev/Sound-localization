# MUSIC 기반 소리 위치 추정 + 빔포밍 분리 (파이썬 스크립트)

안녕! 이 레포는 **여러 음원(mp3)** 을 시뮬레이션된 **4-마이크 ULA(Uniform Linear Array)** 로 수신한 뒤,
MUSIC 알고리즘으로 DoA(도착각)를 추정하고, 추정된 각도로 **Delay-and-Sum 빔포밍**을 돌려 각 소리별 분리된 WAV 파일을 만드는 간단한 파이프라인이야.
사용법은 단순 — `sound/` 폴더에 `1.mp3, 2.mp3, 3.mp3, 4.mp3` 넣고 실행하면 `output/`에 `separated_1.wav` 같은 파일이 생겨.

---

## 핵심 기능(한줄 요약)

* MP3 파일 4개를 불러와서 (librosa 또는 pydub+ffmpeg 필요)
* 마이크 어레이 수신 신호를 시뮬레이션
* STFT 기반 공분산 누적으로 MUSIC 스펙트럼 계산 → Top-N DoA 추정
* 각 DoA에 대해 시간영역 빔포밍(블록 단위) 수행 → 분리된 WAV 출력

---

## 요구사항 (Dependencies)

* Python 3.8+
* 필수 라이브러리:

  * `numpy`
  * `scipy`
* MP3 로딩 (하나는 필요):

  * `librosa` (권장: `pip install librosa`)
    또는
  * `pydub` + `ffmpeg` (`pip install pydub` + 시스템에 ffmpeg 설치)
* (선택) `soundfile` 등은 필요 없음 — 출력은 `scipy.io.wavfile` 사용

예시 설치:

```bash
pip install numpy scipy librosa
# 또는
pip install numpy scipy pydub
# 그리고 시스템에 ffmpeg 설치 (OS 패키지 매니저나 ffmpeg.org)
```

---

## 빠른 시작

1. 레포/스크립트가 있는 폴더에 `sound/` 폴더 만들기
2. `sound/` 안에 `1.mp3`, `2.mp3`, `3.mp3`, `4.mp3` 넣기
3. 스크립트 실행:

```bash
python3 your_script_name.py
```

4. 실행이 끝나면 `output/` 폴더에 `separated_1.wav`, `separated_2.wav`, ... 생성

스크립트는 파일이 없으면 경고를 띄우고 종료하니, 파일명/경로 맞춰 넣어줘.

---

## 주요 파라미터 (스크립트 상단)

직접 조정해서 실험해볼 수 있는 값들:

* `SAMPLE_RATE` — 샘플링레이트 (기본 48000)
* `BLOCK_SIZE` — 블록 길이(FFT 길이), 기본 4096
* `OVERLAP` — 블록 오버랩 비율 (현재 0.0)
* `CHANNELS` — 시뮬레이션 마이크 수 (기본 4)
* `MIC_SPACING_M` — 마이크 간격(m) (기본 0.035)
* `SPEED_OF_SOUND` — 소리 속도 (m/s)
* `N_SOURCES_EST` — MUSIC으로 추정할 소스 개수
* `AZIMUTH_GRID_DEG` — 각도 검색 그리드 (기본 -90\~90 deg, 0.5도 스텝)
* `F_MIN`, `F_MAX` — MUSIC에 포함할 주파수 밴드 (Hz)
* `COV_ALPHA` — 공분산 누적 시 지수평활 계수 (0\~1)
* `SOURCE_FILES`, `SOURCE_ANGLES_DEG` — 시뮬레이션용 입력 파일명 및 각도 (테스트/시뮬 전용)
* `OUTPUT_DIR` — 분리 파일 출력 폴더

---

## 내부 동작(짧게)

1. MP3를 불러와 RMS 정규화
2. `simulate_array_from_sources()`로 단일 채널 음원들을 여러 마이크 신호로 시뮬레이션 (지연과 위상 적용)
3. 블록 단위로 FFT를 계산하여 각 주파수 밴드에 대한 공분산 행렬 업데이트 (`COV_ALPHA` 사용)
4. 각 주파수별 공분산 행렬을 모아 MUSIC 스펙트럼 계산 → 각도 스코어 합산 → 정규화
5. `find_top_k_angles()`로 Top-N 각도 추출
6. `beamform_full()`에서 추출된 각도로 블록별 빔포밍(Delay-and-Sum 유사 연산) 수행
7. 출력 정규화 후 WAV로 저장

---

## 파일 구조 (예시)

```
repo/
├─ sound/
│  ├─ 1.mp3
│  ├─ 2.mp3
│  ├─ 3.mp3
│  └─ 4.mp3
├─ your_script.py      # 제공한 스크립트
└─ output/             # 실행 후 여기에 separated_*.wav 생성
```

---

## 사용 팁 & 트러블슈팅

* MP3 로딩 에러가 나면 `librosa` 또는 `pydub`+`ffmpeg` 설치 여부를 확인해. 에러 메시지에 설치 안내가 나와 있어.
* MUSIC 성능은 마이크 수, 배열 길이, 소스 간각도 차, SNR, 주파수 밴드(F\_MIN/F\_MAX)에 민감해.

  * 저주파(예: <500Hz)는 위상 분해능이 떨어져서 제외하는 편이 좋아.
  * 블록 크기(`BLOCK_SIZE`)를 늘리면 주파수분해능은 좋아지지만 시간분해능/메모리 부담이 증가함.
* 시뮬레이션은 사실적인 룸 반사/잡음 모델을 포함하지 않음 — 실녹음으로 실험하려면 레벨 정렬/동기화/잡음처리가 필요.
* `COV_ALPHA`를 1.0에 가깝게 하면 이전 블록의 공분산 영향력이 커짐(천천히 적응). 반대로 작게 하면 빠르게 적응.

---

## 제한 사항 (알아둘 점)

* 현재 스크립트는 **시뮬레이션 기반**이며 실제 마이크 하드웨어 입력을 직접 다루지 않음.
* 빔포밍은 단순한 Delay-and-Sum 방식(주파수 도메인에서 간단히 합산)이며, 강력한 분리를 원하면 MVDR, GSC, Wiener 필터 등의 방법 고려 필요.
* MUSIC은 정교하지만 공분산 추정 품질, 소스 상호간섭, 동적 상황(이동원) 등에서 성능이 떨어질 수 있음.

---

## 개선 아이디어 (확장 제안)

* 실제 마이크 녹음 입력(다중채널 WAV) 처리 추가
* MVDR 또는 지방화된 빔포머 도입하여 간섭 억제 성능 개선
* 잡음/잔향(리버브) 모델 추가해 현실성 높이기
* SRP-PHAT 등으로 소스 수 추정 모듈 추가 (현재는 `N_SOURCES_EST` 고정)
* GUI 또는 실시간 스트리밍 처리 (WS/Socket 기반)로 즉시 피드백

---

## 라이선스

원하는 라이선스 명시하거나 자유롭게 사용해. (예: MIT)

---

필요하면 이 README를 깃허브용 `README.md` 형식으로 만들어주거나, 한국어/영어 두 가지 버전으로도 정리해줄게.
또는 톤을 바꿔(공식·친근·논문 형식 등)도 써줄 수 있음 — 어떻게 쓸까?
