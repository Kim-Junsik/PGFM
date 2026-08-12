# PGFM — Propagated Growth Flow Matching for Combinatorial Perturbation

조합 섭동(A+B) 하의 단일세포 발현 반응을 예측한다. 대조군 세포를 잠재 공간으로 인코딩한 뒤,
**섭동별 벡터장과 그 Lie 괄호**로 구성된 속도장을 따라 수송하여 섭동 상태의 세포를 만든다.
학습에서 함께 본 적 없는 조합으로 일반화하는 것이 목표다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.pdf](docs/ARCHITECTURE.pdf) | 모델 전체 (그림 6개, P-CAB / E-RCA 상세 포함) — 외부 전달용 |
| [docs/LieCFM-method.pdf](docs/LieCFM-method.pdf) | Method 섹션 초안 (ICLR 형식) — **갱신 필요** |
| [docs/HANDOFF-LT-VAE-LieCFM.pdf](docs/HANDOFF-LT-VAE-LieCFM.pdf) | 원 명세서. 데이터·split·평가 정의와 **폐기된 방향**의 기록 |

`docs/ARCHITECTURE.pdf` 는 `scripts/make_architecture_pdf.py` 로 재생성한다.

---

## 현재 구현 상태

```
코드          2,900+ 줄  (src/ 22개 · scripts/ 6개 · tests/ 1개)
테스트        26개 통과   (구조 보장 14 + split 동일성 12)
백본          4종 모두 end-to-end 실행 확인
목표선        미달
```

| 항목 | 상태 |
|---|---|
| 데이터 · split · 평가 · 기준선 | 완료, 검증됨 |
| 백본 4종 (mlp / transformer / scvi / pcab) | 완료, 1에폭 실행 확인 |
| LieCFM 동역학 + 구조 보장 4가지 | 완료, 14개 테스트 |
| split 동일성 계약 | 완료, 12개 테스트 |
| P-CAB / E-RCA | 완료, 최적화됨 (fwd+bwd 84 → 31 ms) |
| cell-eval 어댑터 | 작성 완료, **첫 실행 검증 미완** |
| 인코더 / 디코더 독립 축 분리 | 미착수 (선택) |
| 목표선 도달 | **미달** — `resid_R2` 가 아직 음수 |

## 검증된 측정값 — 재도출 불필요

| 항목 | 값 | 확인 방법 |
|---|---|---|
| split (5 fold) | **scDFM 원본 파일을 직접 읽음** (복사본 없음) | `pytest tests/test_splits.py` |
| 유전자 공간 | 3,000 HVG + 표적 101 = **3,074** | 이전 파이프라인과 완전 일치 |
| 평가 공간 포함 | 10 fold 중 9개가 1000/1000 | HVG 3,000 을 쓸 수 있는 근거 |
| KEGG 마스크 | K = 176 주석 + 101 자유 = **277**, 밀도 0.879 % | `src/data/kegg.py` |
| 섭동 표적 커버리지 | **47 / 101** | KEGG 가 발생 TF 를 담지 않음 |
| 데이터 0 비율 | **41.2 %** | hurdle 디코더가 필요한 이유 |
| 노이즈 바닥 보정 | λ = 1.000 (n_eff 16~375) | 대조군 4그룹 귀무분포 |

## 목표선

`results/baselines_*.json` 이 최신이다. 아래는 `ridge_alpha=1.0` 기준.

| split | edist_rel ↓ | resid_R2 ↑ |
|---|---|---|
| additive | **0.2856** | **0.3953** |
| combinations | **0.2883** | **0.1853** |

`ridge_alpha` 는 잠정 고정값이다. 테스트셋을 보고 고르는 것은 정당하지 않으므로,
최종 수치를 인용하기 전에 train double 안에서 내부 CV 로 선택해야 한다.

## 원 명세서에서 대체된 항목

`HANDOFF-LT-VAE-LieCFM.pdf` 4장의 아래 세 가지는 더 이상 유효하지 않다.
나머지(데이터 구조, split 정의, 평가 지표, 폐기된 방향)는 여전히 근거다.

| 항목 | 명세서 | 현재 |
|---|---|---|
| 마스크 결합 | `softmax(QK/√d + λ·M_GRN)`, M_GRN 은 0 / −inf | `tanh(M_prior + α·M_residual)` 곱셈 게이트 |
| prior 소스 | MSigDB / Reactome / GO 중 미정 | KEGG 176 + 자유 토큰 101 |
| 인코더 | P-CAB 고정 | 설정 축 (mlp / transformer / scvi / pcab) |

## 이전 GRN 스크립트 (`legacy/`)

재사용 전에 반드시 고쳐야 할 것 — 전체 목록은 `legacy/README.md`.

1. **[치명]** 산출물이 vocab 정렬 N×N 유전자–유전자 행렬. P-CAB 이 필요한 것은 K×G
2. **[치명]** combinations split 누수 — held-out single 23개를 무조건 보존
3. **[차단]** 삭제된 코드베이스 참조 (`src.data_process.data` 등)
4. **[재현성]** `grnboost2()` 에 `seed` 미전달

---

# 실행

## 0. 준비

한 번만 하면 된다. 데이터 캐시(299 MB)를 만들고 split 을 검증한다.

```bash
python scripts/build_data.py
```

`assets/norman_modeled.h5ad` 가 이미 있으면 건너뛴다. 다시 만들려면 `--force`.
split 만 확인하려면 `--validate-only`.

기대 출력 — 이 네 줄이 맞아야 이후가 의미를 갖는다.

```
additive == scDFM reference        : True
combinations == derivation of it   : True
targets missing from var           : 0
modelled gene space                : 3,074
```

## 1. 기준선 — 넘어야 할 선

```bash
python scripts/run_baselines.py
python scripts/run_baselines.py --set split.method=combinations
```

10초 정도 걸린다. 현재 목표선:

| split | edist_rel ↓ | resid_R2 ↑ |
|---|---|---|
| additive (ridge_additive) | **0.2856** | **0.3953** |
| combinations (ridge_additive) | **0.2883** | **0.1853** |

`ridge_alpha` 는 1.0 으로 고정해 두었다. 테스트셋을 보고 고르는 것은 정당하지 않으므로,
최종 수치를 인용하기 전에 train double 안에서 내부 CV 로 선택해야 한다.

## 2. 학습

```bash
python scripts/train.py
```

기본값은 `mlp` 백본 + `hurdle` 디코더 + `commutator` 상호작용, additive split fold 0.
결과는 `results/runs/<tag>/` 에 `train.log`, `checkpoint.pt`, `results.json` 으로 남는다.

### 설정 바꾸기

모든 값은 점 표기로 덮어쓴다. 없는 키를 쓰면 **즉시 예외**가 난다(오타가 조용히
기본값으로 도는 것을 막기 위해).

```bash
# 백본 교체
python scripts/train.py --set model.backbone=pcab
python scripts/train.py --set model.backbone=transformer train.batch_size=64
python scripts/train.py --set model.backbone=scvi model.decoder_head=zinb

# 상호작용 3-way (2단계 판정)
python scripts/train.py --tag s2_additive   --set model.interaction=additive
python scripts/train.py --tag s2_commutator --set model.interaction=commutator
python scripts/train.py --tag s2_free       --set model.interaction=free_mlp

# 생성원 형태
python scripts/train.py --set model.generator=affine

# split
python scripts/train.py --set split.method=combinations split.fold=2

# 에폭
python scripts/train.py --set train.stage1_epochs=60 train.stage2_epochs=80
```

### 빠른 확인용

`max_steps_per_epoch` 를 두면 한 에폭이 그 스텝 수에서 끊긴다. 설정이 끝까지
도는지 보는 용도이고, 성능 수치로 인용하면 안 된다.

```bash
python scripts/train.py --tag smoke --set train.stage1_epochs=1 train.stage2_epochs=1 \
    train.max_steps_per_epoch=40 eval.n_gen_cells=64
```

### 배치 크기 기준

RTX 3050 8.6 GB 에서 측정한 값이다.

| 백본 | 권장 배치 | forward+backward |
|---|---|---|
| `mlp` | 256 | 빠름 |
| `scvi` | 256 | 빠름 |
| `transformer` | 64 | 중간 |
| `pcab` | 32~64 | 31 ms (배치 32) |

`pcab` 의 어텐션 텐서는 `[B, 277, 3074]` 라 배치 32 에서 27 M 원소다. 전체 유전자
(19,264)로 가면 170 M 이 되므로 그때는 배치를 더 줄이거나 유전자 축 청킹이 필요하다.

## 3. 결과 읽기

```
stage1 epoch  20/20  recon 0.08  gate_bce 0.51  kl 0.99
stage2 epoch  30/30  [all] loss 0.29  fm 0.13
...
  resid_R2_pooled              -0.88
  edist_rel                     0.59
  edist_rel_autoencoder_floor   1.01
```

**세 번째 줄을 먼저 본다.** `edist_rel` 이 `floor` 보다 낮아야 수송이 실제로
일어난 것이다. 둘이 같으면 flow 가 아무 일도 하지 않고 오토인코더 성능만 보고
있는 것이며, 그 상태에서는 `interaction` 을 바꿔도 차이가 나타나지 않는다.

`resid_R2` 는 평균 기반이라 디코더의 분산 문제와 무관하다. 두 지표가 갈릴 때는
이쪽이 더 신뢰할 만하다.

## 4. 외부 비교 (cell-eval)

```bash
python scripts/run_celleval.py results/runs/<tag> --profile minimal
```

cell-eval 0.5.42 는 `.env-celleval/python.exe` (Python 3.11) 에 설치되어 있고,
의존성이 이 환경과 충돌하므로 별도 프로세스로 호출된다. `--export-only` 를 주면
`pred.h5ad` / `real.h5ad` 만 쓰고 멈춘다.

## 5. Ablation

```bash
sh scripts/run_stage2_ablation.sh
```

`generator` 2종 × `interaction` 3종 = 6개를 순차 실행한다. 판정 순서:

1. `commutator` vs `additive` — Lie 괄호가 기여하는가
2. `commutator` vs `free_mlp` — 대수적 형태가 자유 MLP 보다 나은가
3. `affine` vs `neural_field` — 비선형 벡터장이 필요한가

3번에서 `affine` 이 이기면 "고차 동역학이 불필요하다" 는 발견이 되고 논문의
프레이밍이 바뀐다. 지는 결과가 아니라 다른 결과다.

## 6. 테스트

무엇을 건드리든 돌린다. 1분 안에 끝난다 (구조 테스트가 명시적 야코비안을 만들어 대조하므로 대부분의 시간을 차지한다).

```bash
python -m pytest tests/ -q
```

26개가 통과해야 한다. 둘 다 정확도 테스트가 아니다 — **실패는 학습 부족이 아니라
계약 위반**을 뜻한다.

### `tests/test_structure.py` (14개) — 학습 없이 성립해야 하는 성질

대조군 불변 `v(z,t,{}) = 0`, 단일 직접 감독 `v(z,t,{a}) = u_a`, 가법 정확 복원
`Λ=0 → Σu_a`, 순열 불변, 괄호 반대칭, 그리고 jvp 교환자가 명시적 야코비안과
1e-8 까지 일치하는지. 실패하면 속도장 합성이 틀린 것이다.

### `tests/test_splits.py` (12개) — split 동일성

split 이 `data/norman/split_results.pkl` 과 **같은 리스트, 같은 순서**인지 확인한다.
"동등"이 아니라 축자 일치다. 유전자 공간은 자유롭게 바꿔도 되지만(n_hvg 1000 / 3000 /
5000 / 전체에서 검사) **split 은 절대 움직이면 안 된다.**

split 은 조용히 깨지는 종류다 — 이름 변경, 재정렬, 캐시 복사본, 유전자 공간 변경 중
무엇으로도 어긋날 수 있고 학습 루프는 아무것도 눈치채지 못한다. 그러면 "저자 split 에서
저자 방법이 이겼다"는 반박을 막는다는 목적 자체가 사라지는데, 그 사실을 심사 중에
알게 된다. 그래서 주석이 아니라 테스트로 고정했다.

`assets/splits_*.pkl` 같은 캐시본이 다시 생기면 `test_no_cached_split_artifact_exists`
가 실패한다. 캐시는 원본과 어긋나도 아무도 모르기 때문에, 의도적으로 금지한다.

## 알아두면 좋은 함정

- **`--set model.interaction=none` 은 쓰지 말 것.** 값 이름은 `additive` 다.
  (`none` 은 `null` 로 해석될 뻔했던 이력이 있어 이름을 바꿨다.)
- **`stage2_recon_weight` 를 0 으로 두지 말 것.** 인코더를 미세조정하면서 flow
  matching 만 최소화하면 **잠재를 붕괴시키는 것이 전역 최적**이 되어 수송이 사라진다.
- **`hurdle_gate=soft` 는 분포 지표에 불리하다.** 조건부 기댓값이라 정확한 0 을 만들지
  못하고 세포간 변동이 붕괴한다. 분포를 비교하는 `edist_rel` 에는 `sample` 이 맞다.
- **원본 데이터를 다시 정규화하지 말 것.** `norman.h5ad` 의 X 는 이미 log1p 정규화본이다.
