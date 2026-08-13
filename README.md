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
테스트        33개 통과   (구조 보장 14 + split 19)
백본          4종 모두 end-to-end 실행 확인
목표선        미달
```

| 항목 | 상태 |
|---|---|
| 데이터 · split · 평가 · 기준선 | 완료, 검증됨 |
| 백본 4종 (mlp / transformer / scvi / pcab) | 완료, 1에폭 실행 확인 |
| LieCFM 동역학 + 구조 보장 4가지 | 완료, 14개 테스트 |
| split 동일성 계약 + 생성기 | 완료, 19개 테스트 |
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

진입점은 셋뿐이다. `scripts/` 안의 파일은 이들이 호출하는 내부 모듈이므로 직접 부를 일이 없다.

```bash
python data_prepare.py     # 데이터 준비 + 기준선   (한 번, 약 5분)
sh train.sh                # 학습
sh test.sh                 # 평가
```

전제 파일 — 이것만 있으면 나머지는 전부 만들어진다.

```
data/norman/norman.h5ad          원본 (2.2 GB)
data/norman/split_results.pkl    split 원본
.env-celleval/                   test.sh --celleval 를 쓸 때만
```

## 0. `data_prepare.py` — 데이터 준비

```bash
python data_prepare.py
python data_prepare.py --force                # KEGG 재다운로드 + 캐시 재생성
python data_prepare.py --set data.n_hvg=5000  # 유전자 공간 변경
```

네 단계를 순서대로 수행한다.

| | 내용 | 산출물 |
|---|---|---|
| 1 | KEGG 스냅샷 다운로드 | `assets/kegg/` |
| 2 | split 검증 | — |
| 3 | 모델 공간 캐시 (HVG + 섭동 표적) | `assets/norman_modeled.h5ad` |
| 4 | 기준선 계산 | `results/baselines_*.json` |

**4단계가 여기 있는 이유**는 그것이 테스트가 아니기 때문이다. 모델을 로드하지도, 학습된
무언가를 평가하지도 않는다. ridge-additive 가 얼마나 하는지는 **데이터셋의 상수**이므로
한 번 계산해 두고 나중에 읽는다.

2단계에서 이 두 줄이 나와야 한다. 아니면 그 뒤 숫자는 전부 의미가 없다.

```
reference folds are clean          : True
combinations derived from them     : True
```

## 1. `train.sh` — 학습

학습 명령은 하나다. 아래는 같은 명령의 변형이지 별도의 단계가 아니다.

```bash
sh train.sh                                      # 모델 1개. 이것이 전부
sh train.sh --tag run1 model.backbone=pcab       # 이름 붙인 run + 설정 변경
sh train.sh model.interaction=additive split.method=combinations
sh train.sh train.stage1_epochs=60 train.stage2_epochs=80
sh train.sh --ablation                           # 같은 명령을 6번 (sweep)
sh train.sh --ablation model.backbone=pcab       # 다른 백본으로 같은 sweep
```

`key.path=value` 로 모든 설정을 덮어쓴다. **없는 키를 쓰면 즉시 예외**가 난다 —
오타가 조용히 기본값으로 도는 것을 막기 위해서다.

결과는 `results/runs/<tag>/` 에 `train.log`, `checkpoint.pt`, `results.json` 으로 남는다.

### `--ablation` — 가장 먼저 할 일

`generator` 2종 × `interaction` 3종 = 6개를 순차 실행한다. **백본은 건드리지 않는다** —
`model.backbone` 은 인코더/디코더를, `generator`/`interaction` 은 잠재 동역학을 정하는
별개의 축이고, 둘을 같이 바꾸면 이득이 어느 쪽에서 왔는지 구분할 수 없다.

실행 이름에 백본이 들어가므로(`s2_mlp_*`, `s2_pcab_*`) 백본을 바꿔 다시 돌려도
이전 결과를 덮어쓰지 않는다. 판정 순서:

1. `commutator` vs `additive` — Lie 괄호가 기여하는가
2. `commutator` vs `free_mlp` — 대수적 형태가 자유 MLP 보다 나은가
3. `affine` vs `neural_field` — 비선형 벡터장이 필요한가

3번에서 `affine` 이 이기면 "고차 동역학이 불필요하다" 는 **발견**이 되고 논문의
프레이밍이 바뀐다. 지는 결과가 아니라 다른 결과다.

### 주요 설정 축

```
model.backbone         mlp | transformer | scvi | pcab
model.decoder_head     mse | hurdle | zinb
model.hurdle_gate      soft | hard | sample
model.generator        affine | neural_field
model.interaction      additive | commutator | free_mlp
model.mask_combine     gate | logit_bias
model.mask_activation  tanh | sigmoid
split.method           additive | combinations
train.coupling         uot | ot | random
```

### 배치 크기

RTX 3050 8.6 GB 실측. `pcab` 의 어텐션 텐서는 `[B, 277, 3074]` 로 배치 32 에서
27 M 원소다.

| 백본 | 권장 배치 | fwd+bwd |
|---|---|---|
| `mlp` · `scvi` | 256 | 빠름 |
| `transformer` | 64 | 중간 |
| `pcab` | 32~64 | 31 ms (배치 32) |

### 설정이 끝까지 도는지만 볼 때

```bash
sh train.sh --tag smoke train.stage1_epochs=1 train.stage2_epochs=1     train.max_steps_per_epoch=40 eval.n_gen_cells=64
```

한 에폭이 40 스텝에서 끊긴다. 성능 수치로 인용하면 안 된다.

## 2. `test.sh` — 평가

```bash
sh test.sh                                       # 가장 최근 run
sh test.sh results/runs/<tag>                    # 특정 run
sh test.sh --summary                             # 전체 run 한 표에
sh test.sh results/runs/<tag> --celleval         # cell-eval 0.5.42 채점 추가
```

내부 지표는 학습이 이미 기록해 두었으므로 여기서는 읽어와 기준선 옆에 놓는다.
`--celleval` 만 추가 계산을 한다 — 대조군 세포를 수송해 `pred.h5ad` / `real.h5ad` 를
쓰고, 의존성이 충돌하므로 별도 Python 3.11 환경에 넘긴다.

### 결과 읽는 법

```
run                       backbone  gen           interaction  resid_R2   edist   floor
verify_fix                mlp       neural_field  commutator    -0.8789  0.5950  1.0113
```

**`edist` 보다 `floor` 를 먼저 본다.** floor 는 수송을 끈 오토인코더 성능이다.
둘이 같으면 flow 가 아무 일도 하지 않았다는 뜻이고, 그 상태에서 `interaction` 을
바꿔 봐야 차이가 나타나지 않는다. 위 예에서는 0.5950 < 1.0113 이므로 수송이 일어났다.

`resid_R2` 는 평균 기반이라 디코더의 분산 문제와 무관하다. 두 지표가 갈릴 때는
이쪽이 더 신뢰할 만하다.

## 3. 테스트

무엇을 건드리든 돌린다. 모델 실행에는 필요 없지만 **코드를 수정했으면 필수**다.

```bash
python -m pytest tests/ -q
```

33개가 통과해야 한다. 정확도 테스트가 아니다 — **실패는 학습 부족이 아니라 계약 위반**을
뜻한다.

### `tests/test_structure.py` (14개) — 학습 없이 성립해야 하는 성질

대조군 불변 `v(z,t,{}) = 0`, 단일 직접 감독 `v(z,t,{a}) = u_a`, 가법 정확 복원
`Λ=0 → Σu_a`, 순열 불변, 괄호 반대칭, 그리고 jvp 교환자가 명시적 야코비안과 1e-8 까지
일치하는지. 실패하면 속도장 합성이 틀린 것이다.

### `tests/test_splits.py` (19개) — split 동일성과 생성기

split 이 `data/norman/split_results.pkl` 과 **같은 리스트, 같은 순서**인지 확인한다.
"동등" 이 아니라 축자 일치다. 유전자 공간은 자유롭게 바꿔도 되지만(n_hvg 1000 / 3000 /
5000 / 전체에서 검사) **split 은 절대 움직이면 안 된다.**

split 은 조용히 깨지는 종류다 — 이름 변경, 재정렬, 캐시 복사본, 유전자 공간 변경 중
무엇으로도 어긋날 수 있고 학습 루프는 아무것도 눈치채지 못한다. 그러면 "저자 split 에서
저자 방법이 이겼다"는 반박을 막는다는 목적 자체가 사라지는데, 그 사실을 심사 중에 알게
된다. 그래서 주석이 아니라 테스트로 고정했다.

split 자체가 맞아도 **거기서 파생되는 train 집합**은 따로 깨질 수 있다. 실제로
combinations 의 train 이 double 110 → 88 로, single 78 → 0 으로 줄어든 적이 있고,
ridge-additive 목표선이 0.1853 에서 0.0487 로 **낮아져 있었다** — 즉 넘기 쉬워졌는데
아무것도 이상해 보이지 않았다. 그래서 개수까지 못박는 테스트가 두 개 더 있다.


## 4. 새 데이터로 학습하기

필요한 것은 h5ad 하나와 설정 몇 줄이다. 코드를 고칠 일은 없다.

### 데이터에 요구되는 것

| 항목 | 요구 |
|---|---|
| `X` | log1p 정규화된 발현 행렬 (raw count 면 먼저 정규화할 것 — 파이프라인은 정규화하지 않는다) |
| `obs['condition']` | 조건 이름. 기본 형식은 `A+B` / `A+ctrl` / `ctrl` |
| `var_names` | 유전자 심볼. P-CAB 마스크가 KEGG 와 심볼로 매칭한다 |

### 조건 이름이 다를 때

```bash
--set data.control_label=control data.condition_separator=+
```

`ctrl` 은 하드코딩이 아니라 설정이다(`src/data/conventions.py`). 대조군을 찾지 못하면
평가 도중이 아니라 **로드 시점에 명확한 예외**가 난다.

### split — 세 가지 출처

**① 데이터셋이 pkl 로 제공** (Norman)

```bash
--set split.source=reference_pkl split.reference_pkl=data/newset/splits.pkl
```

**② 세포마다 obs 에 들어 있음** (combosciplex)

```bash
--set split.source=obs_column split.obs_key=split split.obs_test_value=test
```

한 조건의 세포가 여러 값에 걸치면 다수결로 배정한다. 그러지 않으면 같은 조건이
train 과 test 양쪽에 들어간다.

**③ split 이 아예 없음 — 시드에서 생성**

```bash
--set split.source=generated split.generate_scheme=combinations split.generate_seed=0
```

| `generate_scheme` | 홀드아웃 | 용도 |
|---|---|---|
| `doubles` | double 의 일부 | 조합 일반화. single 은 남으므로 가법 기준선 계산 가능 |
| `combinations` | double + **그 구성 single 까지** | 더 어렵다. 가법 기준선이 계산 불가가 되는 것이 요점 |
| `group` | obs 컬럼이 지정한 **그룹 통째로** | **cell line / donor / batch 일반화** |

생성된 split 은 **디스크에 쓰지 않는다.** 매 로드마다 시드에서 재계산한다 — 파일로
저장하면 그 파일이 시드와 어긋나도 아무도 모르기 때문이다. 또한 `default_rng` 가 아니라
legacy `RandomState` 를 쓴다. numpy 는 `default_rng` 의 스트림을 버전 간 보장하지 않고,
numpy 를 올렸을 때 split 이 조용히 바뀌면 그 split 으로 측정한 모든 수치가 무효가 된다.

### cell line 데이터

조합 일반화와 **다른 축**이다. 둘 다 보려면 따로 돌려서 비교한다.

```bash
python data_prepare.py --set   data.raw_h5ad=data/newset/new.h5ad   data.cache_h5ad=assets/new_modeled.h5ad   split.source=generated split.generate_scheme=group split.group_key=cell_type

sh train.sh data.cache_h5ad=assets/new_modeled.h5ad   split.source=generated split.generate_scheme=group split.group_key=cell_type
```

`split.group_key` 가 obs 에 없으면 어느 컬럼을 지정해야 하는지 알려주는 예외가 난다.

### 유전자 섭동이 아닐 때 (약물 등)

```bash
--set data.force_include_targets=false
```

강제 포함은 "섭동 표적이 var_names 의 유전자" 라는 전제 위에 있다. 약물 이름은
유전자가 아니므로 넣을 것이 없다. combosciplex 에서 실측: 섭동 이름 중 var_names 에
존재하는 것이 **0개**다.

같은 이유로 P-CAB 의 "섭동이 패스웨이 토큰과 정렬된다" 는 이야기도 성립하지 않는다.
그 데이터가 약물–패스웨이 대응을 obs 에 들고 있다면(combosciplex 의 `pathway1` /
`pathway2`) 그쪽을 쓰는 편이 낫다.

### 조합 벤치마크가 될 수 있는지 먼저 확인할 것

double 의 구성 single 이 데이터에 없으면 가법 기준선을 계산할 수 없고, 그러면
`resid_R2` 로 판정할 수가 없다. combosciplex 는 double 25개 중 **7개만** 구성 single
을 가지고 있어 조합 벤치마크로 쓸 수 없다 — 분포 이동 테스트로만 쓴다.

`python data_prepare.py` 의 기준선 단계에서 `skip` 열을 보면 바로 드러난다.


## 5. 두 데이터셋 — 실행 명령어 전체

그대로 복사해 쓰면 된다. Norman 이 기본값이므로 설정을 주지 않고,
combosciplex 는 매번 설정 세 개를 붙인다.

### Norman — 조합 일반화 (주 데이터)

```bash
# 0. 데이터 준비 (한 번)
python data_prepare.py

# 1. 2단계 판정 — 여기부터 시작한다
sh train.sh --ablation
sh test.sh --summary

# 2. 1번을 통과했다면 2x2 를 완성한다
sh train.sh --ablation model.backbone=pcab
sh test.sh --summary

# 3. 백본 비교
sh train.sh --tag nm_mlp         model.backbone=mlp
sh train.sh --tag nm_transformer model.backbone=transformer train.batch_size=64
sh train.sh --tag nm_scvi        model.backbone=scvi model.decoder_head=zinb
sh train.sh --tag nm_pcab        model.backbone=pcab train.batch_size=32
sh test.sh --summary

# 4. 대조군
sh train.sh --tag nm_mse       model.decoder_head=mse
sh train.sh --tag nm_softgate  model.hurdle_gate=soft
sh train.sh --tag nm_maskadd   model.backbone=pcab model.mask_combine=logit_bias
sh train.sh --tag nm_masksig   model.backbone=pcab model.mask_activation=sigmoid
sh train.sh --tag nm_maskfree  model.backbone=pcab model.mask_mode=residual_only
sh train.sh --tag nm_randcpl   train.coupling=random

# 5. combinations split
sh train.sh --ablation split.method=combinations
sh test.sh --summary

# 6. 외부 비교
sh test.sh results/runs/nm_pcab --celleval
```

### combosciplex — 분포 이동 (보조 데이터)

```bash
# 0. 데이터 준비 (한 번)
python data_prepare.py --set   data.raw_h5ad=data/combosciplex/combosciplex.h5ad   data.cache_h5ad=assets/combosciplex_modeled.h5ad   data.control_label=control   data.force_include_targets=false   split.source=obs_column split.obs_test_value=ood

# 1. 학습 — 설정 세 개를 매번 붙인다
sh train.sh --ablation   data.cache_h5ad=assets/combosciplex_modeled.h5ad   data.control_label=control   split.source=obs_column split.obs_test_value=ood
sh test.sh --summary

# 2. 백본 비교
sh train.sh --tag cb_mlp   data.cache_h5ad=assets/combosciplex_modeled.h5ad data.control_label=control   split.source=obs_column split.obs_test_value=ood

sh train.sh --tag cb_pcab model.backbone=pcab train.batch_size=32   data.cache_h5ad=assets/combosciplex_modeled.h5ad data.control_label=control   split.source=obs_column split.obs_test_value=ood
sh test.sh --summary
```

| combosciplex 설정 | 이유 |
|---|---|
| `data.control_label=control` | 대조군이 `control+control` 이고 single 은 `control+Drug` 로 쓴다 |
| `data.force_include_targets=false` | 섭동이 약물이라 var_names 에 없다 — 실측 **0 / 17** |
| `split.source=obs_column` | split pkl 이 없고 `obs['split']` 에 들어 있다 |
| `split.obs_test_value=ood` | `test` 는 조건 안에서 세포를 나눈 것이라 조건 단위 홀드아웃이 아니다 |

### 두 데이터셋의 차이

| | Norman | combosciplex |
|---|---|---|
| 크기 | 84,986 × 19,264 | 63,378 × 27,518 |
| 섭동 | 유전자 (CRISPRa) | 약물 |
| single / double | 101 / 125 | 6 / 25 |
| 구성 single 을 가진 double | **125 / 125** | **7 / 25** |
| 평가 가능한 test double | 37 (fold 당) | **3** |
| 쓸 수 있는 주장 | **조합 일반화** | **분포 이동 일반화** |

마지막 두 줄이 핵심이다. combosciplex 의 ood 에서 구성 single 까지 갖춘 double 은 **3개**뿐이라
`resid_R2` 로 통계적 주장을 할 수 없다. 조합 일반화는 Norman 으로 하고, combosciplex 는
"다른 modality 에서도 동역학이 작동한다" 에만 쓴다.

### 순서가 중요하다

1번이 실패하면 2~6번은 의미가 없다. 기여 ① (Lie 괄호) 이 없으면 논문은 "패스웨이 인코더
하나" 가 되고, 그것만으로는 약하다. `--ablation` 은 백본을 이름에 넣으므로
(`s2_mlp_*`, `s2_pcab_*`) 두 sweep 이 서로를 덮어쓰지 않는다.

2번이 필요한 이유는 리뷰어가 **"pcab 위에서도 Lie 괄호가 여전히 도움이 되는가"** 를
묻기 때문이다. mlp 에서만 보였다면 두 기여가 서로를 대체하지 않는다는 말을 할 수 없다.

| | `additive` | `commutator` |
|---|---|---|
| **mlp** | A | B |
| **pcab** | C | D |

`B > A` 이고 `D > C` 여야 한다. `D ≈ C` 라면 pcab 을 쓰는 순간 괄호가 무의미해진다는
뜻이고, 기여 ① 이 ② 에 흡수되어 논문은 사실상 기여 하나짜리가 된다.

## 알아두면 좋은 함정

- **`--set model.interaction=none` 은 쓰지 말 것.** 값 이름은 `additive` 다.
  (`none` 은 `null` 로 해석될 뻔했던 이력이 있어 이름을 바꿨다.)
- **`stage2_recon_weight` 를 0 으로 두지 말 것.** 인코더를 미세조정하면서 flow
  matching 만 최소화하면 **잠재를 붕괴시키는 것이 전역 최적**이 되어 수송이 사라진다.
- **`hurdle_gate=soft` 는 분포 지표에 불리하다.** 조건부 기댓값이라 정확한 0 을 만들지
  못하고 세포간 변동이 붕괴한다. 분포를 비교하는 `edist_rel` 에는 `sample` 이 맞다.
- **원본 데이터를 다시 정규화하지 말 것.** `norman.h5ad` 의 X 는 이미 log1p 정규화본이다.
- **`scripts/` 안의 파일을 직접 부르지 말 것.** 진입점 셋이 순서와 인자를 맞춰 호출한다.
  개별 호출은 단계를 건너뛰기 쉽고, 실제로 그렇게 해서 기준선이 낮아진 적이 있다.
