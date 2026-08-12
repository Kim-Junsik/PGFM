# legacy — 참조 전용, 실행 불가

여기 있는 스크립트는 **삭제된 이전 코드베이스(scDFM 파생)를 전제로 작성**되었다.
지금 상태로는 실행되지 않으며, `src/` 로 되돌리면 안 된다.

## 왜 실행되지 않나

| 참조 | 상태 |
|---|---|
| `src.data_process.data.Data` | 없음 |
| `src.utils._preprocessing` | 없음 |
| `src.utils.utils.process_vocab` | 없음 |
| `src/models/origin/layers.py` | 없음 (주석에서만 참조) |
| `src/tokenizer/*_highly_vocab.json` | 없음 |
| `data/scDFM/{norman,combosciplex}/` | 없음 — 실제 경로는 `data/norman/`, `data/combosciplex/` |

## 왜 지우지 않았나

코드 형태로만 보존되는 지식이 있다.

- **`select_grn_input_adata()` 의 5-fold 교집합 논리** — norman 의 fold 는 비중첩 분할이 아니라
  서로 다른 시드의 독립적인 5회 70/30 추출이므로 fold 간 test 가 겹친다. 따라서 GRN 입력은
  "모든 fold 에서 train 인 double" 로 제한해야 한다. 비자명한 추론이고 뼈대는 재사용 가치가 있다.
- **`run_grnboost2()` 의 dask LocalCluster 기동/정리** — 한 프로세스에서 클러스터를 두 번 이상
  만들 때 나오는 CommClosedError 처리 포함.
- **정규화 규칙** — `log1p -> p99 clip -> max scale`.
- **환경 고정 지식** (파일 헤더) — pertpy/scvi-tools 가 옛 jax 를 끌어와 flow_matching 이 요구하는
  새 jax 와 충돌한다. combosciplex 의 화합물 주석을 pertpy 없이 재구현한 이유.
- **옛 `layers.py` 의 마스크 결합식** `softmax(mask_prior + alpha * mask_residual)` —
  현재 `tanh(M_GRN + α·M_learn)` 설계의 출처.

## 재사용 전 반드시 고칠 것

전체 목록과 근거는 **`docs/PIPELINE-v1.pdf` 5장**에 있다. 요약하면:

1. **[치명] 산출물이 vocab 정렬 N×N 유전자–유전자 행렬** — P-CAB 이 필요한 것은 K×G.
2. **[치명] combinations split 누수** — `select_grn_input_adata()` 가 single/control 을 무조건
   전량 보존하는데, combinations split 에서는 held-out single 23개가 test 다.
   `--split_method` 를 바꿔도 해당 분기는 그대로다. **논문의 생사가 걸린 split 이므로 최우선.**
3. **[차단]** 위 표의 깨진 참조 전부. vocab/tokenizer 계열은 되살릴 필요 없다 —
   새 P-CAB 은 토크나이저 없이 유전자 임베딩을 직접 쓰고, 정렬 기준은 `adata.var_names` 하나다.
4. **[재현성]** `grnboost2()` 에 `seed` 미전달 — 재실행마다 다른 GRN 이 나온다.
5. **[운영]** 유전자 축이 HVG 선택에 종속 — 전체 유전자로 1회 실행 후 열 슬라이싱할 것.

## 현재 계획에서의 위치

GRNBoost2 는 **임계 경로가 아니다.** 확정된 prior 소스는 KEGG + 자유 토큰이고
(`docs/PIPELINE-v1.pdf` 1.2절), GRNBoost2 는 나중에 `grn_source: both` 로 행을 이어붙일 때만
필요하다. 그 시점에 위 5개를 먼저 처리한다.

`build_combosciplex_processed_cache.py` 는 combosciplex 전용이다. 명세서 2.1절이 이 데이터를
조합 벤치마크로 쓸 수 없다고 결론냈으므로(25개 double 중 18개가 구성 single 부재) 우선순위는 더 낮다.
PubChem/rdkit 의존성도 이 파일에서만 필요하다.
