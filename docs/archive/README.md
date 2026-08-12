# archive — 대체된 문서

지우지 않고 옮겨두었다. `docs/` 는 git 미추적이라 삭제하면 복구가 불가능하고,
아래 문서들은 **왜 그 설계를 버렸는지**에 대한 기록으로서 가치가 남아 있다.

| 파일 | 대체한 문서 | 옮긴 이유 |
|---|---|---|
| `LieCFM-design-v1.pdf` | `LieCFM-design-v2.pdf` → `HANDOFF-LT-VAE-LieCFM.pdf` | v2 가 같은 내용을 코드 근거와 함께 다시 씀 |
| `LieCFM-design-v2.pdf` | `HANDOFF-LT-VAE-LieCFM.pdf` + `PIPELINE-v1.pdf` | 부제가 "모델 미착수" 상태를 기술한다. 그 안의 수치는 지금은 없는 코드에서 재생성된 것이라 현재 저장소와 대응하지 않는다 |
| `fig_architecture.pdf` / `.png` | `ARCHITECTURE.pdf` 그림 1~2, `PCAB.pdf` | 하이퍼네트워크가 저랭크 생성원 `(U_a, V_a, β_a)` 를 내놓는 설계인데, 구현된 것은 `AffineGenerator`(완전 행렬)와 `NeuralFieldGenerator`(공유 MLP + 섭동 임베딩)다. 그림이 구현과 다르다 |
| `PIPELINE-v1.pdf` | `docs/README.md` | 3장 "구현 순서" 가 0~5단계 계획인데 0~3단계가 이미 구현되어 낡았다. 살아있던 내용(측정값, 목표선, 명세서 대체 항목, legacy 문제 목록)은 README.md 로 옮겼다 |
| `pathway.txt` | — | 패스웨이 토큰 수 K 를 리뷰어에게 소명하는 방법에 대한 외부 메모. 자유 토큰 아이디어는 우리 결론과 일치하지만, 제안된 "PCA 누적 기여율 90~95%" 기준은 이 데이터에서 성립하지 않는다 — `norman.npz` 의 `explained_var_ratio` 가 128 주성분에서 0.185 라 90% 를 요구하면 K 가 수천이 된다 |

## 옮긴 것 하나 더

`fig_composition.pdf` / `.png` 는 `assets/figures/` 로 갔다. 문서가 아니라 논문 그림이라
`docs/` 가 아니라 에셋 자리가 맞다. 내용은 유효하다 — 교환하는 생성원과 교환하지 않는 생성원을
교환하는 생성원과 교환하지 않는 생성원을 대비시켜 `additivity = commutativity`, 그 어긋남이
epistasis 라는 것을 보이는 그림이고, **현재 설계의 핵심 직관을 그대로 담고 있다.**
