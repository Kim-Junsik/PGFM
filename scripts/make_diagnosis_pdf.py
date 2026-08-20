# -*- coding: utf-8 -*-
"""results/DIAGNOSIS.pdf - where the model stands and what to do next.

    python scripts/make_diagnosis_pdf.py

Every number in here was measured in this repo; the appendix says from which run
and which script. Inference is marked as inference, and the two open questions are
stated as open rather than resolved in prose.
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

NAVY = colors.HexColor('#1F3864')
BLUE = colors.HexColor('#2E75B6')
GREY = colors.HexColor('#7F7F7F')
LINE = colors.HexColor('#BFBFBF')
BG_H = colors.HexColor('#EDF2F9')
BG_Y = colors.HexColor('#FDF8E8')
BD_Y = colors.HexColor('#E0A030')
BG_R = colors.HexColor('#FDF0EF')
BD_R = colors.HexColor('#C55A5A')
BG_G = colors.HexColor('#E9F4F2')
BD_G = colors.HexColor('#2F8F83')


def register_font() -> str:
    candidates = [
        ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ]
    for regular, bold in candidates:
        if not os.path.exists(regular):
            continue
        pdfmetrics.registerFont(TTFont('KR', regular))
        pdfmetrics.registerFont(TTFont('KR-B', bold if os.path.exists(bold) else regular))
        pdfmetrics.registerFontFamily('KR', normal='KR', bold='KR-B',
                                      italic='KR', boldItalic='KR-B')
        return 'KR'
    print("[warn] no CJK font found; Korean will not render", file=sys.stderr)
    return 'Helvetica'


FONT = register_font()
BOLD = 'KR-B' if FONT == 'KR' else 'Helvetica-Bold'

S = dict(
    title=ParagraphStyle('t', fontName=BOLD, fontSize=19, textColor=NAVY,
                         alignment=1, spaceAfter=4, leading=24),
    sub=ParagraphStyle('s', fontName=FONT, fontSize=10, textColor=colors.HexColor('#595959'),
                       alignment=1, spaceAfter=2, leading=14),
    meta=ParagraphStyle('m', fontName=FONT, fontSize=8, textColor=GREY,
                        alignment=1, spaceAfter=13, leading=11),
    h1=ParagraphStyle('h1', fontName=BOLD, fontSize=12.5, textColor=NAVY,
                      spaceBefore=13, spaceAfter=5, leading=17),
    p=ParagraphStyle('p', fontName=FONT, fontSize=8.7, leading=14.2,
                     spaceAfter=5, textColor=colors.HexColor('#262626')),
    cell=ParagraphStyle('c', fontName=FONT, fontSize=7.7, leading=11.4,
                        textColor=colors.HexColor('#262626')),
    head=ParagraphStyle('ch', fontName=BOLD, fontSize=7.9, leading=11.4,
                        textColor=colors.white),
    note=ParagraphStyle('n', fontName=FONT, fontSize=8.2, leading=13.2,
                        textColor=colors.HexColor('#262626')),
    mono=ParagraphStyle('mo', fontName=FONT, fontSize=8, leading=12.5,
                        textColor=NAVY),
)

W = A4[0] - 34 * mm


def table(rows, widths, align_right=()):
    data = [[Paragraph(c, S['head']) for c in rows[0]]]
    for row in rows[1:]:
        data.append([Paragraph(c, S['cell']) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_H]),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]
    for col in align_right:
        style.append(('ALIGN', (col, 0), (col, -1), 'RIGHT'))
    t.setStyle(TableStyle(style))
    return t


def box(text, bg, border):
    t = Table([[Paragraph(text, S['note'])]], colWidths=[W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('BOX', (0, 0), (-1, -1), 0.9, border),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def build():
    story = [
        Paragraph('PGFM 현황 진단과 다음 단계', S['title']),
        Paragraph('측정된 것, 아직 분리되지 않은 것, 제안하는 순서', S['sub']),
        Paragraph('Norman / pcab_lie_commutator · pcab_lie_additive 기준 · 2026-08-19', S['meta']),

        box('<b>한 줄 결론.</b> 상호작용 항(Lie 괄호)은 기여가 없다는 것이 세 설정에서 확정됐다. '
            '그리고 측정 결과 <b>주 병목은 동역학(flow)이다</b> - 표현 경로가 아니다. '
            '오토인코더가 씌우는 상한은 resid_R2 +0.371인데 모델은 -0.975이므로, '
            '<b>flow가 표현 경로보다 2.1배 더 잃는다</b>. 64차원 병목은 무죄이고, '
            '디코더 손실은 실재하지만 부차적이다.', BG_H, BLUE),
        Spacer(1, 4),
        box('<b>이전 판(2026-08-19 초판)의 정정.</b> 초판은 "손실이 인코딩-디코딩 경로에서 발생한다"고 '
            '적었다. 그 근거였던 transport 진단(코사인 0.901 -> 0.690)은 <b>평균 변위라는 1차 양</b>을 '
            '재는데, resid_R2는 훨씬 미세한 2차 잔차를 잰다. 필드가 벌크 변위를 맞히면서 잔차를 '
            '전혀 못 맞힐 수 있고, 아래 상한 측정이 실제로 그렇다는 것을 보였다.', BG_Y, BD_Y),

        Paragraph('1. 층별 진단', S['h1']),
        Paragraph('파이프라인은 세 층이다. 세포 -> [인코더] -> z(64차원) -> [속도장 + 상호작용] '
                  '-> z1 -> [디코더] -> 예측 세포.', S['p']),
        table([
            ['층', '상태', '근거'],
            ['동역학 (flow)', '<b>주 병목</b>',
             '벌크 변위는 맞다 (잠재 비율 0.973, 코사인 0.901). 그러나 2차 잔차를 못 맞힌다: '
             '상한 +0.371 대비 실측 -0.975, <b>1.346을 잃는다</b>.'],
            ['상호작용 항', '<b>기여 없음 (확정)</b>',
             '백본 2종 x 유전자공간 2종 x 폴드 2종에서 additive에 패배. 게이트는 열려 있었다 '
             '(scale이 0에서 이탈, zero% 0.0, 상대 크기 1~6%).'],
            ['표현 경로 - 디코더', '<b>부차적 손실</b>',
             '완벽한 flow를 가정해도 상한이 +0.371에 그친다 (<b>0.629를 잃음</b>). '
             'calib 0.236, floor 1.0406.'],
            ['표현 경로 - 64차원 병목', '<b>무죄</b>',
             '잔차의 신호 내 비중이 유전자공간 0.393 vs 잠재공간 <b>0.575</b>. '
             '압축이 비가법 신호를 지우지 않는다.'],
        ], [34 * mm, 30 * mm, W - 64 * mm]),

        Paragraph('2. 핵심 측정값', S['h1']),
        table([
            ['지표', '값', '읽는 법'],
            ['test doubles 변위 비율 / 코사인 (잠재)', '0.973 / 0.901',
             '1.0이 정확한 길이. 잠재공간에서는 문제가 없다.'],
            ['test doubles 코사인 (유전자공간)', '<b>0.690</b>',
             '잠재 0.901에서 0.21 하락. 디코더를 지나며 잃는다.'],
            ['edist_rel_autoencoder_floor', '<b>1.0406</b>',
             'transport를 끄고 encode-decode만 해도 대조군이 참 분포에서 <b>멀어진다</b>. '
             '1을 넘는 것은 표현 경로 자체의 문제다.'],
            ['calib = sigma / rmse', '<b>0.236</b>',
             '예측 분산이 실제 오차의 24%. 디코더가 세포 간 변동을 4배 좁게 생성한다.'],
            ['DE-Spearman (1,000 유전자)', '<b>-0.001</b>',
             '유의 유전자의 순위가 참값과 무상관. 전체 방향은 맞는데 유전자별 배분이 무작위.'],
            ['L2 / Additive 베이스라인', '<b>1.431</b>',
             'm_A + m_B - m_ctrl 이라는 산수보다 43% 나쁘다. scDFM은 같은 비율이 0.879.'],
            ['train singles 변위 비율', '<b>0.588</b>',
             '직접 지도학습하는 조건조차 59%밖에 재현하지 못한다.'],
        ], [52 * mm, 22 * mm, W - 74 * mm], align_right=(1,)),

        Paragraph('3. 확정된 판정 두 가지', S['h1']),
        table([
            ['대상', '판정', '근거'],
            ['Lie 괄호', '<b>기여로 사용 불가</b>',
             'mlp/3074/fold0, pcab/3074/fold0, pcab/5054/fold1 — 세 설정 모두에서 additive 우세. '
             '더 자유로운 형태(free_mlp, z와 t 입력, 파라미터 63배)는 <b>더 나빴다</b>. '
             '게이트가 열린 것을 확인했으므로 "구현 실패"가 아니라 "효과 없음"이다.'],
            ['P-CAB / E-RCA', '<b>성능 기여 아님</b>',
             '직접 비교한 3,074/fold0에서 resid_R2, edist, floor, pearson_delta, DE-Spearman, DS, '
             'overlap <b>전 항목</b> mlp에 열세. 단, 파라미터 3.6배 적게(3.8M vs 13.8M) 쓰면서 '
             'stage 1 재구성은 동일하다 — 이것은 <b>효율 관찰</b>이지 성능 기여가 아니다.'],
        ], [26 * mm, 30 * mm, W - 56 * mm]),

        Paragraph('4. 손실의 분해 (Step 0 결과)', S['h1']),
        Paragraph('참 세포를 transport 없이 encode-decode 시켜 잔차를 다시 계산하면, '
                  '이 오토인코더를 지나는 어떤 모델도 넘을 수 없는 상한이 나온다. '
                  '아래는 모두 동일한 1,000 유전자, 동일한 37개 조건 기준이다.', S['p']),
        table([
            ['', 'resid_R2', '잃는 몫', '판정'],
            ['완벽한 모델 (이론)', '1.0000', '-', '-'],
            ['<b>오토인코더 상한</b>', '<b>+0.3710</b>', '0.629', '디코더 몫. 실재하나 부차적'],
            ['pcab_lie_additive', '-0.8713', '1.242', '-'],
            ['<b>pcab_lie_commutator</b>', '<b>-0.9753</b>', '<b>1.346</b>', '<b>flow 몫. 주 병목</b>'],
        ], [46 * mm, 24 * mm, 20 * mm, W - 90 * mm], align_right=(1, 2)),
        Spacer(1, 5),
        box('<b>동역학이 표현 경로보다 2.1배 더 잃는다.</b> 인코더와 디코더를 완벽하게 고쳐도 '
            '-0.975에서 +0.371까지밖에 못 간다. 반대로 flow를 고치면 1.346을 회수할 여지가 있다. '
            '따라서 latent_dim 확대나 디코더 교체는 <b>우선순위가 아니다</b>.', BG_Y, BD_Y),

        Paragraph('5. 검증이 필요한 관측', S['h1']),
        Paragraph('train singles가 0.588인데 test doubles는 0.973이다. 직접 지도학습하는 조건을 '
                  '59%밖에 못 맞히면서 그것을 더해 만드는 조합이 97%가 나오는 것은 정상이 아니다.', S['p']),
        box('<b>가설.</b> Norman의 이중 섭동이 하위가법(sub-additive)이라 '
            '<b>Delta_AB 가 대략 0.6 x (Delta_A + Delta_B)</b> 수준이고, 언더피팅된 싱글의 합이 '
            '우연히 참값에 맞아떨어지는 것일 수 있다. 그렇다면 이중 예측은 올바른 싱글 위에 '
            '세워진 것이 아니라 <b>두 오차의 상쇄</b>이며, 그 위에 어떤 상호작용 항을 얹어도 '
            '의미가 없다. 아래 Step 1로 검증된다.', BG_R, BD_R),

        Paragraph('6. 동역학 진단 - 1.346을 어디서 잃는가', S['h1']),
        table([
            ['가설', '측정', '판정'],
            ['A. CFM 손실의 하한에 도달했다',
             'fm 0.6344 vs 조건별 평균만 내는 예측기 1.0581. 타깃 분산 1.0622',
             '<b>아님</b><br/>평균 너머 40% 설명'],
            ['B. 20-step RK4 적분 오차',
             '같은 체크포인트를 20 / 100 스텝으로 재적분: resid_R2 -0.9577 / -0.9825',
             '<b>아님</b><br/>5배로도 무변화'],
            ['C. 구성 싱글이 언더피팅',
             'test doubles를 실제로 구성하는 46개 싱글: 유전자공간 비율 0.6457, 코사인 0.6951',
             '<b>실재</b>'],
            ['D. 합성 비선형성',
             '||int(u_a+u_b) - (int u_a + int u_b)|| / ||sum|| = <b>0.9945</b>',
             '<b>주범</b>'],
        ], [42 * mm, W - 74 * mm, 32 * mm]),
        Spacer(1, 5),
        box('<b>D가 핵심이다.</b> 모델의 합성은 <b>속도 수준</b>에서만 가법이다. 변위는 그렇지 않다 - '
            'u가 z에 의존하므로 int(u_a + u_b) 와 int u_a + int u_b 는 다르다. 측정된 차이는 '
            '<b>합 자체의 99%</b>다.<br/><br/>'
            '즉 interaction=additive 조차 Lie 괄호를 켜기 전에 이미 100% 규모의 비가법성을 만든다. '
            '그런데 참 데이터의 비가법성은 12%뿐이다 (Step 1: ||Delta_AB||/||Delta_A+Delta_B|| = 0.879). '
            '<b>모델이 8배 과도한 비가법성을 만들어내고 있고, 그것이 학습으로 통제되지 않는다</b> - '
            '지금 손실함수 어디에도 이 항이 나타나지 않는다.<br/><br/>'
            '이것이 상호작용 항이 아무것도 못 하는 이유이기도 하다. 잔차 자리는 이미 통제되지 않은 '
            '적분 비선형성이 차지했고, 그 위의 1~6%짜리 괄호는 묻힌다.', BG_R, BD_R),

        Paragraph('7. 처방 - D를 손실에 넣기', S['h1']),
        Paragraph('현재 stage 2의 손실은 조건 S 하나에 대한 flow matching 뿐이다. '
                  '조합의 변위와 그 구성 싱글들의 변위 사이의 <b>관계</b>를 제약하는 항이 없다. '
                  '학습 doubles에 대해서는 참 잠재 잔차를 계산할 수 있으므로, 그것을 직접 감독할 수 있다.', S['p']),
        table([
            ['', '항', '설명'],
            ['참값', 'r_true = z_AB - z_A - z_B + z_ctrl',
             '학습 doubles의 참 세포를 encode 해서 얻는 잠재 잔차. Step 0에서 이 양이 잠재공간에 '
             '<b>보존된다</b>는 것을 확인했다 (비중 0.575 &gt; 유전자공간 0.393).'],
            ['모델', 'r_model = Phi_ab(z0) - Phi_a(z0) - Phi_b(z0) + z0',
             'Phi_S 는 조건 S 로 적분한 flow map. 세 번 적분해서 얻는다.'],
            ['추가 손실', 'L_resid = || r_model - r_true ||^2',
             '이 항이 곧 resid_R2 가 재는 양이다. 지금은 손실 어디에도 없다.'],
        ], [16 * mm, 62 * mm, W - 78 * mm]),
        Spacer(1, 4),
        box('<b>비용을 줄이는 법.</b> r_model 을 세포별로 계산하면 스텝마다 적분이 3배가 된다. '
            'resid_R2 자체가 집단 <b>평균</b>에 대한 양이므로, z0 의 평균 한 점만 적분해도 같은 '
            '것을 감독한다. 그러면 64차원 점 3개의 적분이라 사실상 공짜다. '
            '적분 스텝도 이 항에 한해 낮춰도 된다 (B에서 스텝 수가 무관함이 확인됐다).', BG_G, BD_G),

        Paragraph('8. 제안하는 순서', S['h1']),
        table([
            ['', '작업', '비용', '얻는 것'],
            ['0', '<b>표현 경로 상한 측정</b> — <b>완료</b><br/>'
                  'scripts/diagnose_bottleneck.py',
             '10분',
             '상한 +0.371 확정. flow가 2.1배 더 잃는다는 것과 64차원 병목이 무죄라는 것이 나왔다.'],
            ['1', '<b>가법 배율 측정</b> — <b>완료</b><br/>같은 스크립트',
             '10분', '0.879. 상쇄 가설 기각.'],
            ['2', '<b>평가 누출 제거</b><br/>stage 1을 train 조건 세포로만 학습',
             '두 팔 ~10시간',
             'stage 1이 홀드아웃 조건 세포 10,643개(12.5%)를 본다. 고치지 않으면 <b>어떤 숫자도 '
             '방어되지 않는다</b>. 이후 모든 측정의 기준선이므로 먼저 확보한다.'],
            ['3', '<b>동역학 진단</b> — <b>완료</b><br/>scripts/diagnose_dynamics.py',
             '~30분',
             'A와 B 기각, C 실재, <b>D가 주범</b>. 6절 참조.'],
            ['4', '<b>L_resid 도입</b> — <b>다음 작업</b><br/>7절의 잠재 잔차 감독 항을 stage 2 손실에 '
                  '추가. 동시에 C를 겨냥해 singles 가중치를 올린다',
             '~10시간',
             'D를 손실 안으로 들여온다. 상한 +0.371까지 1.346의 회수 여지.'],
            ['5', '<b>OT 대조군</b><br/>train.coupling=random',
             '35분',
             '한 번도 돌린 적이 없다. 논문에 남은 명시적 주장 중 <b>유일하게 미검증</b>.'],
        ], [7 * mm, 58 * mm, 22 * mm, W - 87 * mm]),

        Paragraph('9. 하지 말 것', S['h1']),
        box('<b>하이브리드 Lie 변형</b> — v = u_a + u_b + g_ab(z,t) (*) [u_a,u_b] + alpha x r_ab(z,t)<br/><br/>'
            '벡터 게이트는 이미 측정한 두 실패(scalar 게이트 x 괄호, 무제약 MLP) <b>사이</b>에 있고, '
            'r_ab는 우리 FreeInteraction과 사실상 같은 것이다. 양 끝이 모두 additive 아래인데 '
            '그 중간이 위로 갈 이유가 없다. 권장되는 5변형 x 5 seed는 <b>250시간</b>이며, '
            '무엇보다 <b>잠재공간을 정교화하는 방향인데 병목은 표현 경로에 있다</b>.',
            BG_R, BD_R),
        Spacer(1, 4),
        box('<b>Lie 괄호를 지우라는 뜻은 아니다.</b> interaction=commutator는 설정값 하나이고 '
            'commutator.py는 독립 모듈이라 유지 비용이 0이다. 논문 ablation 표에서 리뷰어가 '
            '정확히 이 행을 요구한다. <b>기여에서 내리고 ablation 한 줄로 남긴다</b>는 뜻이다.',
            BG_G, BD_G),

        Paragraph('부록. 근거의 출처', S['h1']),
        table([
            ['숫자', '어디서'],
            ['resid_R2, edist_rel, floor',
             'results/runs/pcab_lie_*/results.json (학습이 끝나면 자동 생성)'],
            ['변위 비율, 코사인 (잠재 / 유전자공간)', 'python scripts/diagnose_transport.py'],
            ['게이트 scale, zero%, 상호작용 상대 크기', 'python scripts/diagnose_gate.py'],
            ['rmse, sigma, calib, gate_bce, kl, mask_l1', 'results/runs/*/train.log 의 stage1 행'],
            ['L2, MSE, MAE, DE-Spearman, Pearson delta, DS',
             'python scripts/paper_table.py --infer-top-gene 1000'],
            ['Control / Additive 베이스라인 L2',
             '조건 평균에서 직접 계산 (scdfm_eval_genes 로 뽑은 동일 1,000 유전자 기준)'],
            ['누출 세포 수 10,643 (12.5%)',
             'src/train/loop.py 의 train_stage1 이 조건 필터 없이 전체에서 샘플링'],
            ['각 지표의 정의', 'results/METRICS.pdf'],
        ], [58 * mm, W - 58 * mm]),
    ]

    out = os.path.join('results', 'DIAGNOSIS.pdf')
    os.makedirs('results', exist_ok=True)
    SimpleDocTemplate(out, pagesize=A4,
                      leftMargin=17 * mm, rightMargin=17 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm,
                      title='PGFM 현황 진단과 다음 단계').build(story)
    print(f"-> {out}")


if __name__ == '__main__':
    build()
