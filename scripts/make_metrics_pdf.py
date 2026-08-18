# -*- coding: utf-8 -*-
"""results/METRICS.pdf - what every column in the evaluation output means.

    python scripts/make_metrics_pdf.py

A glossary, not an analysis. Descriptions and best-value directions for the
cell-eval columns are copied from its own registry (cell_eval/metrics/_impl.py),
not inferred from the names, so they say what the library says. Our own columns
are described from src/eval/metrics.py and src/eval/predict.py.
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor('#1F3864')
BLUE = colors.HexColor('#2E75B6')
GREY = colors.HexColor('#7F7F7F')
LINE = colors.HexColor('#BFBFBF')
BG_H = colors.HexColor('#EDF2F9')
BG_N = colors.HexColor('#FDF8E8')
BD_N = colors.HexColor('#E0A030')


def register_font() -> str:
    """Malgun on Windows, Nanum/Noto in a Linux container.

    Falls back to Helvetica instead of raising: the column names are all ASCII,
    so an English-only PDF still carries the table even if the prose breaks.
    """
    candidates = [
        ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if not os.path.exists(regular):
            continue
        pdfmetrics.registerFont(TTFont('KR', regular))
        pdfmetrics.registerFont(TTFont('KR-B', bold if os.path.exists(bold) else regular))
        pdfmetrics.registerFontFamily('KR', normal='KR', bold='KR-B',
                                      italic='KR', boldItalic='KR-B')
        return 'KR'
    print("[warn] no CJK font found; Korean text will not render", file=sys.stderr)
    return 'Helvetica'


FONT = register_font()
BOLD = 'KR-B' if FONT == 'KR' else 'Helvetica-Bold'

S = dict(
    title=ParagraphStyle('t', fontName=BOLD, fontSize=19, textColor=NAVY,
                         alignment=1, spaceAfter=4, leading=24),
    sub=ParagraphStyle('s', fontName=FONT, fontSize=10, textColor=colors.HexColor('#595959'),
                       alignment=1, spaceAfter=2, leading=14),
    meta=ParagraphStyle('m', fontName=FONT, fontSize=8, textColor=GREY,
                        alignment=1, spaceAfter=14, leading=11),
    h1=ParagraphStyle('h1', fontName=BOLD, fontSize=12.5, textColor=NAVY,
                      spaceBefore=13, spaceAfter=5, leading=17),
    p=ParagraphStyle('p', fontName=FONT, fontSize=8.6, leading=14,
                     spaceAfter=5, textColor=colors.HexColor('#262626')),
    cell=ParagraphStyle('c', fontName=FONT, fontSize=7.7, leading=11.4,
                        textColor=colors.HexColor('#262626')),
    code=ParagraphStyle('co', fontName=FONT, fontSize=7.5, leading=11.4,
                        textColor=NAVY),
    head=ParagraphStyle('ch', fontName=BOLD, fontSize=7.9, leading=11.4,
                        textColor=colors.white),
    note=ParagraphStyle('n', fontName=FONT, fontSize=8.1, leading=13,
                        textColor=colors.HexColor('#262626')),
)

W = A4[0] - 34 * mm


def table(rows, widths):
    data = [[Paragraph(c, S['head']) for c in rows[0]]]
    for row in rows[1:]:
        data.append([Paragraph(row[0], S['code']),
                     Paragraph(row[1], S['cell']),
                     Paragraph(row[2], S['cell'])])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_H]),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    return t


def note(text):
    t = Table([[Paragraph(text, S['note'])]], colWidths=[W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_N),
        ('BOX', (0, 0), (-1, -1), 0.9, BD_N),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


H = ['열 이름', '뜻', '방향']
COLS = [46 * mm, W - 46 * mm - 16 * mm, 16 * mm]

FILES = [
    ['파일', '내용', '언제'],
    ['results/runs/&lt;tag&gt;/<br/><b>results.json</b>',
     '우리 하네스가 계산한 내부 지표. 런당 값 하나씩. '
     '<b>모델이 좋아졌는지 판단할 때 보는 파일.</b>', '학습 직후<br/>자동'],
    ['.../celleval/<br/><b>agg_results.csv</b>',
     'cell-eval 지표의 요약 통계. 행이 통계량(count/mean/std/min/25%/50%/75%/max)이고 '
     '열이 지표. 보통 <b>mean 행</b>만 읽으면 된다.', 'run_celleval<br/>실행 후'],
    ['.../celleval/<br/><b>results.csv</b>',
     '같은 지표를 <b>조건별로</b> 남긴 것. 37행. 조건별 대응표본 비교나 '
     '이상치 추적에 쓴다.', '동일'],
    ['.../celleval/<br/><b>real_de.csv</b><br/><b>pred_de.csv</b>',
     '차등발현(DE) 검정 원본. (조건 × 유전자) 행. 위 DE 지표들이 여기서 계산된다.',
     '동일'],
]

OURS = [
    H,
    ['n_evaluated', '채점된 test 이중 섭동 조건 수. 단일 항이 빠진 조건은 제외되므로 '
                    '기대값보다 작을 수 있다.', '—'],
    ['resid_R2_pooled',
     '비가법 잔차를 얼마나 복원했는가. <b>0점이 <font name="%s">m_A + m_B - m_ctrl</font> '
     '공식과 동률</b>이고, 음수면 그 산수보다 못하다는 뜻. 조건별 제곱합을 모아서 한 번에 '
     '계산한다(pooled).' % FONT, '↑'],
    ['resid_R2_mean', '같은 값을 조건별로 계산한 뒤 평균. 작은 조건이 과대 반영되므로 '
                      'pooled 쪽을 우선한다.', '↑'],
    ['edist_rel',
     '예측 세포와 실제 세포 사이의 energy distance를, 대조군과 실제 사이의 같은 값으로 '
     '나눈 것. <b>0이면 완벽, 1이면 대조군과 동급</b>.', '↓'],
    ['r_de20', '참 delta 상위 20개 유전자에서의 Pearson. 모든 베이스라인에서 0.98 근처로 '
               '<b>포화되어 모델을 구분하지 못한다</b>. 기록만 하고 판단에는 쓰지 않는다.', '↑'],
    ['edist_rel_autoencoder_floor',
     'transport를 끄고 오토인코더만 통과시킨 edist_rel. 이 모델이 낼 수 있는 <b>하한</b>이다. '
     '<b>1을 넘으면</b> 인코딩·디코딩이 대조군 세포를 참 분포에서 오히려 더 멀리 보낸다는 뜻.', '↓'],
]

PAIR = [
    H,
    ['mse', '각 섭동의 대조군 대비 평균제곱오차.', '↓'],
    ['mae', '각 섭동의 대조군 대비 평균절대오차.', '↓'],
    ['mse_delta', '섭동-대조군 <b>변화량</b>에서 잰 평균제곱오차. 공통 기저가 빠져 보통 '
                  '더 정보량이 많다.', '↓'],
    ['mae_delta', '섭동-대조군 변화량에서 잰 평균절대오차. '
                  '(cell-eval 소스의 설명문에는 "Mean squared error"라고 적혀 있으나 '
                  '이름과 구현은 MAE다 — 상류 오타.)', '↓'],
    ['pearson_delta', '대조군 대비 평균 변화량끼리의 Pearson 상관. '
                      '<b>논문 표의 Pearson Δ</b>.', '↑'],
    ['discrimination_score_l1',
     '예측 pseudobulk가 자기 조건의 참 pseudobulk에 얼마나 가까운지를 <b>정규화된 순위</b>로 '
     '점수화. 1이 최고, 0이 최악. 기본적으로 타깃 유전자를 제외하지만, 우리 double 라벨'
     '(AHR+FEV)은 개별 유전자명과 일치하지 않아 <b>제외가 실질적으로 작동하지 않는다</b>. '
     '<b>논문 표의 DS</b>.', '↑'],
    ['discrimination_score_l2', '위와 같되 거리 척도가 L2.', '↑'],
    ['discrimination_score_cosine', '위와 같되 거리 척도가 코사인.', '↑'],
    ['pearson_edistance', '예측과 실제의 "대조군으로부터의 energy distance"끼리 구한 Pearson 상관.', '↑'],
    ['clustering_agreement', '실제 섭동 중심점과 예측 섭동 중심점의 군집 구조 일치도.', '↑'],
]

DE = [
    H,
    ['overlap_at_N<br/>overlap_at_50 / 100 / 200 / 500',
     '참 상위 k개 DE 유전자와 예측 상위 k개의 겹침. <font name="%s">_at_N</font>은 k를 '
     '해당 조건의 실제 유의 유전자 수로 잡는다.' % FONT, '↑'],
    ['precision_at_N<br/>precision_at_50 / 100 / 200 / 500',
     '같은 상위 k 집합에 대한 정밀도.', '↑'],
    ['de_spearman_sig', '유의 DE 유전자 <b>개수</b>에 대한 Spearman 상관.', '↑'],
    ['de_spearman_lfc_sig',
     '유의 유전자들의 <b>log fold change</b>에 대한 Spearman 상관. '
     '<b>논문 표의 DE-Spearman ρ가 이 열이다.</b> <font name="%s">--profile full</font> '
     '에서만 계산된다(minimal에는 없음).' % FONT, '↑'],
    ['de_direction_match', 'DE 변화의 <b>방향</b>(증가/감소) 일치도.', '↑'],
    ['de_sig_genes_recall', '참 유의 유전자 중 예측이 찾아낸 비율(재현율).', '↑'],
    ['de_nsig_counts_real<br/>de_nsig_counts_pred',
     '유의 유전자의 <b>개수</b>. cell-eval이 best_value를 NONE으로 등록한 <b>점수가 아닌 '
     '진단값</b>이다. pred가 real을 크게 넘으면 예측 세포의 분산이 좁아 DE 검정이 '
     '과대 유의 판정을 내고 있다는 신호.', '—'],
    ['pr_auc', '"이 유전자가 유의한가"를 이진 분류로 보았을 때의 PR-AUC.', '↑'],
    ['roc_auc', '같은 문제의 ROC-AUC.', '↑'],
]

DECSV = [
    ['열 이름', '뜻', ''],
    ['target / reference', '비교된 조건과 그 기준(대조군).', ''],
    ['feature', '유전자 이름.', ''],
    ['target_mean / reference_mean', '해당 유전자의 각 집단 평균 발현.', ''],
    ['fold_change / percent_change', '두 평균의 비 / 변화율.', ''],
    ['statistic', '검정통계량.', ''],
    ['p_value / fdr', '원p값과 다중검정 보정(FDR) 후 값. 유의 판정은 fdr 기준.', ''],
]

MAP = [
    ['논문 표 열', '우리 쪽 출처', ''],
    ['L2', '<font name="%s">scripts/paper_table.py</font>가 직접 계산 '
           '(cell-eval에 없는 지표).' % FONT, ''],
    ['MSE / MAE', 'agg_results.csv 의 <font name="%s">mse</font> / '
                  '<font name="%s">mae</font>' % (FONT, FONT), ''],
    ['DE-Spearman ρ', 'agg_results.csv 의 <font name="%s">de_spearman_lfc_sig</font>' % FONT, ''],
    ['Pearson Δ', 'agg_results.csv 의 <font name="%s">pearson_delta</font>' % FONT, ''],
    ['DS', 'agg_results.csv 의 <font name="%s">discrimination_score_l1</font>' % FONT, ''],
    ['Pearson Δ^ / Δ^(20)',
     '<b>계산 불가.</b> 정의가 학습셋에 있는 해당 섭동의 centroid를 요구하는데 이 split은 '
     '이중 섭동을 통째로 홀드아웃하며, 세포 간 1:1 대응도 없다.', ''],
]


def build():
    story = [
        Paragraph('평가 지표 사전', S['title']),
        Paragraph('results/ 아래 CSV·JSON의 열이 각각 무엇을 뜻하는가', S['sub']),
        Paragraph('PGFM · cell-eval 0.5.42 · 값 해석이 아니라 열 이름의 정의 · Δ^ 는 논문의 Δ-hat', S['meta']),

        Paragraph('1. 어떤 파일에 무엇이 들어 있는가', S['h1']),
        table(FILES, COLS),

        Paragraph('2. results.json — 우리 하네스의 내부 지표', S['h1']),
        Paragraph('학습이 끝나면 자동으로 생긴다. cell-eval을 돌리지 않아도 존재한다.', S['p']),
        table(OURS, COLS),
        Spacer(1, 5),
        note('<b>판단은 이 표로 한다.</b> 12개 런에서 실측한 값의 폭을 보면 '
             'resid_R2 가 0.899 움직이는 동안 pearson_delta 는 0.011, '
             'de_spearman_lfc_sig 는 0.015 밖에 움직이지 않았다. '
             'cell-eval 지표는 논문 표를 채우고 외부와 같은 자로 재기 위한 것이지, '
             '개발 중 "이 변경이 도움이 됐나"를 가리기에는 해상도가 부족하다.'),

        Paragraph('3. cell-eval — 평균·분포 계열 (ANNDATA_PAIR)', S['h1']),
        Paragraph('예측 세포 집단과 실제 세포 집단을 직접 비교한다. '
                  '아래 설명과 방향은 cell-eval 레지스트리에 등록된 값 그대로다.', S['p']),
        table(PAIR, COLS),

        Paragraph('4. cell-eval — 차등발현(DE) 계열', S['h1']),
        Paragraph('두 집단에 각각 DE 검정을 돌린 뒤 그 결과끼리 비교한다. '
                  '따라서 예측 세포의 분산이 왜곡되면 이 계열 전체가 함께 왜곡된다.', S['p']),
        table(DE, COLS),

        Paragraph('5. real_de.csv / pred_de.csv 의 열', S['h1']),
        table(DECSV, COLS),

        Paragraph('6. 논문 표 8열과의 대응', S['h1']),
        table(MAP, [34 * mm, W - 34 * mm - 4 * mm, 4 * mm]),
        Spacer(1, 5),
        note('<b>화살표 방향 주의.</b> ↑ 는 클수록, ↓ 는 작을수록 좋다는 뜻이고, '
             '— 는 점수가 아니라 진단값이라 좋고 나쁨이 정의되지 않는다는 뜻이다. '
             'cell-eval 이 각 지표에 등록한 best_value(ONE / ZERO / NONE)를 그대로 옮겼다.'),
    ]
    out = os.path.join('results', 'METRICS.pdf')
    os.makedirs('results', exist_ok=True)
    SimpleDocTemplate(out, pagesize=A4,
                      leftMargin=17 * mm, rightMargin=17 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm,
                      title='PGFM 평가 지표 사전').build(story)
    print(f"-> {out}")


if __name__ == '__main__':
    build()
