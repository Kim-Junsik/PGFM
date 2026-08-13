# -*- coding: utf-8 -*-
"""docs/ARCHITECTURE.pdf - 외부 전달용 아키텍처 설명서."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import math
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Flowable, KeepTogether)

pdfmetrics.registerFont(TTFont('KR', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KR-B', 'C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('KR', normal='KR', bold='KR-B', italic='KR', boldItalic='KR-B')

NAVY = colors.HexColor('#1F3864')
BLUE = colors.HexColor('#2E75B6')
TEAL = colors.HexColor('#2F8F83')
PLUM = colors.HexColor('#8B5A8F')
AMBER = colors.HexColor('#C8862A')
GREY = colors.HexColor('#7F7F7F')
LINE = colors.HexColor('#BFBFBF')
BG_B = colors.HexColor('#EDF2F9')
BG_P = colors.HexColor('#FDF0EF')
BG_Y = colors.HexColor('#FDF8E8')
BG_C = colors.HexColor('#F2F4F7')
BG_T = colors.HexColor('#E9F4F2')
BG_M = colors.HexColor('#F5EEF6')
BD_P = colors.HexColor('#C55A5A')
BD_Y = colors.HexColor('#E0A030')

W = A4[0] - 40 * mm

S = dict(
    title=ParagraphStyle('t', fontName='KR-B', fontSize=21, textColor=NAVY,
                         alignment=1, spaceAfter=4, leading=26),
    sub=ParagraphStyle('s', fontName='KR', fontSize=10.5, textColor=colors.HexColor('#595959'),
                       alignment=1, spaceAfter=3, leading=14),
    meta=ParagraphStyle('m', fontName='KR', fontSize=8, textColor=GREY,
                        alignment=1, spaceAfter=14, leading=11),
    h1=ParagraphStyle('h1', fontName='KR-B', fontSize=13.5, textColor=NAVY,
                      spaceBefore=15, spaceAfter=7, leading=17),
    h2=ParagraphStyle('h2', fontName='KR-B', fontSize=10.5, textColor=BLUE,
                      spaceBefore=10, spaceAfter=5, leading=14),
    p=ParagraphStyle('p', fontName='KR', fontSize=8.8, leading=14.5,
                     spaceAfter=6, textColor=colors.HexColor('#262626')),
    cap=ParagraphStyle('cap', fontName='KR', fontSize=7.6, leading=11,
                       alignment=1, textColor=GREY, spaceBefore=3, spaceAfter=8),
    cell=ParagraphStyle('c', fontName='KR', fontSize=8, leading=12.2,
                        textColor=colors.HexColor('#262626')),
    cellh=ParagraphStyle('ch', fontName='KR-B', fontSize=8, leading=12.2,
                         textColor=colors.white),
    code=ParagraphStyle('co', fontName='KR', fontSize=7.8, leading=12.5,
                        textColor=colors.HexColor('#1A1A1A')),
    note=ParagraphStyle('n', fontName='KR', fontSize=8.2, leading=13.5,
                        textColor=colors.HexColor('#262626')),
)


def P(t, s='p'):
    return Paragraph(t, S[s])


def tbl(rows, widths, align=None):
    data = [[Paragraph(c, S['cellh']) for c in rows[0]]] + \
           [[Paragraph(c, S['cell']) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1, hAlign='LEFT')
    st = [('BACKGROUND', (0, 0), (-1, 0), NAVY),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 4.5),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
          ('LEFTPADDING', (0, 0), (-1, -1), 6),
          ('RIGHTPADDING', (0, 0), (-1, -1), 6),
          ('LINEBELOW', (0, 0), (-1, -1), 0.4, LINE),
          ('BOX', (0, 0), (-1, -1), 0.4, LINE),
          ('LINEAFTER', (0, 0), (-2, -1), 0.4, LINE)]
    for i in range(2, len(data), 2):
        st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F7F9FC')))
    t.setStyle(TableStyle(st))
    return t


def box(html, kind='b'):
    bg, bd = {'b': (BG_B, NAVY), 'p': (BG_P, BD_P), 'y': (BG_Y, BD_Y)}[kind]
    t = Table([[Paragraph(html, S['note'])]], colWidths=[W], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), bg),
                           ('LINEBEFORE', (0, 0), (0, -1), 2.2, bd),
                           ('TOPPADDING', (0, 0), (-1, -1), 7),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 9)]))
    return t


def code(lines):
    keep = [re.sub(r'  +', lambda m: '&nbsp;' * len(m.group()), ln) for ln in lines]
    t = Table([[Paragraph('<br/>'.join(keep), S['code'])]], colWidths=[W], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), BG_C),
                           ('TOPPADDING', (0, 0), (-1, -1), 7),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                           ('LEFTPADDING', (0, 0), (-1, -1), 11),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 9)]))
    return t


# ----------------------------------------------------------------- diagrams
class Diagram(Flowable):
    """Vector diagram drawn straight onto the canvas."""

    def __init__(self, height, painter):
        Flowable.__init__(self)
        self.width, self.height, self.painter = W, height, painter

    def draw(self):
        self.painter(self.canv, self.width, self.height)


def rbox(c, x, y, w, h, label, sub=None, fill=colors.white, stroke=NAVY,
         lw=0.9, fs=8.2, tc=None, radius=3):
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.setLineWidth(lw)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
    c.setFillColor(tc or stroke)
    c.setFont('KR-B', fs)
    if sub:
        c.drawCentredString(x + w / 2, y + h / 2 + 2.0, label)
        c.setFont('KR', fs - 1.4)
        c.setFillColor(GREY)
        c.drawCentredString(x + w / 2, y + h / 2 - 7.5, sub)
    else:
        c.drawCentredString(x + w / 2, y + h / 2 - 3, label)


def arrow(c, x1, y1, x2, y2, colour=NAVY, lw=1.0, label=None, dashed=False):
    c.setStrokeColor(colour)
    c.setLineWidth(lw)
    if dashed:
        c.setDash(2, 2)
    c.line(x1, y1, x2, y2)
    c.setDash()
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 4.2
    c.setFillColor(colour)
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(x2 - size * math.cos(ang - 0.42), y2 - size * math.sin(ang - 0.42))
    path.lineTo(x2 - size * math.cos(ang + 0.42), y2 - size * math.sin(ang + 0.42))
    path.close()
    c.drawPath(path, stroke=0, fill=1)
    if label:
        c.setFont('KR', 7)
        c.setFillColor(GREY)
        c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 4, label)


def paint_pipeline(c, w, h):
    """cell -> latent -> transport -> latent -> cell"""
    bw, bh, y = 74, 40, h - 60
    xs = [0, 102, 192, 318, 408]      # 5 boxes + 4 gaps = 482pt, the text width
    rbox(c, xs[0], y, bw, bh, '대조군 세포', 'x0  (3,074)', BG_B, NAVY)
    rbox(c, xs[1], y, 62, bh, 'z0', '(64)', colors.white, TEAL)
    rbox(c, xs[2], y, 98, bh, 'RK4 적분', 't: 0 → 1', BG_Y, AMBER)
    rbox(c, xs[3], y, 62, bh, 'z1', '(64)', colors.white, TEAL)
    rbox(c, xs[4], y, bw, bh, '예측 세포', 'x1_hat  (3,074)', BG_B, NAVY)

    arrow(c, xs[0] + bw, y + bh / 2, xs[1], y + bh / 2, TEAL, 1.1, '인코더')
    arrow(c, xs[1] + 62, y + bh / 2, xs[2], y + bh / 2, AMBER, 1.1)
    arrow(c, xs[2] + 98, y + bh / 2, xs[3], y + bh / 2, AMBER, 1.1)
    arrow(c, xs[3] + 62, y + bh / 2, xs[4], y + bh / 2, TEAL, 1.1, '디코더')

    # velocity field feeding the integrator
    fy = y - 62
    rbox(c, 132, fy, 300, 40, 'v(z, t, S) = Σ u_a(z,t)  +  Λ_ab · [u_a, u_b](z,t)',
         'LieCFM 속도장 — 논문의 기여', colors.white, PLUM, 1.1, 8.0)
    arrow(c, 241, fy + 40, 241, y, PLUM, 1.1)

    # condition input
    c.setFont('KR', 7.4)
    c.setFillColor(GREY)
    c.drawString(0, fy + 14, '섭동 집합 S')
    c.drawString(0, fy + 4, '예: {AHR, FEV}')
    arrow(c, 60, fy + 20, 132, fy + 20, GREY, 0.8)

    c.setFont('KR', 7.2)
    c.setFillColor(GREY)
    c.drawString(0, y - 14, '측정된 대조군 세포에서 출발 — 사전분포에서 표본을 뽑지 않는다')


def paint_liecfm(c, w, h):
    """generator composition"""
    y = h - 46
    rbox(c, 0, y, 86, 34, 'u_a(z, t)', '섭동 a 의 생성원', colors.white, TEAL)
    rbox(c, 0, y - 52, 86, 34, 'u_b(z, t)', '섭동 b 의 생성원', colors.white, TEAL)

    rbox(c, 138, y - 26, 122, 34, '[u_a, u_b]', 'Lie 괄호 (jvp 2회)', BG_Y, AMBER)
    arrow(c, 86, y + 17, 138, y - 4, AMBER, 0.9)
    arrow(c, 86, y - 35, 138, y - 14, AMBER, 0.9)

    rbox(c, 286, y - 26, 60, 34, 'Λ_ab', '반대칭 게이트', colors.white, PLUM)
    arrow(c, 260, y - 9, 286, y - 9, PLUM, 0.9)

    rbox(c, 372, y - 26, 110, 34, 'v(z, t, {a,b})', '합성 속도장', BG_B, NAVY, 1.1)
    arrow(c, 346, y - 9, 372, y - 9, NAVY, 1.0, '×')
    # additive path
    c.setStrokeColor(TEAL)
    c.setLineWidth(0.9)
    c.setDash(3, 2)
    c.line(86, y + 17, 358, y + 17)
    c.line(86, y - 35, 358, y - 35)
    c.line(358, y + 17, 358, y - 2)
    c.line(358, y - 35, 358, y - 16)
    c.setDash()
    arrow(c, 358, y - 9, 372, y - 9, TEAL, 0.9)
    c.setFont('KR', 7)
    c.setFillColor(GREY)
    c.drawCentredString(222, y + 21, '가법 경로  Σ u_a   (Λ = 0 이면 이것만 남는다)')


def paint_pcab(c, w, h):
    """mask construction and multiplicative gating"""
    y = h - 44
    rbox(c, 0, y, 96, 32, 'M_prior', 'KEGG {0,1}  277×3074', BG_B, NAVY)
    rbox(c, 0, y - 46, 96, 32, 'M_residual', '학습 파라미터', colors.white, PLUM)
    c.setFont('KR-B', 11)
    c.setFillColor(GREY)
    c.drawCentredString(122, y - 18, '+')
    c.setFont('KR', 7.4)
    c.drawCentredString(122, y - 30, 'α ·')
    arrow(c, 102, y + 16, 140, y - 8, GREY, 0.8)
    arrow(c, 102, y - 30, 140, y - 12, GREY, 0.8)

    rbox(c, 142, y - 30, 74, 34, 'tanh', '(-1, 1)', BG_Y, AMBER)
    rbox(c, 240, y - 30, 86, 34, 'M_total', '부호 있는 강도', colors.white, AMBER, 1.1)
    arrow(c, 216, y - 13, 240, y - 13, AMBER, 0.9)

    rbox(c, 240, y + 22, 86, 30, 'A = softmax_g', '세포별 어텐션', colors.white, TEAL)
    c.setFont('KR-B', 11)
    c.setFillColor(GREY)
    c.drawCentredString(346, y + 2, '⊙')
    arrow(c, 326, y + 37, 360, y + 10, TEAL, 0.9)
    arrow(c, 326, y - 13, 360, y + 2, AMBER, 0.9)

    rbox(c, 364, y - 14, 100, 36, 'H = (A ⊙ M) V', '패스웨이 토큰 표현', BG_B, NAVY, 1.1)
    c.setFont('KR', 7.2)
    c.setFillColor(GREY)
    c.drawString(0, y - 62, 'logit 에 더하면 음수가 "덜 본다"가 되어 억제를 표현할 수 없다 — 곱셈이라야 실제로 빼진다')


def paint_hurdle(c, w, h):
    y = h - 40
    rbox(c, 0, y - 16, 80, 34, 'z', '잠재', colors.white, TEAL)
    rbox(c, 132, y + 12, 110, 30, 'σ(gate)', '검출 확률', colors.white, BLUE)
    rbox(c, 132, y - 44, 110, 30, 'N(μ, s²)', '발현량 분포', colors.white, PLUM)
    arrow(c, 80, y + 6, 132, y + 27, BLUE, 0.9)
    arrow(c, 80, y - 8, 132, y - 29, PLUM, 0.9)

    rbox(c, 274, y + 12, 72, 30, 'B ~ Bern', '0 또는 1', BG_Y, AMBER)
    rbox(c, 274, y - 44, 72, 30, 'M ~ 표본', '≥ 0', BG_Y, AMBER)
    arrow(c, 242, y + 27, 274, y + 27, AMBER, 0.9)
    arrow(c, 242, y - 29, 274, y - 29, AMBER, 0.9)

    c.setFont('KR-B', 12)
    c.setFillColor(GREY)
    c.drawCentredString(366, y - 6, '×')
    arrow(c, 346, y + 27, 386, y + 2, AMBER, 0.9)
    arrow(c, 346, y - 29, 386, y - 14, AMBER, 0.9)
    rbox(c, 386, y - 16, 96, 34, 'x_hat = B · M', '정확한 0 을 낸다', BG_B, NAVY, 1.1)
    c.setFont('KR', 7.2)
    c.setFillColor(GREY)
    c.drawString(0, y - 62, '평균만 내면(σ·μ) 세포간 변동이 붕괴한다 — 실측 std 0.098 vs 실제 0.495')


def op(c, x, y, symbol, colour=GREY, fs=12):
    c.setFont('KR-B', fs); c.setFillColor(colour)
    c.drawCentredString(x, y, symbol)


def paint_mask_detail(c, w, h):
    y = h - 40
    rbox(c, 0, y, 118, 34, 'M_prior', 'KEGG {0,1}   277 × 3074', BG_B, NAVY)
    rbox(c, 0, y - 50, 118, 34, 'M_residual', '학습 파라미터  851,498', colors.white, PLUM)
    op(c, 140, y - 12, '+')
    c.setFont('KR', 7); c.setFillColor(GREY); c.drawCentredString(140, y - 24, 'α = 1.0')
    arrow(c, 118, y + 17, 160, y - 6, GREY, 0.8)
    arrow(c, 118, y - 33, 160, y - 12, GREY, 0.8)
    rbox(c, 162, y - 30, 70, 34, 'tanh', '부호 도입', BG_Y, AMBER)
    arrow(c, 232, y - 13, 262, y - 13, AMBER)
    rbox(c, 264, y - 30, 106, 34, 'M_total', '(-1, 1)  277 × 3074', BG_M, PLUM, 1.1)
    c.setFont('KR', 7.2); c.setFillColor(GREY)
    c.drawString(388, y + 6, '주석 토큰  176')
    c.drawString(388, y - 5, '자유 토큰  101')
    c.drawString(388, y - 16, '밀도  0.879 %')
    c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(384, y - 24, 384, y + 16)



def paint_pcab_encoder(c, w, h):
    top = h - 34
    # inputs
    rbox(c, 0, top - 12, 74, 32, 'x', '세포 발현  B × 3074', BG_B, NAVY)
    rbox(c, 0, top - 74, 74, 28, 'Q', '패스웨이 쿼리 277×64', colors.white, PLUM)
    rbox(c, 0, top - 112, 74, 28, 'gene_key', '3074 × 64', colors.white, TEAL)
    rbox(c, 0, top - 150, 74, 28, 'gene_value', '3074 × 64', colors.white, TEAL)

    # base scores (batch independent)
    rbox(c, 104, top - 88, 96, 34, 'Q · gene_key\u1d40', '277 × 3074  (1회)', BG_T, TEAL)
    arrow(c, 74, top - 60, 104, top - 66, TEAL, 0.85)
    arrow(c, 74, top - 98, 104, top - 74, TEAL, 0.85)

    op(c, 224, top - 74, '⊙')
    arrow(c, 200, top - 71, 212, top - 71, TEAL, 0.85)
    arrow(c, 37, top - 12, 37, top + 2, GREY, 0.0)          # spacer
    c.setStrokeColor(NAVY); c.setLineWidth(0.85)
    c.line(37, top - 12, 37, top - 22); c.line(37, top - 22, 224, top - 22)
    arrow(c, 224, top - 22, 224, top - 66, NAVY, 0.85, 'x', 8)

    rbox(c, 242, top - 88, 84, 34, 'scores', 'B × 277 × 3074', colors.white, NAVY)
    arrow(c, 236, top - 71, 242, top - 71, NAVY, 0.85)
    rbox(c, 344, top - 88, 76, 34, 'softmax_g', '행 합 = 1', BG_Y, AMBER)
    arrow(c, 326, top - 71, 344, top - 71, AMBER)

    # gating row
    bot = top - 152
    rbox(c, 344, bot, 76, 30, 'M_total', '277 × 3074', BG_M, PLUM)
    op(c, 382, bot + 44, '⊙', PLUM, 13)
    arrow(c, 382, top - 88, 382, bot + 52, AMBER, 0.85)
    arrow(c, 382, bot + 30, 382, bot + 38, PLUM, 0.85)

    rbox(c, 240, bot, 84, 30, 'weights ⊙ x', 'B × 277 × 3074', colors.white, NAVY)
    arrow(c, 344, bot + 15, 324, bot + 15, NAVY, 0.85)
    c.setStrokeColor(NAVY); c.setLineWidth(0.85)
    c.line(37, top - 22, 37, bot + 15); c.line(37, bot + 15, 240, bot + 15)

    rbox(c, 120, bot, 96, 30, '@ gene_value', '→ B × 277 × 64', BG_T, TEAL)
    arrow(c, 240, bot + 15, 216, bot + 15, TEAL, 0.85)
    c.setStrokeColor(TEAL); c.setLineWidth(0.85)
    c.line(37, top - 150, 37, bot - 10); c.line(37, bot - 10, 168, bot - 10)
    arrow(c, 168, bot - 10, 168, bot, TEAL, 0.85)

    rbox(c, 0, bot - 56, 96, 30, 'LayerNorm', 'H  B × 277 × 64', colors.white, NAVY)
    arrow(c, 120, bot + 15, 60, bot + 15, NAVY, 0.0)
    c.setStrokeColor(NAVY); c.line(120, bot + 15, 108, bot + 15)
    c.line(108, bot + 15, 108, bot - 41); arrow(c, 108, bot - 41, 96, bot - 41, NAVY, 0.85)

    rbox(c, 140, bot - 56, 96, 30, 'flatten', 'B × 17,728', colors.white, NAVY)
    arrow(c, 96, bot - 41, 140, bot - 41, NAVY, 0.85)
    rbox(c, 268, bot - 56, 100, 30, 'μ , logvar', 'B × 64', BG_B, NAVY, 1.1)
    arrow(c, 236, bot - 41, 268, bot - 41, NAVY)



def paint_pcab_decoder(c, w, h):
    y = h - 36
    rbox(c, 0, y - 14, 66, 30, 'z', 'B × 64', BG_B, NAVY)
    rbox(c, 0, y - 62, 66, 26, 'token_emb', '277 × 64', colors.white, PLUM)
    rbox(c, 96, y - 34, 96, 34, '공유 MLP', '토큰마다 평가', BG_M, PLUM)
    arrow(c, 66, y + 1, 96, y - 8, PLUM, 0.85)
    arrow(c, 66, y - 49, 96, y - 24, PLUM, 0.85)
    rbox(c, 214, y - 34, 92, 34, 'tokens', 'B × 277 × 64', colors.white, PLUM)
    arrow(c, 192, y - 17, 214, y - 17, PLUM)

    # attention side
    rbox(c, 0, y - 118, 66, 26, 'gene_query', '3074 × 64', colors.white, TEAL)
    rbox(c, 0, y - 156, 66, 26, 'token_key', '277 × 64', colors.white, TEAL)
    rbox(c, 96, y - 148, 96, 32, 'softmax_k', '3074 × 277 (1회)', BG_Y, AMBER)
    arrow(c, 66, y - 105, 96, y - 124, AMBER, 0.85)
    arrow(c, 66, y - 143, 96, y - 132, AMBER, 0.85)
    rbox(c, 214, y - 148, 92, 32, 'M_total\u1d40', '3074 × 277', BG_M, PLUM)
    op(c, 202, y - 130, '⊙', PLUM, 12)
    arrow(c, 306, y - 132, 330, y - 132, PLUM, 0.85)
    rbox(c, 332, y - 148, 90, 32, 'weights', '3074 × 277', colors.white, NAVY)

    op(c, 352, y - 84, '@', NAVY, 13)
    arrow(c, 352, y - 116, 352, y - 94, NAVY, 0.85)
    arrow(c, 306, y - 17, 352, y - 74, NAVY, 0.85)
    rbox(c, 332, y - 46, 90, 34, 'h', 'B × 3074 × 64', BG_T, TEAL, 1.1)





def footer(canv, doc):
    canv.saveState()
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.4)
    canv.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canv.setFont('KR', 7.5)
    canv.setFillColor(GREY)
    canv.drawString(20 * mm, 10.5 * mm, 'LT-VAE + LieCFM · 모델 아키텍처')
    canv.drawRightString(A4[0] - 20 * mm, 10.5 * mm, str(doc.page))
    canv.restoreState()


s = []
s += [P('LT-VAE + LieCFM', 'title'),
      P('조합 섭동 반응 예측 — 모델 아키텍처', 'sub'),
      P('PGFM 프로젝트 · 2026-08-12 · 외부 전달용', 'meta')]

s += [box('<b>한 줄 요약.</b> 대조군 세포를 잠재 공간으로 인코딩한 뒤, <b>섭동별 벡터장과 그 Lie 괄호</b>로 '
          '구성된 속도장을 따라 수송하여 섭동 상태의 세포를 만든다. 학습에서 함께 본 적 없는 유전자 조합 '
          '(A+B)의 반응을 예측하는 것이 목표다.', 'b')]

# ---------------------------------------------------------------- 1
s += [P('1. 무엇을 푸는 문제인가', 'h1')]
s += [P('단일세포 CRISPRa 데이터(Norman, K562 세포주)에서 <b>두 유전자를 동시에 활성화했을 때의 발현 반응</b>을 '
        '예측한다. 데이터는 84,986 세포 × 19,264 유전자이고, 조건은 대조군 1개 + 단일 섭동 101개 + '
        '이중 섭동 125개다.', 'p')]
s += [P('근본적인 난이도가 두 가지 있다.', 'p')]
s += [tbl([
    ['난이도', '내용', '이 설계의 대응'],
    ['<b>짝이 없음</b>', '대조군 세포와 섭동 세포는 서로 다른 세포다. 같은 세포의 전후를 관측할 수 없다',
     '조건별 <b>불균형 최적수송(UOT)</b>으로 미니배치 안에서 대응을 만든다. 무작위 짝짓기는 대조군으로만 사용'],
    ['<b>조합 외삽</b>', '평가 대상인 A+B 조합은 학습에서 한 번도 관측되지 않았다',
     '<b>쌍으로 색인되는 파라미터를 두지 않는다.</b> 상호작용을 섭동별 생성원만으로 구성하므로, '
     '본 적 없는 쌍도 표현 가능하다'],
], [0.16, 0.42, 0.42])]

s += [P('2. 전체 데이터 흐름', 'h1')]
s += [Diagram(150, paint_pipeline)]
s += [P('그림 1. 예측은 생성이 아니라 <b>수송</b>이다. 사전분포에서 표본을 뽑는 것이 아니라, '
        '실제 대조군 세포를 출발점으로 삼아 속도장을 따라 이동시킨다. 따라서 예측 집단은 '
        '대조군 집단이 원래 가진 이질성을 물려받는다.', 'cap')]

s += [code([
    'x0  ──encode──▶  z0  ──∫v dt──▶  z1  ──decode──▶  x1_hat',
    '',
    '  x0 : 대조군 세포의 발현 벡터 (log1p, 3,074 유전자)',
    '  z0 : 표준화된 잠재 (64차원)',
    '  v  : 섭동 집합 S 에 조건화된 속도장',
    '  x1_hat : 섭동 상태로 예측된 세포',
])]

# ---------------------------------------------------------------- 3
s += [P('3. 핵심 — LieCFM 잠재 동역학', 'h1')]
s += [P('논문의 첫 번째 기여다. 섭동 a 마다 잠재 공간 위의 벡터장 u_a(z,t) 를 두고, '
        '두 섭동이 함께 가해질 때의 상호작용을 <b>Lie 괄호</b>로 표현한다.', 'p')]
s += [code([
    'v(z, t, S)  =  Σ<sub>a∈S</sub> u_a(z,t)   +   Σ<sub>{a,b}⊂S</sub> Λ_ab · [u_a, u_b](z,t)',
    '',
    '[u_a, u_b](z)  =  J_b(z) u_a(z)  -  J_a(z) u_b(z)',
])]
s += [Diagram(112, paint_liecfm)]
s += [P('그림 2. 가법 경로(점선)와 상호작용 경로. Λ = 0 이면 정확히 가법 모델로 되돌아간다.', 'cap')]

s += [box('<b>왜 Lie 괄호인가.</b> 두 벡터장을 차례로 따라갔을 때 순서를 바꾸면 도착점이 달라지는데, '
          '그 차이를 1차로 기술하는 양이 정확히 Lie 괄호다. "두 섭동이 서로의 효과를 어떻게 바꾸는가"라는 '
          '생물학적 질문과 대응한다.<br/><br/>'
          '<b>왜 계산이 가능한가.</b> J 를 명시적으로 만들면 표본당 O(d²) 메모리가 들어 비선형 생성원에는 '
          '쓸 수 없다. 두 항 모두 야코비-벡터 곱이므로 <b>forward-mode 자동미분(jvp) 2회</b>로 '
          'O(d) 에 계산한다. 명시적 야코비안과 1e-8 까지 일치함을 테스트로 확인했다.', 'y')]

s += [P('3.1 구조적으로 보장되는 성질 4가지', 'h2')]
s += [P('학습으로 얻는 것이 아니라 <b>구성상 성립</b>하며, 각각 수치 단언으로 검증한다(14개 테스트 통과).', 'p')]
s += [tbl([
    ['성질', '내용', '근거'],
    ['대조군 불변', 'v(z, t, { }) = 0', '빈 합'],
    ['단일 직접 감독', 'v(z, t, {a}) = u_a(z,t)', '쌍 항이 존재하지 않음'],
    ['가법 정확 복원', 'Λ = 0 이면 정확히 Σ u_a', '곱해지는 항이 0'],
    ['순열 불변', 'v(·,{a,b}) = v(·,{b,a})', 'Λ 과 괄호가 모두 반대칭 → 곱은 대칭'],
], [0.20, 0.40, 0.40])]

s += [P('3.2 생성원의 두 가지 형태', 'h2')]
s += [tbl([
    ['형태', '정의', '의미'],
    ['<b>affine</b>', 'u_a(z,t) = A_a z + b_a', '괄호가 bilinear Koopman 표준형 — 이미 알려진 형태. '
     '이쪽이 이기면 "고차 동역학이 불필요하다"는 발견이 된다'],
    ['<b>neural_field</b>', '공유 MLP + 섭동별 임베딩', '<b>비선형 벡터장의 괄호</b>를 AD 로 계산하는 부분이 새롭다'],
], [0.16, 0.30, 0.54])]

# ---------------------------------------------------------------- 4
s += [P('4. 표현 백본 — 교체 가능한 축', 'h1')]
s += [P('기여는 동역학이지 인코더가 아니다. 같은 속도장이 <b>여러 표준 표현 위에서 작동함</b>을 보이는 것이 '
        '"운 좋은 아키텍처 하나"가 아니라는 근거가 되므로, 백본은 설정 축으로 두었다. '
        '네 가지 모두 동일한 인터페이스를 만족하여 학습·평가 코드는 무엇이 올라갔는지 알지 못한다.', 'p')]
s += [tbl([
    ['백본', '구조', '파라미터', '비고'],
    ['<b>mlp</b>', '평범한 MLP VAE', '13.8 M', '현재 기준점'],
    ['<b>transformer</b>', 'Perceiver 방식 — 유전자를 32개 토큰으로 압축 후 self-attention',
     '11.8 M', '유전자 간 full self-attention 은 세포당 945만 쌍이라 불가'],
    ['<b>scvi</b>', 'MLP 인코더 + library 헤드 + ZINB 디코더', '10.6 M',
     'raw count 가 정확히 복원되므로 사용 가능'],
    ['<b>pcab</b>', '패스웨이 cross-attention (아래 5장)', '<b>3.8 M</b>', '논문의 두 번째 기여'],
], [0.14, 0.42, 0.12, 0.32])]

# ---------------------------------------------------------------- 5
s += [P('5. P-CAB 인코더 / E-RCA 디코더', 'h1')]
s += [P('두 번째 기여다. K 개의 학습된 <b>패스웨이 토큰</b>이 유전자에 cross-attention 한다. '
        'Transformer 가 아니다 — self-attention 을 적층하지 않고 cross-attention 한 겹이며, '
        '복잡도가 O(G²) 가 아니라 <b>O(G·K)</b> 다. 4장의 백본 중 하나이지 별도 모델이 아니다.', 'p')]

s += [P('5.1 마스크 — 생물학적 prior 가 들어오는 곳', 'h2')]
s += [Diagram(104, paint_mask_detail)]
s += [P('그림 3. M_total 은 학습되며 부호를 가진다. 자유 토큰은 prior 가 전부 0 이라 '
        'tanh(α · M_residual) 이 되어 처음부터 학습된다.', 'cap')]
s += [code(['M_total  =  tanh( M_prior  +  α · M_residual )        ∈ (-1, 1)<sup>277 × 3074</sup>'])]
s += [box('<b>가산이 아니라 곱셈이어야 하는 이유.</b> softmax logit 에 더하면 음수 항목은 "덜 본다"를 뜻할 뿐이고, '
          '값 집계는 여전히 V 의 볼록 결합이다. 즉 <b>억제(repression)는 표현 가능한 함수 공간 안에 없다.</b> '
          '곱셈 게이트에서는 M_total 이 부호 있는 연결 강도가 되어 음수가 실제로 빼진다.<br/><br/>'
          'KEGG 도 GRNBoost2 도 부호를 제공하지 않으므로 <b>억제는 전부 M_residual 이 배워야 한다.</b> '
          'α 는 그 예산이며, 주석된 엣지의 부호를 뒤집으려면 α·|M_residual| &gt; 1 이 필요하다.', 'y')]
s += [P('KEGG 주석 패스웨이 176개(질병 제외, 크기 10~300) + <b>자유 토큰 101개</b> = K 277, 밀도 0.879 %. '
        '섭동 표적 101개 중 47개만 유효 패스웨이에 속한다 — 나머지는 HOX·FOX·DLX 같은 발생 TF 로 '
        'KEGG 가 담지 않는 계열이라, prior 가 비어 있는 행을 붙여 모델이 직접 채우게 한다.', 'p')]

s += [KeepTogether([P('5.2 인코더', 'h2'), Diagram(268, paint_pcab_encoder)])]
s += [P('그림 4. 277개 패스웨이 토큰이 3,074개 유전자에 cross-attention 한다. B = 배치. '
        '잠재는 패스웨이 토큰들의 요약이다.', 'cap')]
s += [tbl([
    ['설계', '이유'],
    ['<b>cross-attention 1겹</b>', '유전자 간 self-attention 은 세포당 945만 쌍(전체 유전자면 3.7억)이라 '
     '8.6 GB 로는 불가능하다. 쿼리를 K개 토큰으로 두면 O(G·K)'],
    ['<b>키·값을 발현으로 스케일</b>', '어텐션이 <b>세포마다 달라진다</b>. 마스크만으로는 모든 세포가 같은 '
     '패턴을 갖게 되는데, 그것이 기존 masked-linear-decoder 계열이 하는 일이다'],
], [0.24, 0.76])]

s += [P('5.3 계산을 줄이는 두 가지 대수 변형', 'h2')]
s += [code([
    'score[b,k,g]  =  Σ<sub>d</sub> Q[k,d] · ( gene_key[g,d] · x[b,g] )',
    '              =  ( Q · gene_key<sup>T</sup> )[k,g]  ·  x[b,g]        <- [K,G] 는 배치 무관',
    '',
    'H[b,k,:]      =  Σ<sub>g</sub> ( A[b,k,g] · M[k,g] · x[b,g] ) · gene_value[g,:]',
])]
s += [tbl([
    ['항목', '변형 전', '변형 후', '배수'],
    ['encode', '20.8 ms', '<b>7.1 ms</b>', '2.9×'],
    ['decode', '8.0 ms', '<b>3.4 ms</b>', '2.4×'],
    ['forward + backward', '84.1 ms', '<b>31.3 ms</b>', '2.7×'],
    ['build_prior (마스크 조립)', '수 분', '<b>0.2 s</b>', '—'],
], [0.40, 0.20, 0.20, 0.20])]
s += [Spacer(1, 4)]
s += [P('배치 32, RTX 3050 8.6 GB. 참조 einsum 구현과 최대 오차 3.73e-09 로 동등함을 확인했다.', 'p')]

s += [KeepTogether([P('5.4 디코더 (E-RCA)', 'h2'), Diagram(206, paint_pcab_decoder)])]
s += [P('그림 5. 유전자 쿼리가 패스웨이 토큰에 attention 하여 유전자별 표현을 만든다. '
        '읽어내기는 유전자별 사영이며 헤드 전체가 9,414 파라미터다.', 'cap')]
s += [box('<b>출력층은 반드시 유전자별 사영이어야 한다.</b> Flatten + Linear 로 두면 전체 유전자 기준 '
          '<b>237억 파라미터</b>가 된다. 공유 방향 + 유전자별 bias 로 두면 9,414개이고, '
          '유전자 정체성은 이미 표현 h 안에 있으므로 잃는 것이 없다. 같은 함정을 디코더 입력에서 한 번 더 '
          '밟았다 — z → (K · d_v) 를 dense 로 두니 631 M 이 나왔고, 토큰 임베딩을 조건으로 받는 '
          '<b>공유 MLP 를 토큰마다 평가</b>하는 형태로 바꿔 25 K 로 줄였다.<br/><br/>'
          '<b>마스크는 인코더와 공유한다.</b> 억제 엣지 하나를 두 방향에서 보면 디코더는 "패스웨이 활성↑ → '
          '유전자↓", 인코더는 "그 유전자가 낮게 관측됨 → 패스웨이 활성 높음" 이고 둘 다 같은 음수 가중이다. '
          '따로 두면 같은 엣지에 모순된 부호를 학습할 수 있다.', 'p')]

s += [P('5.5 파라미터 구성', 'h2')]
s += [tbl([
    ['구성', '파라미터', '비고'],
    ['mask.residual (277 × 3074)', '851,498', '학습되는 마스크 보정'],
    ['to_mu / to_logvar', '2,269,312', '가장 큰 항목'],
    ['gene_key / gene_value', '393,472', ''],
    ['from_latent (공유 MLP)', '25,152', 'dense 였다면 631 M'],
    ['head (유전자별 사영)', '<b>9,414</b>', 'Flatten 이었다면 54.5 M'],
    ['<b>합계</b>', '<b>3,799,025</b>', 'mlp 백본은 13.8 M'],
], [0.40, 0.24, 0.36])]
s += [Spacer(1, 5)]
s += [P('<font face="KR-B">PathwayMask.learned_edges()</font> 는 M_total 에서 활성화된 prior 를 뺀 값, '
        '즉 <b>모델이 KEGG 주석에 추가하거나 삭제한 엣지</b>를 돌려준다. P-CAB 의 기여를 "인코더 교체"가 '
        '아니라 <b>"섭동 반응 데이터로 패스웨이 주석을 정정한다"</b>로 제시할 수 있고, 선행 패스웨이 VAE '
        '(VEGA / expiMap / pmVAE) 대비 차별점을 문장이 아니라 그림으로 답하게 된다.', 'p')]

s += [P('6. 디코더 — hurdle 분해', 'h1')]
s += [P('VAE 디코더는 원래 분포 p(x|z) 다. 그런데 MSE 로 학습하면 p(x|z) = N(f(z), σ²I) 에서 σ² 를 '
        '상수로 고정한 것과 같아, 신경망은 <b>평균만</b> 배우고 추론은 E[x|z] 를 돌려준다. '
        'VAE 이미지 복원이 뿌옇게 나오는 것과 같은 현상이며, 여기서는 <b>세포간 변동의 붕괴</b>로 나타난다.', 'p')]
s += [Diagram(112, paint_hurdle)]
s += [P('그림 6. 두 성분 모두 평균이 아니라 표본으로 실현한다.', 'cap')]
s += [P('데이터의 41.2% 가 정확히 0 인데 MSE 헤드는 정확한 0 을 한 번도 만들지 못한다. '
        'hurdle 분해는 <b>검출 여부</b>와 <b>발현량</b>을 분리하여 각각 확률적으로 실현한다. '
        '평가 지표가 분포 간 거리이므로 이 차이가 그대로 점수에 반영된다.', 'p')]
s += [tbl([
    ['게이트 실현', '세포간 std', '0 비율', 'edist_rel'],
    ['soft  σ × μ (조건부 기댓값)', '0.0977', '0.0%', '6.637'],
    ['hard  임계값 0.5', '0.2317', '35.6%', '4.408'],
    ['<b>sample  Bernoulli(σ)</b>', '<b>0.3361</b>', '<b>38.1%</b>', '<b>1.461</b>'],
    ['<i>실제 데이터</i>', '<i>0.4947</i>', '<i>49.4%</i>', '—'],
], [0.40, 0.20, 0.18, 0.22])]
s += [Spacer(1, 5)]
s += [P('학습은 확률로 해야 한다(하드 임계값은 미분 불가). 추론에서만 실현 방식을 고른다. '
        '게이트 자체는 잘 학습된다 — BCE 0.5118 로, 유전자별 기저율(0.5676)과 '
        '유전자별×라이브러리 10분위(0.5402) 비모수 기준선을 모두 앞선다.', 'p')]

# ---------------------------------------------------------------- 7
s += [P('7. 학습 — 2단계', 'h1')]
s += [tbl([
    ['단계', '목적', '손실'],
    ['<b>1</b>', '잠재 공간을 먼저 만든다 (속도장 비활성)', '재구성 + KL'],
    ['<b>2</b>', 'OT 로 커플링된 잠재 사이의 직선 보간을 따라 속도장을 맞춘다',
     '<b>flow matching + 재구성</b>'],
], [0.08, 0.52, 0.40])]
s += [Spacer(1, 5)]
s += [box('<b>2단계에서 재구성 항을 반드시 유지해야 한다.</b> 인코더를 미세조정하면서 손실이 flow matching '
          '하나뿐이면, <b>잠재를 붕괴시키는 것이 전역 최적</b>이 된다 — z1 - z0 = 0 이 되어 어떤 속도장이든 '
          '완벽한 점수를 받기 때문이다. 실제로 이 상태에서 ||z1 - z0|| 이 0.019 까지 떨어졌고(||z0|| 은 8) '
          '수송이 전혀 일어나지 않았다. 재구성 항을 되살리자 edist_rel 이 처음으로 '
          '오토인코더 바닥(1.011) 아래(0.595)로 내려갔다.', 'p')]
s += [P('그 밖의 학습 설계: 잠재는 2단계 시작 전 <b>표준화</b>한다(원래 스케일에서는 타깃 속도가 1e-2 수준이라 '
        '기울기가 수치 잡음에 묻힌다). 단일 섭동만으로 워밍업한 뒤 조합을 합류시킨다 — 모든 u_a 는 '
        '단일 섭동만으로 식별 가능하므로, 먼저 안정시켜야 상호작용 항이 단일 섭동의 오차를 흡수하지 않는다.', 'p')]

# ---------------------------------------------------------------- 8
s += [P('8. 평가', 'h1')]
s += [P('관례 지표(top-20 DE Pearson)는 모든 기준선에서 0.96~0.98 로 포화되어 모델을 구분하지 못한다. '
        '보고는 하되 판정에는 쓰지 않는다. 판정은 두 지표로 한다.', 'p')]
s += [tbl([
    ['지표', '정의', '읽는 법'],
    ['<b>edist_rel</b>', 'E(예측, 실제) / E(대조군, 실제), E 는 Székely energy distance',
     '<b>분포</b> 간 거리. 0 = 완벽, 1 = 대조군을 그대로 낸 것과 같음'],
    ['<b>resid_R2</b>', 'r = m_AB - m_A - m_B + m_ctrl 을 얼마나 복원하는가',
     '<b>비가법 성분</b>만 본다. 정확히 가법인 예측은 구성상 0 점'],
], [0.16, 0.40, 0.44])]
s += [Spacer(1, 5)]
s += [P('두 번째 지표가 핵심이다. "조합 반응을 맞혔다"는 주장은 단일 섭동의 합으로 설명되지 않는 부분을 '
        '맞혔을 때만 성립하는데, resid_R2 가 정확히 그것만 측정한다. 노이즈 바닥은 조건별 세포 수'
        '(46~1,005, 20배 차)에 맞춰 보정한다.', 'p')]

# ---------------------------------------------------------------- 9
s += [P('9. 설정 축 (ablation)', 'h1')]
s += [code([
    'model.backbone         : mlp | transformer | scvi | pcab',
    'model.decoder_head     : mse | hurdle | zinb',
    'model.hurdle_gate      : soft | hard | sample',
    'model.hurdle_magnitude : point | gaussian',
    '',
    'model.generator        : affine | neural_field',
    'model.interaction      : additive | commutator | free_mlp   <- 핵심 대조군',
    '',
    'model.mask_combine     : gate | logit_bias',
    'model.mask_mode        : hybrid | prior_only | residual_only',
    'model.mask_activation  : tanh | sigmoid',
    'model.mask_alpha       : 1.0',
    '',
    'train.coupling         : uot | ot | random',
])]
s += [box('<b>반드시 보고해야 할 대조군 셋.</b> '
          '<b>interaction=free_mlp</b> — 자유 MLP 가 Lie 괄호와 비슷하면 대수적 형태의 기여가 없다는 뜻이다. '
          '<b>mask_mode=residual_only</b> — 자유 행렬이 KEGG 를 그냥 재학습한 것 아니냐는 질문에 답한다. '
          '<b>mask_activation=sigmoid</b> — 부호 도입의 기여를 분리한다.', 'b')]

# ---------------------------------------------------------------- 10
s += [P('10. 현재 상태', 'h1')]
s += [P('데이터 파이프라인, 평가 하네스, 기준선, 백본 4종, LieCFM 동역학, 구조 검증 테스트까지 구현되어 있다. '
        '기준선 대비 목표선은 additive split 기준 edist_rel &lt; 0.286 / resid_R2 &gt; 0.395 이며, '
        '현재는 그에 미치지 못한다. 짧은 실행에서 edist_rel 0.595 로 오토인코더 바닥을 처음 통과한 단계다.', 'p')]
s += [P('다음 판정은 <b>interaction 3-way 비교</b>다. 여기서 Lie 괄호가 가법 모델과 자유 MLP 를 모두 이기지 '
        '못하면, 논문의 첫 번째 기여는 성립하지 않는다. 실행 중이다.', 'p')]

SimpleDocTemplate('docs/ARCHITECTURE.pdf', pagesize=A4,
                  leftMargin=20 * mm, rightMargin=20 * mm,
                  topMargin=18 * mm, bottomMargin=20 * mm,
                  title='LT-VAE + LieCFM 모델 아키텍처',
                  author='PGFM').build(s, onFirstPage=footer, onLaterPages=footer)
print('OK')
