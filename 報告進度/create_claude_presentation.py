"""
create_claude_presentation.py
為 `firstry/claude改進版.py` 生成向指導教授說明的專題簡報（13 頁，強化論文數學公式剖析）：
  1. 高解析度實證圖表與 600 DPI 公式圖片 -> 報告進度/claude_assets/
  2. PowerPoint 簡報檔（含每頁講稿備忘錄）   -> 報告進度/claude改進版_教授說明簡報.pptx
  3. 互動式網頁簡報預覽                    -> 報告進度/claude改進版_簡報網頁預覽.html
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


# ---------------------------------------------------------------------------
# 路徑設定
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../dl/報告進度
DL_DIR = os.path.dirname(BASE_DIR)                              # .../dl
ASSETS_DIR = os.path.join(BASE_DIR, "claude_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PPTX_OUT = os.path.join(BASE_DIR, "claude改進版_教授說明簡報.pptx")
HTML_OUT = os.path.join(BASE_DIR, "claude改進版_簡報網頁預覽.html")

ORIG_PLOT_CANDIDATES = [
    os.path.join(DL_DIR, "etf_tranad_project", "results", "figures", "etf_anomaly_trend.png"),
    os.path.join(ASSETS_DIR, "orig_trend.png"),
]
CLAUDE_PLOT_SRC = os.path.join(DL_DIR, "firstry", "etf_anomaly_trend_claude.png")
TOTAL_PAGES = 13


# ---------------------------------------------------------------------------
# 1. 生成高解析度公式圖片 & 實證圖表
# ---------------------------------------------------------------------------
def configure_plot_fonts():
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "cm"


def render_formula_png(latex_str, out_path, fontsize=17, color="#0F172A", figsize=(6.2, 0.8)):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=figsize, dpi=600)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, color=color, ha="center", va="center")
    fig.savefig(out_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def build_timeline_chart(out_path):
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(11.5, 2.05), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    segments = [
        (1, 51, "#1D4ED8", "#EFF6FF", "訓練集 Train (2010–2018)", "2,234 視窗 · 健康常態基準\n計算 μ / σ（無前視）與 P98 門檻 = 20.79"),
        (53, 19, "#D97706", "#FFFBEB", "驗證集 Val (2019–2021)", "757 視窗 · 含 2020 疫情\n只用於 Early Stopping，不用來定門檻"),
        (73, 26, "#059669", "#ECFDF5", "盲測集 Test (2022–2026/11)", "1,222 視窗 · 嚴格隔離盲測\n抓出 30 天重大異常 (2.5%) 與 XAI 歸因"),
    ]
    for x, w, edge, bg, title, sub in segments:
        rect = patches.FancyBboxPatch(
            (x, 2.3), w, 5.8,
            boxstyle="round,pad=0.25,rounding_size=0.6",
            linewidth=2, edgecolor=edge, facecolor=bg
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 6.5, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=edge)
        ax.text(x + w / 2, 4.1, sub, ha="center", va="center", fontsize=9.2, color="#334155", linespacing=1.45)

    ax.annotate("", xy=(99, 1.1), xytext=(1, 1.1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#64748B"))
    ax.text(50, 0.35, "嚴格依時間軸單向切分（4,242 個交易日，K=30 滑動視窗，Causal Mask 確保不偷看未來）",
            ha="center", va="center", fontsize=9.5, color="#475569", fontweight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def build_training_dynamics_chart(out_path):
    configure_plot_fonts()
    epochs = np.arange(1, 16)
    l1 = [0.4444, 0.1926, 0.1568, 0.3051, 0.6633, 0.6716, 0.7894, 0.9636, 1.0690, 1.1847, 1.2327, 1.4809, 1.3530, 1.9026, 2.3670]
    l2 = [0.7370, 0.0008, -0.0679, -0.1910, -0.4770, -0.4979, -0.6265, -0.8073, -0.9063, -1.0182, -1.0719, -1.3235, -1.2047, -1.7197, -2.1536]
    val_rec = [0.8703, 0.6217, 0.6004, 0.6684, 0.8125, 0.7079, 0.5877, 0.6682, 0.8759, 0.8838, 0.7387, 0.9682, 1.1473, 1.2672, 1.4223]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.9), dpi=300, sharex=True)

    ax1.plot(epochs, l1, marker="o", ms=4.5, lw=2.0, color="#2563EB", label="L1 (Encoder + Dec1)")
    ax1.plot(epochs, l2, marker="s", ms=4.5, lw=2.0, color="#EF4444", label="L2 (Decoder 2 對抗項)")
    ax1.axhline(0, color="#64748B", ls="--", lw=1.0)
    ax1.axvline(2.5, color="#F59E0B", ls=":", lw=1.4)
    ax1.text(2.7, 1.85, "Epoch 3 起 w < 0.5\nL2 轉負（開始放大誤差）", fontsize=8.5, color="#B45309", fontweight="bold")
    ax1.set_ylabel("Training Loss", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax1.set_title("真實對抗訓練動態（雙優化器分離更新）", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax1.grid(True, ls=":", alpha=0.6)

    ax2.plot(epochs, val_rec, marker="D", ms=4.5, lw=2.2, color="#059669", label="Validation Recon Error (O1)")
    ax2.scatter([7], [0.5877], color="#DC2626", s=85, zorder=5, label="最佳權重 Epoch 7 (0.5877)")
    ax2.axvline(7, color="#DC2626", ls="--", lw=1.2, alpha=0.7)
    ax2.set_ylim(0.45, 1.56)
    ax2.annotate("鎖定最佳模型\nval_rec = 0.5877", xy=(7, 0.5877), xytext=(8.8, 0.52),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
                 fontsize=8.5, fontweight="bold", color="#DC2626")
    ax2.annotate("Epoch 15 早停\n防止過度發散", xy=(15, 1.4223), xytext=(11.3, 1.32),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2),
                 fontsize=8.5, fontweight="bold", color="#334155")
    ax2.set_xlabel("Epoch", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Val Recon MSE", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_xticks(epochs)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax2.grid(True, ls=":", alpha=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def build_sensitivity_chart(out_path):
    configure_plot_fonts()
    labels = ["P95\n(12.99)", "P97\n(16.41)", "P98 ★\n(20.79)", "P99\n(31.48)"]
    days = [84, 49, 30, 13]
    pcts = [6.9, 4.0, 2.5, 1.1]
    colors = ["#94A3B8", "#60A5FA", "#2563EB", "#94A3B8"]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)
    ax.axhline(24.44, color="#DC2626", ls="--", lw=1.4, zorder=1, label="P98 理論預期異常率 2.0% (約 24.4 天)")
    bars = ax.bar(labels, days, width=0.52, color=colors, edgecolor="#1E293B", linewidth=0.8, zorder=2)
    bars[2].set_edgecolor("#1E3A8A")
    bars[2].set_linewidth(2.0)

    for bar, d, p in zip(bars, days, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2,
                f"{d} 天\n({p:.1f}%)", ha="center", va="bottom", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0),
                fontsize=9.5, fontweight="bold", color="#0F172A")

    ax.set_ylim(0, 102)
    ax.set_ylabel("測試集異常天數 (共 1,222 天)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("訓練集健康常態門檻敏感度分析 (P95 ~ P99)", fontsize=10.8, fontweight="bold", color="#0F172A")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(axis="y", ls=":", alpha=0.6)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def build_xai_cases_chart(out_path):
    configure_plot_fonts()
    events = [
        "2026-11-05 (近期事件)\nScore=57.17 → QQQ 板塊破裂",
        "2022-05-09 (能源衝擊)\nScore=42.15 → XLE 板塊破裂",
        "2025-04-09 (關稅暫緩暴漲)\nScore=246.57 → 系統性風險",
        "2025-04-04 (關稅崩跌日)\nScore=131.99 → 系統性風險",
    ]
    tickers = ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLI"]
    data = np.array([
        [28.1, 44.7,  7.1,  2.8,  2.6, 14.7],
        [17.7, 16.4,  8.8, 46.8,  3.5,  6.8],
        [32.6, 32.5,  6.2,  8.9,  5.9, 14.0],
        [20.0, 15.2, 18.2, 20.3, 10.7, 15.6],
    ])
    palette = ["#F59E0B", "#1D4ED8", "#06B6D4", "#DC2626", "#10B981", "#64748B"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)
    left = np.zeros(len(events))
    y = np.arange(len(events))

    for j, (t, col) in enumerate(zip(tickers, palette)):
        vals = data[:, j]
        ax.barh(y, vals, left=left, height=0.56, label=t, color=col, edgecolor="white", linewidth=0.8)
        for i, v in enumerate(vals):
            if v >= 10.0:
                ax.text(left[i] + v / 2, y[i], f"{t}\n{v:.1f}%", ha="center", va="center",
                        fontsize=7.8, fontweight="bold", color="white")
        left += vals

    ax.axvline(35.0, color="#0F172A", ls="--", lw=1.6)
    ax.text(36.0, 3.55, "集中度門檻 35%", fontsize=8.8, fontweight="bold", color="#0F172A", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(events, fontsize=8.8, fontweight="bold", color="#1E293B")
    ax.set_ylim(-0.55, 3.85)
    ax.set_xlim(0, 102)
    ax.set_xlabel("各 ETF 重構誤差貢獻佔比 (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("XAI 板塊歸因實例：單一板塊破裂 vs. 全市場系統性風險", fontsize=10.5, fontweight="bold", color="#0F172A")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=8.5, frameon=False)
    ax.grid(axis="x", ls=":", alpha=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def prepare_assets():
    assets = {}
    orig_src = next((p for p in ORIG_PLOT_CANDIDATES if os.path.exists(p)), None)
    if orig_src:
        dst = os.path.join(ASSETS_DIR, "orig_trend.png")
        shutil.copy2(orig_src, dst)
        assets["orig_trend"] = dst
    if os.path.exists(CLAUDE_PLOT_SRC):
        dst = os.path.join(ASSETS_DIR, "claude_trend.png")
        shutil.copy2(CLAUDE_PLOT_SRC, dst)
        assets["claude_trend"] = dst

    assets["timeline"] = build_timeline_chart(os.path.join(ASSETS_DIR, "fig_timeline.png"))
    assets["train_dyn"] = build_training_dynamics_chart(os.path.join(ASSETS_DIR, "fig_training_dynamics.png"))
    assets["sensitivity"] = build_sensitivity_chart(os.path.join(ASSETS_DIR, "fig_sensitivity.png"))
    assets["xai_cases"] = build_xai_cases_chart(os.path.join(ASSETS_DIR, "fig_xai_cases.png"))

    # 600 DPI 論文公式圖片
    assets["eq_input"] = render_formula_png(
        r"$r_{t,i} = \ln\left(\frac{P_{t,i}}{P_{t-1,i}}\right), \quad w_{t,i} = \frac{r_{t,i} - \mu_{\mathrm{train}, i}}{\sigma_{\mathrm{train}, i}}, \quad W_t \in \mathbb{R}^{30 \times 6}$",
        os.path.join(ASSETS_DIR, "eq_input.png"),
        fontsize=19.0, color="#1E3A8A", figsize=(9.5, 0.85)
    )
    assets["eq_pe"] = render_formula_png(
        r"$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$",
        os.path.join(ASSETS_DIR, "eq_pe.png"),
        fontsize=18.0, color="#1E3A8A", figsize=(8.2, 0.85)
    )
    assets["eq_attn_focus"] = render_formula_png(
        r"$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V, \quad F = (O_1 - W)^2$",
        os.path.join(ASSETS_DIR, "eq_attn_focus.png"),
        fontsize=18.0, color="#065F46", figsize=(8.2, 0.85)
    )
    assets["eq_adv_losses"] = render_formula_png(
        r"$\mathcal{L}_1 = \frac{1}{n}\,\|O_1 - W\|_2^2 \;+\; \left(1 - \frac{1}{n}\right)\|\hat{O}_2 - W\|_2^2$"
        + "\n"
        + r"$\mathcal{L}_2 = \frac{1}{n}\,\|O_2 - W\|_2^2 \;-\; \left(1 - \frac{1}{n}\right)\|\hat{O}_2 - W\|_2^2$",
        os.path.join(ASSETS_DIR, "eq_adv_losses.png"),
        fontsize=19.5, color="#0F172A", figsize=(8.8, 1.45)
    )
    assets["eq_cancel"] = render_formula_png(
        r"$\mathcal{L}_1 + \mathcal{L}_2 = w(\|O_1-W\|^2+\|O_2-W\|^2) + (1-w)\|\hat{O}_2-W\|^2 - (1-w)\|\hat{O}_2-W\|^2$",
        os.path.join(ASSETS_DIR, "eq_cancel.png"),
        fontsize=14, color="#991B1B", figsize=(6.2, 0.55)
    )
    assets["eq_split"] = render_formula_png(
        r"$\mathcal{L}_1 \rightarrow \nabla_{\theta_{\mathrm{enc},\mathrm{dec1}}}, \qquad \mathcal{L}_2 \rightarrow \nabla_{\theta_{\mathrm{dec2}}}$",
        os.path.join(ASSETS_DIR, "eq_split.png"),
        fontsize=16, color="#1E3A8A", figsize=(5.2, 0.55)
    )
    assets["eq_score_xai"] = render_formula_png(
        r"$s_t = \frac{1}{2}\|O_{1,t}-W_t\|_2^2 + \frac{1}{2}\|\hat{O}_{2,t}-W_t\|_2^2 = \sum_{i=1}^{6}\mathrm{Attr}_{t,i}, \quad \mathrm{Contrib}_{t,i}\% = \frac{\mathrm{Attr}_{t,i}}{s_t}\times 100\%$",
        os.path.join(ASSETS_DIR, "eq_score_xai.png"),
        fontsize=18.5, color="#1E3A8A", figsize=(10.2, 0.95)
    )
    assets["eq_score_dayt"] = render_formula_png(
        r"$s_t = \frac{1}{2}\|O_{1,t}-W_t\|_2^2 + \frac{1}{2}\|\hat{O}_{2,t}-W_t\|_2^2 = \sum_{i=1}^{6}\mathrm{Attr}_{t,i}$",
        os.path.join(ASSETS_DIR, "eq_score_dayt.png"),
        fontsize=15.5, color="#1E3A8A", figsize=(6.0, 0.55)
    )
    return assets


# ---------------------------------------------------------------------------
# 2. PowerPoint 建構輔助工具
# ---------------------------------------------------------------------------
C_NAVY_DARK   = RGBColor(15, 23, 42)
C_NAVY_TITLE  = RGBColor(30, 41, 59)
C_BLUE        = RGBColor(37, 99, 235)
C_BLUE_BG     = RGBColor(239, 246, 255)
C_BLUE_BORDER = RGBColor(147, 197, 253)
C_CYAN        = RGBColor(56, 189, 248)
C_EMERALD     = RGBColor(5, 150, 105)
C_EMERALD_BG  = RGBColor(236, 253, 245)
C_CORAL       = RGBColor(220, 38, 38)
C_CORAL_BG    = RGBColor(254, 242, 242)
C_AMBER       = RGBColor(217, 119, 6)
C_AMBER_BG    = RGBColor(255, 251, 235)
C_BG_LIGHT    = RGBColor(248, 250, 252)
C_WHITE       = RGBColor(255, 255, 255)
C_BORDER      = RGBColor(226, 232, 240)
C_TEXT        = RGBColor(51, 65, 85)
C_MUTED       = RGBColor(100, 116, 139)

FONT_ZH = "Microsoft JhengHei"
FONT_MATH = "Cambria Math"


def set_bg(slide, prs, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg


def add_runs(p, segments, default_size=11, default_color=C_TEXT):
    for seg in segments:
        if isinstance(seg, str):
            text, st = seg, {}
        else:
            text, st = seg
        r = p.add_run()
        r.text = text
        is_math = st.get("math", False)
        r.font.name = st.get("font", FONT_MATH if is_math else FONT_ZH)
        r.font.size = Pt(st.get("size", default_size))
        r.font.bold = st.get("bold", False)
        r.font.italic = st.get("italic", is_math)
        r.font.color.rgb = st.get("color", default_color)


def add_header(slide, tag, title, page_num, total_pages=TOTAL_PAGES):
    tb_tag = slide.shapes.add_textbox(Inches(0.75), Inches(0.32), Inches(9.5), Inches(0.28))
    tf_tag = tb_tag.text_frame
    tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_bottom = 0
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = tag
    p_tag.font.name = FONT_ZH
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_BLUE

    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.65), Inches(0.32), Inches(0.93), Inches(0.30))
    pill.fill.solid()
    pill.fill.fore_color.rgb = C_BLUE_BG
    pill.line.color.rgb = C_BLUE_BORDER
    pill.line.width = Pt(1)
    tf_p = pill.text_frame
    tf_p.margin_top = tf_p.margin_bottom = 0
    p_p = tf_p.paragraphs[0]
    p_p.text = f"{page_num:02d} / {total_pages}"
    p_p.font.name = FONT_ZH
    p_p.font.size = Pt(9.5)
    p_p.font.bold = True
    p_p.font.color.rgb = C_BLUE
    p_p.alignment = PP_ALIGN.CENTER

    tb_title = slide.shapes.add_textbox(Inches(0.75), Inches(0.60), Inches(11.83), Inches(0.55))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title
    p_t.font.name = FONT_ZH
    p_t.font.size = Pt(20.5)
    p_t.font.bold = True
    p_t.font.color.rgb = C_NAVY_TITLE

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.20), Inches(11.83), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = C_BORDER
    line.line.fill.background()

    tb_f = slide.shapes.add_textbox(Inches(0.75), Inches(7.14), Inches(11.83), Inches(0.24))
    tf_f = tb_f.text_frame
    tf_f.margin_left = tf_f.margin_top = tf_f.margin_bottom = 0
    p_f = tf_f.paragraphs[0]
    p_f.text = "firstry/claude改進版.py 專題說明簡報  ·  TranAD (VLDB 2022) 數學公式剖析與金融板塊異常偵測實證"
    p_f.font.name = FONT_ZH
    p_f.font.size = Pt(8.5)
    p_f.font.color.rgb = C_MUTED


def make_card(slide, left, top, width, height, title="", title_color=C_NAVY_TITLE, bg=C_WHITE, border=C_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg
    if border:
        card.line.color.rgb = border
        card.line.width = Pt(1.2)
    else:
        card.line.fill.background()

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.22), top + Inches(0.14), width - Inches(0.44), Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_ZH
        p.font.size = Pt(12.8)
        p.font.bold = True
        p.font.color.rgb = title_color
    return card


def add_bullets(slide, left, top, width, height, items, font_size=10.5, space_after=6):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for idx, segs in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        add_runs(p, segs, default_size=font_size)
        p.space_after = Pt(space_after)
    return tb


def add_pic_fit(slide, img_path, left, top, max_w, max_h):
    from PIL import Image
    with Image.open(img_path) as im:
        iw, ih = im.size
    ratio = iw / ih
    box_ratio = max_w / max_h
    if ratio >= box_ratio:
        w = max_w
        h = w / ratio
    else:
        h = max_h
        w = h * ratio
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    return slide.shapes.add_picture(img_path, x, y, width=w, height=h)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------------------
# 3. 建構 13 頁 PowerPoint
# ---------------------------------------------------------------------------
def build_pptx(assets):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: 封面頁
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, prs, C_NAVY_DARK)

    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.05), Inches(5.35), Inches(0.38))
    badge.fill.solid()
    badge.fill.fore_color.rgb = C_BLUE
    badge.line.fill.background()
    p_b = badge.text_frame.paragraphs[0]
    p_b.text = "US ETF ANOMALY RADAR  ·  TRANAD CAPSTONE PROJECT"
    p_b.font.name = FONT_ZH
    p_b.font.size = Pt(9.5)
    p_b.font.bold = True
    p_b.font.color.rgb = C_WHITE
    p_b.alignment = PP_ALIGN.CENTER

    tb = s1.shapes.add_textbox(Inches(0.85), Inches(1.72), Inches(11.6), Inches(1.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "跨板塊美股 ETF 風險異常偵測與\n成因診斷系統（基於 TranAD）"
    p1.font.name = FONT_ZH
    p1.font.size = Pt(34)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE

    p2 = tf.add_paragraph()
    p2.text = "「美股市場異常雷達」— 抓出板塊脫鉤變天訊號、找出出事產業，並剖析背後數學與程式改進"
    p2.font.name = FONT_ZH
    p2.font.size = Pt(15.5)
    p2.font.color.rgb = C_CYAN
    p2.space_before = Pt(14)

    highlights = [
        ("解決什麼痛點？", "抓出「大盤看似沒事、底下板塊已脫鉤」的結構風險，避免只看單一漲跌幅漏報", RGBColor(30, 58, 138)),
        ("能幫使用者幹嘛？", "亮燈時立刻判斷該「全市場降槓桿避險」還是「只針對單一板塊換倉」", RGBColor(6, 95, 70)),
        ("理論剖析與工程除錯", "拆解 4 組論文公式並修復抵消 / 拖尾 Bug；盲測異常率 40.0% ➔ 2.5%", RGBColor(127, 29, 29)),
    ]
    cw, ch, gap = Inches(3.68), Inches(1.42), Inches(0.39)
    for i, (ht, hd, hbg) in enumerate(highlights):
        lx = Inches(0.85) + i * (cw + gap)
        make_card(s1, lx, Inches(4.32), cw, ch, ht, title_color=C_CYAN, bg=RGBColor(30, 41, 59), border=hbg)
        tb_c = s1.shapes.add_textbox(lx + Inches(0.22), Inches(4.82), cw - Inches(0.44), Inches(0.82))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        tf_c.margin_top = 0
        pc = tf_c.paragraphs[0]
        pc.text = hd
        pc.font.name = FONT_ZH
        pc.font.size = Pt(10.5)
        pc.font.color.rgb = RGBColor(226, 232, 240)

    tb_foot = s1.shapes.add_textbox(Inches(0.85), Inches(6.35), Inches(11.6), Inches(0.5))
    pf = tb_foot.text_frame.paragraphs[0]
    pf.text = "主講檔案：firstry/claude改進版.py   |   對照文獻：Tuli et al., TranAD (VLDB 2022)   |   資料期間：2010-01-05 ~ 2026-11-13"
    pf.font.name = FONT_ZH
    pf.font.size = Pt(10.5)
    pf.font.color.rgb = RGBColor(148, 163, 184)

    set_notes(s1, (
        "【第 1 頁講稿】\n"
        "教授好，我們專題簡單來說就是做一套『美股 ETF 市場異常雷達』。\n"
        "它先用平靜時期學習 6 大產業 ETF 平常怎麼一起連動；在盲測期間只要市場突然脫離平常默契就亮燈警告，並直接指出是全市場系統性風險，還是哪個特定板塊出事。\n"
        "今天這份簡報也會同步呈現我們對 claude改進版.py 的論文公式剖析，以及修掉原版兩大 Bug 後的實證結果。"
    ))

    # =========================================================================
    # SLIDE 2: 核心定位（三句話說清楚專題目的 + 運作流程）
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    set_bg(s2, prs, C_BG_LIGHT)
    add_header(s2, "01  PLAIN-ENGLISH PURPOSE & WORKFLOW", "我們在做什麼？用「平靜期默契」抓出市場變天與出事產業", 2)

    cols2 = [
        ("1. 痛點：只看大盤漲跌會漏掉風險", C_CORAL, C_CORAL_BG, C_CORAL, [
            [("• 傳統看盤的盲點：", {"bold": True, "color": C_NAVY_TITLE}), ("只盯 SPY 今天跌幾%，但很多股災剛冒頭時大盤看似風平浪靜，底下板塊已經開始「各走各的」。", {})],
            [("• 為什麼選 TranAD？", {"bold": True, "color": C_CORAL}), ("一般 Autoencoder 太會「順手把小異常修成正常」；TranAD 用兩階段機制專門放大剛脫鉤的細微裂縫。", {})],
        ]),
        ("2. 做法：教 AI 記住平靜期的連動默契", C_BLUE, C_BLUE_BG, C_BLUE, [
            [("• 學習健康範本：", {"bold": True, "color": C_BLUE}), ("拿 2010–2018 年相對平穩的市場，教模型記住 SPY, QQQ, XLF, XLE, XLV, XLI 平時誰跟誰一起動。", {})],
            [("• 怎麼抓異常？", {"bold": True, "color": C_NAVY_TITLE}), ("放到 2022–2026 盲測時，模型照過去默契推算今天該怎麼走；「猜錯越多 ＝ 今天市場越反常」。", {})],
        ]),
        ("3. 價值：不只亮紅燈，還給出避險處方", C_EMERALD, C_EMERALD_BG, C_EMERALD, [
            [("• 過濾警報疲勞：", {"bold": True, "color": C_EMERALD}), ("修復 Bug 後，只對比平靜期 98% 日子都還反常的日子亮燈（2.5% 關鍵警報，不再天天亂叫）。", {})],
            [("• 看佔比決定動作：", {"bold": True, "color": C_NAVY_TITLE}), ("誤差分散 (<35%) ➔ 全市場齊崩，該整體降槓桿；誤差集中 (≥35%) ➔ 抓出單一戰犯板塊調倉即可。", {})],
        ]),
    ]
    cw2, gap2 = Inches(3.75), Inches(0.29)
    for i, (t2, tc2, bg2, bd2, items2) in enumerate(cols2):
        lx = Inches(0.75) + i * (cw2 + gap2)
        make_card(s2, lx, Inches(1.42), cw2, Inches(3.15), t2, title_color=tc2, bg=bg2, border=bd2)
        add_bullets(s2, lx + Inches(0.18), Inches(1.98), cw2 - Inches(0.36), Inches(2.45), items2, font_size=10.2, space_after=8)

    make_card(s2, Inches(0.75), Inches(4.76), Inches(11.83), Inches(2.22), "claude改進版.py 大白話運作 4 步驟（昨晚收盤跑一次 ➔ 決定今天怎麼調整部位）", C_NAVY_TITLE)
    steps = [
        ("Step 1  看這 6 檔怎麼動", "計算 6 大 ETF 每日漲跌幅\n只用 2010–18 標準縮放（無前視）", C_BLUE, C_BLUE_BG),
        ("Step 2  學平靜期連動默契", "看過去 30 天（因果遮罩防偷看）\n雙優化器對抗放大可疑偏差", C_EMERALD, C_EMERALD_BG),
        ("Step 3  看今天猜錯多少", "只算最新一天 Day t 誤差總和\n超過平靜期 98% 水準就亮燈", C_AMBER, C_AMBER_BG),
        ("Step 4  抓戰犯決定動作", "拆解 6 檔 ETF 誤差佔比 %\n依 35% 門檻分系統風險或換板塊", C_CORAL, C_CORAL_BG),
    ]
    sw, sh, sgap = Inches(2.68), Inches(1.38), Inches(0.22)
    for idx, (st_t, st_d, st_c, st_bg) in enumerate(steps):
        sx = Inches(0.97) + idx * (sw + sgap)
        sy = Inches(5.36)
        make_card(s2, sx, sy, sw, sh, st_t, title_color=st_c, bg=st_bg, border=st_c)
        tb_s = s2.shapes.add_textbox(sx + Inches(0.18), sy + Inches(0.56), sw - Inches(0.36), Inches(0.72))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True
        tf_s.margin_top = 0
        ps = tf_s.paragraphs[0]
        ps.text = st_d
        ps.font.name = FONT_ZH
        ps.font.size = Pt(10)
        ps.font.color.rgb = C_TEXT
        if idx < 3:
            arr = s2.shapes.add_textbox(sx + sw + Inches(0.01), sy + Inches(0.45), Inches(0.20), Inches(0.4))
            pa = arr.text_frame.paragraphs[0]
            pa.text = "➔"
            pa.font.size = Pt(13)
            pa.font.bold = True
            pa.font.color.rgb = C_MUTED
            pa.alignment = PP_ALIGN.CENTER

    set_notes(s2, (
        "【第 2 頁講稿 — 30 秒白話破題】\n"
        "用三句話概括這個專題在做什麼：\n"
        "第一，只看大盤漲跌會漏掉風險，因為很多危機發生前，科技、能源、金融板塊就已經先出現不合常理的脫鉤；\n"
        "第二，我們拿 2010 到 2018 年平靜的市場教 TranAD 記住 6 檔 ETF 平常怎麼連動，再去檢查 2022 到 2026 年每一天，模型猜錯越多代表今天越反常；\n"
        "第三，亮燈後它不是只喊狼來了，而是看誤差佔比告訴投資人：是全市場一起崩該整體降槓桿，還是只有單一板塊出事、換倉避開就好。\n"
        "實務上在美股收盤後跑一次，就能看昨晚的狀態決定今天怎麼調倉。"
    ))

    # =========================================================================
    # SLIDE 3: 金融化資料前處理與無前視改造（含輸入公式）
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    set_bg(s3, prs, C_BG_LIGHT)
    add_header(s3, "02  INPUT FORMULATION & FINANCIAL ADAPTATIONS", "金融化輸入建構：對數報酬率、Z-score 與無前視時間切分", 3)

    make_card(s3, Inches(0.75), Inches(1.36), Inches(11.83), Inches(2.08), bg=C_WHITE, border=C_BORDER)
    add_pic_fit(s3, assets["timeline"], Inches(0.85), Inches(1.42), Inches(11.63), Inches(1.95))

    make_card(s3, Inches(0.75), Inches(3.56), Inches(11.83), Inches(0.84), bg=C_BLUE_BG, border=C_BLUE_BORDER)
    add_pic_fit(s3, assets["eq_input"], Inches(0.95), Inches(3.62), Inches(8.10), Inches(0.72))
    add_bullets(s3, Inches(9.15), Inches(3.66), Inches(3.30), Inches(0.68), [
        [("公式意涵：", {"bold": True, "color": C_BLUE}), ("將非平穩價格轉成平穩報酬，再以訓練期標準差縮放成 30×6 輸入矩陣。", {})]
    ], font_size=9.5, space_after=0)

    cards3 = [
        ("為什麼公式長這樣 & 改 Adj Close", C_BLUE, [
            [("• 為什麼取對數 ln(P(t)/P(t−1))？", {"bold": True, "color": C_NAVY_TITLE}), ("股價會隨通膨長期上漲（SPY 從 100 漲到 600），直接輸入價格會讓模型把高價當異常；對數報酬率具平穩性且可時間累加。", {})],
            [("• [修正3] 改用 Adj Close：", {"bold": True, "color": C_BLUE}), ("修正原版用 Close 在除息日跳空被當負報酬的假警報。", {})],
        ]),
        ("為何用 Z-score 並拿掉 Sigmoid", C_EMERALD, [
            [("• 原論文 Min-Max + Sigmoid：", {"bold": True, "color": C_NAVY_TITLE}), ("工業感測器有上下限故壓縮到 [0,1]；但金融報酬率屬厚尾分布，若用 Min-Max 會被單一極端值壓扁正常波動。", {})],
            [("• 我們的改造：", {"bold": True, "color": C_EMERALD}), ("採 Z-score + 純 Linear 輸出，黑天鵝跌破 −5σ 也能完整重構。", {})],
        ]),
        ("怎麼用：防前視與 P98 健康錨定", C_AMBER, [
            [("• 嚴格隔離 μ_train, σ_train：", {"bold": True, "color": C_NAVY_TITLE}), ("驗證與測試期全部套用 2010–2018 統計量，絕不偷看未來。", {})],
            [("• 門檻為何不用 Val？", {"bold": True, "color": C_AMBER}), ("2020 疫情會把門檻拉爆造成 2022+ 漏報，故門檻錨定訓練集 P98 (20.793)，Val 專職早停。", {})],
        ]),
    ]
    cw3, gap3 = Inches(3.75), Inches(0.29)
    for i, (t3, c3, items3) in enumerate(cards3):
        lx = Inches(0.75) + i * (cw3 + gap3)
        make_card(s3, lx, Inches(4.54), cw3, Inches(2.46), t3, title_color=c3, bg=C_WHITE, border=c3)
        add_bullets(s3, lx + Inches(0.18), Inches(5.06), cw3 - Inches(0.36), Inches(1.86), items3, font_size=9.7, space_after=5)

    set_notes(s3, (
        "【第 3 頁講稿】\n"
        "進到模型前，先看輸入矩陣 W_t 的數學構造。\n"
        "為什麼不直接餵股價而取對數 ln(P_t/P_{t-1})？因為 2010 年 SPY 才 100 多塊，2026 年超過 600 塊，價格非平穩，模型沒看過 600 塊會整年誤報；同時我們改用 Adj Close 避開除息日跳空。\n"
        "第二，論文把感測器 Min-Max 到 0~1 並接 Sigmoid，但金融報酬率有厚尾特性，所以我們改成 Z-score 搭配無激活 Linear 輸出，且平均和標準差只用 2010–2018 計算，避免前視偏差。"
    ))

    # =========================================================================
    # SLIDE 4 [NEW]: 論文公式剖析 (一) — 位置編碼、因果遮罩與 Focus Score
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    set_bg(s4, prs, C_BG_LIGHT)
    add_header(s4, "03  PAPER FORMULA ANATOMY #1", "論文公式剖析 (一)：位置編碼、因果遮罩與自條件聚焦機制", 4)

    make_card(s4, Inches(0.75), Inches(1.36), Inches(5.80), Inches(1.36), "公式 ①  正弦位置編碼 (Positional Encoding)", C_BLUE, C_BLUE_BG, C_BLUE_BORDER)
    add_pic_fit(s4, assets["eq_pe"], Inches(0.88), Inches(1.82), Inches(5.54), Inches(0.78))

    make_card(s4, Inches(6.78), Inches(1.36), Inches(5.80), Inches(1.36), "公式 ②  因果遮罩注意力與 Focus Score", C_EMERALD, C_EMERALD_BG, RGBColor(110, 231, 183))
    add_pic_fit(s4, assets["eq_attn_focus"], Inches(6.91), Inches(1.82), Inches(5.54), Inches(0.78))

    cw4, gap4 = Inches(3.75), Inches(0.29)
    cols4 = [
        ("1. 為什麼公式長這樣？", C_NAVY_TITLE, C_WHITE, C_BORDER, [
            [("• PE 用 sin/cos 多頻率波：", {"bold": True, "color": C_BLUE}), ("Self-Attention 本質是集合運算（順序打亂輸出不變）；不同波長的正餘弦波讓『相對天數差 Δk』能用線性旋轉矩陣表達，且無需額外可訓練參數。", {})],
            [("• 為什麼 + M (上三角 −∞)？", {"bold": True, "color": C_EMERALD}), ("經 softmax 後 e^(−∞)=0，強制第 t 天的注意力權重完全屏蔽 t+1…30 天。", {})],
            [("• 為什麼 F = (O1 − W)²？", {"bold": True, "color": C_NAVY_TITLE}), ("平方讓正負誤差都變正且放大顯著偏離，成為一張告訴 Phase 2『哪天哪檔最可疑』的焦點地圖。", {})],
        ]),
        ("2. 我們為何選用這組機制？", C_BLUE, C_WHITE, C_BLUE_BORDER, [
            [("• 金融時序具有方向性：", {"bold": True, "color": C_BLUE}), ("『先連跌三天後反彈』與『先反彈後連跌三天』風險截然不同，必須靠 PE 鎖定這 30 天內的先後順序。", {})],
            [("• 嚴防偷看未來（前視）：", {"bold": True, "color": C_EMERALD}), ("Window Encoder 端透過因果遮罩確保在重構 t 時，決策流不會摻入未來報酬資訊。", {})],
            [("• 克服一般 AE 漏報：", {"bold": True, "color": C_NAVY_TITLE}), ("靠 F 把微小破裂訊號顯影出來，再透過 Cross-Attention (Q=視窗, K/V=含 F 編碼) 放大檢視。", {})],
        ]),
        ("3. 在 claude改進版.py 怎麼用？", C_EMERALD, C_EMERALD_BG, C_EMERALD, [
            [("• PositionalEncoding (L82)：", {"bold": True, "color": C_NAVY_TITLE}), ("在 proj1([W⊕F]) 與 proj2(W) 之後注入 d_model=64 的 30 天位置向量。", {})],
            [("• 因果遮罩矩陣 M (L132)：", {"bold": True, "color": C_NAVY_TITLE}), ("mask = torch.triu(full((K,K), -inf), 1) 傳入 self_attn2。", {})],
            [("• 兩階段前向推論 (L154)：", {"bold": True, "color": C_EMERALD}), ("① o1, o2 = model(w)（此時 F=0）\n② focus = (o1 - w)**2（[修正5] 對齊論文）\n③ _, o2_hat = model(w, focus)", {})],
        ]),
    ]
    for i, (t4, tc4, bg4, bd4, items4) in enumerate(cols4):
        lx = Inches(0.75) + i * (cw4 + gap4)
        make_card(s4, lx, Inches(2.90), cw4, Inches(4.08), t4, title_color=tc4, bg=bg4, border=bd4)
        add_bullets(s4, lx + Inches(0.18), Inches(3.48), cw4 - Inches(0.36), Inches(3.38), items4, font_size=10.0, space_after=8)

    set_notes(s4, (
        "【第 4 頁講稿】\n"
        "接下來我們進入論文核心公式的第一部分：位置編碼、因果遮罩與 Focus Score。\n"
        "第一，為什麼 PE 要用不同頻率的正餘弦波？因為 Transformer 本身沒有時間順序概念，不同頻率讓模型能辨識『幾天前』的相對距離，而且不需要多學參數、不容易在金融小樣本上過擬合。\n"
        "第二，為什麼 Self-Attention 要加一個上三角負無窮的矩陣 M？因為經過 softmax，負無窮會變成 0，強迫 Window Encoder 在第 t 天看不到未來。\n"
        "第三，程式怎麼用？在 two_phase_forward 裡，第一次呼叫 model(w) 時 Focus 補零得到粗重構 O1，接著把 (O1-W)² 算成誤差地圖當作第二次呼叫的條件，透過 Cross-Attention 讓解碼端專注在誤差最大的那幾天、那幾檔 ETF。"
    ))

    # =========================================================================
    # SLIDE 5 [NEW]: 論文公式剖析 (二) — 雙階段演化對抗損失 L1, L2
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    set_bg(s5, prs, C_BG_LIGHT)
    add_header(s5, "04  PAPER FORMULA ANATOMY #2", "論文公式剖析 (二)：演化式雙解碼器對抗損失 (L1, L2)", 5)

    make_card(s5, Inches(0.75), Inches(1.36), Inches(11.83), Inches(1.66), "公式 ③  演化對抗損失函數（Adversarial Loss，其中 n = epoch）", C_NAVY_TITLE, C_BLUE_BG, C_BLUE_BORDER)
    add_pic_fit(s5, assets["eq_adv_losses"], Inches(0.95), Inches(1.82), Inches(6.25), Inches(1.10))

    add_bullets(s5, Inches(7.35), Inches(1.80), Inches(5.05), Inches(1.15), [
        [("• 左半項 ", {"bold": True, "color": C_BLUE}), ("1/n · ‖O - W‖²：", {"math": True, "bold": True}), ("基本功項，讓兩解碼器在沒提示時都會正常重構。", {})],
        [("• 右半項 ", {"bold": True, "color": C_CORAL}), ("±(1 - 1/n) · ‖Ô2 - W‖²：", {"math": True, "bold": True}), ("核心對抗項！L1 用「+」想把 Ô2 壓小；L2 用「−」想把 Ô2 放大。", {})],
        [("• 動態權重 ", {"bold": True, "color": C_EMERALD}), ("w = 1/n：", {"math": True, "bold": True}), ("隨 Epoch 從 1.0 → 0.5 → 0.1 遞減，先學穩重構再逐輪開啟對抗。", {})],
    ], font_size=9.8, space_after=5)

    cols5 = [
        ("1. 為什麼公式長這樣？(Min-Max 設計)", C_NAVY_TITLE, C_WHITE, C_BORDER, [
            [("• 借鑑 GAN 的雙人博弈：", {"bold": True, "color": C_BLUE}), ("Decoder 1 (最小化 L1) 像生成器，想做到『就算拿到 Focus 提示，Ô2 也重構得很完美』；Decoder 2 (最小化 L2 = 最大化 ‖Ô2−W‖²) 像判別器，看到 Focus 就刻意放大誤差。", {})],
            [("• 為什麼要有 1/n 衰減？", {"bold": True, "color": C_AMBER}), ("第 1 輪 O1 還是一團亂，這時算出的 Focus 全是雜訊。因此 n=1 時 1−1/n = 0（純學重構），n≥2 才逐步交棒給對抗項。", {})],
        ]),
        ("2. 我們為何在金融上使用它？", C_CORAL, C_WHITE, C_CORAL, [
            [("• 解決「異常被平滑化」：", {"bold": True, "color": C_CORAL}), ("對正常市場資料，O1 準確、Focus 接近 0，Decoder 2 無從放大；對脫鉤初期的金融異常，Focus 亮燈後 Decoder 2 會把誤差成倍放大，拉開正常與異常分數的差距。", {})],
            [("• 解釋原版為何失效：", {"bold": True, "color": C_NAVY_TITLE}), ("看懂這組公式正負號就會發現：絕不能把 L1 + L2 加起來對全部參數求導，否則對抗項直接抵消為 0！", {})],
        ]),
        ("3. 在 claude改進版.py 怎麼用？", C_BLUE, C_BLUE_BG, C_BLUE, [
            [("• 參數分組 (L164–169)：", {"bold": True, "color": C_BLUE}), ("main_params (Encoder+Dec1) 配 opt_main (lr=1e-3)；dec2_params 配 opt_dec2 (lr=5e-4)。", {})],
            [("• 獨立求導 (L195–207)：", {"bold": True, "color": C_NAVY_TITLE}), ("autograd.grad 分別對 L1 與 L2 求梯度，加上 grad_clip=1.0 防止對抗項負向發散。", {})],
            [("• 驗證指標巧思 (L220)：", {"bold": True, "color": C_EMERALD}), ("早停只監控 Phase 1 的 O1 重構誤差，因為 Ô2 被刻意要求放大誤差，不能當收斂指標。", {})],
        ]),
    ]
    for i, (t5, tc5, bg5, bd5, items5) in enumerate(cols5):
        lx = Inches(0.75) + i * (cw4 + gap4)
        make_card(s5, lx, Inches(3.20), cw4, Inches(3.80), t5, title_color=tc5, bg=bg5, border=bd5)
        add_bullets(s5, lx + Inches(0.18), Inches(3.78), cw4 - Inches(0.36), Inches(3.10), items5, font_size=9.9, space_after=7)

    set_notes(s5, (
        "【第 5 頁講稿】\n"
        "這一頁是整篇論文最精華的兩條 Loss 公式。\n"
        "為什麼 L1 用加號、L2 用減號，前面還乘上 1/n？\n"
        "因為第 1 輪剛開始模型還不會重構，1/n = 1 讓兩邊先專心把正常市場學好；隨著 epoch 增加，1 - 1/n 變大，兩位解碼器開始打對台：\n"
        "L1 希望即使給了誤差提示，第二階段依然重構得完美；L2 因為有個負號，最小化它就等於『最大化』第二階段誤差。\n"
        "結果就是：正常日子 Focus 很平靜，誤差小；一旦市場脫鉤，Focus 亮起來，Decoder 2 就會把偏差明顯放大，這就是為什麼在程式裡我們必須用兩個優化器分開更新。"
    ))

    # =========================================================================
    # SLIDE 6: 實作驗證 (一) — 修復對抗抵消 Bug & 實測動態
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    set_bg(s6, prs, C_BG_LIGHT)
    add_header(s6, "05  CRITICAL BUG FIX #1", "實作驗證 (一)：修復原版 Loss 相加抵消，實測對抗真正生效", 6)

    make_card(s6, Inches(0.75), Inches(1.42), Inches(5.85), Inches(2.72), "原版 Bug：L1 + L2 合併求導，對抗項精確抵消為 0", C_CORAL, C_CORAL_BG, C_CORAL)
    add_pic_fit(s6, assets["eq_cancel"], Inches(0.92), Inches(1.94), Inches(5.50), Inches(0.55))
    add_bullets(s6, Inches(0.95), Inches(2.58), Inches(5.45), Inches(1.48), [
        [("• 程式問題 (L156)：", {"bold": True, "color": C_CORAL}), ("(loss1 + loss2).backward() 導致 ±(1−w)‖Ô2−W‖² 互相抵消。", {})],
        [("• 實質後果：", {"bold": True, "color": C_NAVY_TITLE}), ("Phase 2 輸出 Ô2 梯度恆為 0，從未被訓練；proj1 接收 Focus 的權重一直停在隨機亂數！", {})],
        [("• 過擬合惡化：", {"bold": True, "color": C_NAVY_TITLE}), ("盲跑 40 epochs 純重構導致訓練集過擬合 (Train P98 = 1.71)，測試集高達 40.0% 都在亮紅燈。", {})],
    ], font_size=10.0, space_after=5)

    make_card(s6, Inches(0.75), Inches(4.32), Inches(5.85), Inches(2.65), "claude改進版.py 實作效果 [修正1][4][5]", C_BLUE, C_BLUE_BG, C_BLUE)
    add_pic_fit(s6, assets["eq_split"], Inches(0.95), Inches(4.84), Inches(5.45), Inches(0.50))
    add_bullets(s6, Inches(0.95), Inches(5.40), Inches(5.45), Inches(1.48), [
        [("• 參數分組更新：", {"bold": True, "color": C_BLUE}), ("Encoder+Dec1 只吃 L1 梯度 (lr=1e-3)；Dec2 只吃 L2 梯度 (lr=5e-4) + Gradient Clip。", {})],
        [("• Focus 對齊論文：", {"bold": True, "color": C_NAVY_TITLE}), ("由絕對值改為平方誤差 (O1 − W)²。", {})],
        [("• 驗證集早停：", {"bold": True, "color": C_EMERALD}), ("鎖定最佳 Epoch 7 (val_rec = 0.5877)，於 Epoch 15 自動停止。", {})],
    ], font_size=10.0, space_after=5)

    make_card(s6, Inches(6.82), Inches(1.42), Inches(5.76), Inches(5.55), "實測訓練動態 (Epoch 1–15 真實數據)", C_NAVY_TITLE)
    add_pic_fit(s6, assets["train_dyn"], Inches(6.95), Inches(1.94), Inches(5.50), Inches(4.35))
    add_bullets(s6, Inches(7.02), Inches(6.36), Inches(5.38), Inches(0.55), [
        [("圖表解讀：", {"bold": True, "color": C_BLUE}), ("Epoch 3 起 L2 如公式預測轉負（w < 0.5），證實對抗啟動；Epoch 7 取得最低驗證誤差並自動保留。", {})]
    ], font_size=9.5, space_after=0)

    set_notes(s6, (
        "【第 6 頁講稿】\n"
        "套用上一頁的數學公式檢查原版程式，馬上抓出一個致命 Bug：原版把 L1 和 L2 加起來一起 backward。\n"
        "正負對消的結果，就是整個 Phase 2 根本沒訓練到、proj1 的 Focus 權重停在亂數，還因為盲跑 40 輪過度擬合訓練集，導致測試集 40% 天數都在亮紅燈。\n"
        "改成雙優化器分離更新後，右邊實測圖可以看到 Epoch 3 之後 L2 如公式預測轉負，驗證誤差在 Epoch 7 降到最低點 0.5877 並由 Early Stopping 鎖定保存。"
    ))

    # =========================================================================
    # SLIDE 7 [NEW]: 論文公式剖析 (三) — 異常分數與 XAI 歸因分解
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    set_bg(s7, prs, C_BG_LIGHT)
    add_header(s7, "06  PAPER FORMULA ANATOMY #3", "論文公式剖析 (三)：綜合異常分數 s(t) 與 XAI 可加性歸因分解", 7)

    make_card(s7, Inches(0.75), Inches(1.36), Inches(11.83), Inches(1.38), "公式 ④  單日異常分數與逐維度誤差歸因（t = 視窗最後一天，i = 第 i 檔 ETF）", C_NAVY_TITLE, C_BLUE_BG, C_BLUE_BORDER)
    add_pic_fit(s7, assets["eq_score_xai"], Inches(0.95), Inches(1.80), Inches(11.43), Inches(0.84))

    cols7 = [
        ("1. 為什麼公式長這樣？(算子設計)", C_NAVY_TITLE, C_WHITE, C_BORDER, [
            [("• 為什麼 ½ O1 + ½ Ô2？", {"bold": True, "color": C_BLUE}), ("同時融合 Phase 1 的『基礎重構偏離』與 Phase 2 被對抗放大的『焦點偏離』，兼顧穩定性與靈敏度。", {})],
            [("• 為什麼用平方和 (L2²) 而非 Mean？", {"bold": True, "color": C_EMERALD}), ("平方和滿足嚴格『可加性』——總異常分數剛好等於 6 檔 ETF 個別誤差 Attr(t,i) 的總和，不會被平均數稀釋跨資產總衝擊！", {})],
            [("• 為何只取下標 t（最後一天）？", {"bold": True, "color": C_CORAL}), ("論文對每個時間點 t 只評估當下那一個向量，過去 K−1 天僅負責提供上下文脈絡。", {})],
        ]),
        ("2. 我們為何這樣改造？(金融需求)", C_EMERALD, C_WHITE, C_EMERALD, [
            [("• [修正2] 消除 30 天矩形拖尾：", {"bold": True, "color": C_CORAL}), ("原版誤將整個 30×6 視窗加總，導致一次暴跌往後連續 30 天全亮紅燈，且當日衝擊被稀釋 30 倍。", {})],
            [("• [修正6] 讓分數與 XAI 同口徑：", {"bold": True, "color": C_EMERALD}), ("原版總分用兩解碼器 30 天、歸因卻只看 Ô2 的 1 天；統一後 s(t) = Σ Attr(t,i)，數學上完全自洽。", {})],
            [("• [修正8] 解開黑盒子診斷：", {"bold": True, "color": C_NAVY_TITLE}), ("直接算出每檔 ETF 貢獻度 %，讓投資人知道該砍單一板塊還是全面避險。", {})],
        ]),
        ("3. 在 claude改進版.py 怎麼用？", C_BLUE, C_BLUE_BG, C_BLUE, [
            [("• 單日切片計算 (L252–254)：", {"bold": True, "color": C_BLUE}), ("feat_err = 0.5*(o1[:,-1,:]-w[:,-1,:])**2 + 0.5*(o2_hat[:,-1,:]-w[:,-1,:])**2\nscores = feat_err.sum(dim=1)", {})],
            [("• 門檻與觸發 (L309–310)：", {"bold": True, "color": C_NAVY_TITLE}), ("取訓練集 s_t 的第 98 百分位 (20.793)；測試日超標即觸發警報。", {})],
            [("• 集中度診斷 (L258–270)：", {"bold": True, "color": C_CORAL}), ("佔比 < 35% ➔ 系統性風險；≥ 35% ➔ 依首位資產判大盤脫鉤或單一板塊破裂。", {})],
        ]),
    ]
    for i, (t7, tc7, bg7, bd7, items7) in enumerate(cols7):
        lx = Inches(0.75) + i * (cw4 + gap4)
        make_card(s7, lx, Inches(2.92), cw4, Inches(4.06), t7, title_color=tc7, bg=bg7, border=bd7)
        add_bullets(s7, lx + Inches(0.18), Inches(3.50), cw4 - Inches(0.36), Inches(3.36), items7, font_size=9.9, space_after=7)

    set_notes(s7, (
        "【第 7 頁講稿】\n"
        "第三組公式是推論階段的異常分數與 XAI 歸因，也是我們修正幅度最大的地方。\n"
        "為什麼論文分數是 ½‖O1−W‖² + ½‖Ô2−W‖²？它同時結合了第一階段基本誤差與第二階段被放大過的誤差。\n"
        "我們在金融化改造時特別保留『平方和』而不取平均，原因在於數學可加性：總異常分數剛好等於 6 檔 ETF 個別誤差相加，除上總分就能直接算成百分比貢獻度。\n"
        "同時照論文只取最後一天切片 [:, -1, :]，徹底修掉原版把 30 天加總造成的矩形拖尾，並加上 35% 集中度規則來做最終風險診斷。"
    ))

    # =========================================================================
    # SLIDE 8: 實作驗證 (二) — 單日計分對比 Before vs After
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    set_bg(s8, prs, C_BG_LIGHT)
    add_header(s8, "07  CRITICAL BUG FIX #2", "實作驗證 (二)：單日 (Day t) 計分對照，異常率由 40.0% 收斂至 2.5%", 8)

    kpis = [
        ("測試集異常天數 (1,222 天)", "原版 489 天 (40.0%)  ➔  改進版 30 天 (2.5%)", C_BLUE, C_BLUE_BG),
        ("2025/04 關稅衝擊峰值日", "原版 延遲至 05-15  ➔  改進版 精準命中 04-09", C_CORAL, C_CORAL_BG),
        ("分數與歸因口徑 [修正2][6]", "僅取視窗最後一天 [:, -1, :]，兩者總和完全一致", C_EMERALD, C_EMERALD_BG),
    ]
    kw, kh, kgap = Inches(3.78), Inches(0.84), Inches(0.24)
    for i, (kt, kd, kc, kbg) in enumerate(kpis):
        kx = Inches(0.75) + i * (kw + kgap)
        make_card(s8, kx, Inches(1.36), kw, kh, kt, title_color=kc, bg=kbg, border=kc)
        tb_k = s8.shapes.add_textbox(kx + Inches(0.20), Inches(1.78), kw - Inches(0.40), Inches(0.36))
        pk = tb_k.text_frame.paragraphs[0]
        pk.text = kd
        pk.font.name = FONT_ZH
        pk.font.size = Pt(10)
        pk.font.bold = True
        pk.font.color.rgb = C_NAVY_TITLE

    make_card(s8, Inches(0.75), Inches(2.34), Inches(5.78), Inches(3.72), "Before：原版 (30 天視窗總和 → 矩形拖尾 + 40% 紅線條碼)", C_CORAL, C_WHITE, C_CORAL)
    if "orig_trend" in assets:
        add_pic_fit(s8, assets["orig_trend"], Inches(0.88), Inches(2.80), Inches(5.52), Inches(3.16))

    make_card(s8, Inches(6.80), Inches(2.34), Inches(5.78), Inches(3.72), "After：claude改進版 (僅取 Day t → 精準釘住事件當天)", C_BLUE, C_WHITE, C_BLUE)
    if "claude_trend" in assets:
        add_pic_fit(s8, assets["claude_trend"], Inches(6.93), Inches(2.80), Inches(5.52), Inches(3.16))

    make_card(s8, Inches(0.75), Inches(6.18), Inches(11.83), Inches(0.82), bg=C_WHITE, border=C_BORDER)
    add_pic_fit(s8, assets["eq_score_dayt"], Inches(0.90), Inches(6.28), Inches(5.10), Inches(0.62))
    add_bullets(s8, Inches(6.15), Inches(6.28), Inches(6.25), Inches(0.66), [
        [("為什麼原版會出現 30 天高台？", {"bold": True, "color": C_CORAL}),
         (" 原版把 30 天×6 檔共 180 個點加總，只要 4/9 暴衝，往後 29 天滑動視窗都包含這一天！改進版改取最後一天 Day t，前 29 天純作歷史脈絡，徹底根除拖尾。", {})]
    ], font_size=9.8, space_after=0)

    set_notes(s8, (
        "【第 8 頁講稿】\n"
        "把剛剛的公式 ④ 實際跑在同一份截至 2026 年 11 月 13 日的數據上，左右對比非常明顯：\n"
        "左邊原版在 2025 年 4 月出現一個長達 30 天的方形高台，峰值延遲飄移到 5 月 15 日，而且全期有 489 天亮紅燈（40% 警報疲勞）。\n"
        "右邊改進版只取 Day t 計分後，異常收斂到合理的 30 天（2.5%），所有紅點精準釘在衝擊發生當天。"
    ))

    # =========================================================================
    # SLIDE 9: 8 項改進對照總表
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    set_bg(s9, prs, C_BG_LIGHT)
    add_header(s9, "08  SUMMARY OF 8 IMPROVEMENTS", "架構進化對照總表：8 項工程除錯與金融邏輯升級", 9)

    rows = [
        ("[修正1] 對抗訓練更新", "(loss1 + loss2).backward() 相加抵消", "參數分組 + autograd 雙優化器 + 梯度裁剪", "Phase 2 與 Focus 權重真正活化"),
        ("[修正2] 異常計分範圍", "整個 30 天視窗 180 點加總", "僅取視窗最後一天 Day t ([:, -1, :])", "異常率 40.0% ➔ 2.5%，消除拖尾"),
        ("[修正3] ETF 價格欄位", "未還原股利的 Close", "還原股利與分割的 Adj Close", "避免除息跳空被誤判為崩跌"),
        ("[修正4] 驗證集與早停", "切出 win_val 後閒置，盲跑 40 輪", "以 val_rec 早停並載回最佳 Epoch 7", "防止過擬合與對抗後期發散"),
        ("[修正5] Focus Score", "絕對誤差 |O1 − W|", "論文原始平方誤差 (O1 − W)²", "對劇烈異常偏差更敏銳"),
        ("[修正6] 分數歸因口徑", "分數看兩解碼器30天；歸因只看 Ô2 1天", "統一為 ½err(O1) + ½err(Ô2) 的 Day t", "6 檔歸因總和嚴格等於總異常分"),
        ("[修正7] 門檻敏感度", "只算單一 P98 門檻 (1.71)", "輸出 P95 / P97 / P98 / P99 完整對照", "提供客觀門檻選取依據"),
        ("[修正8] 風險診斷規則", "僅看最大貢獻者是否為 SPY", "先檢查最大佔比是否 < 35%（集中度）", "避免全市場同跌被誤判為單一板塊"),
    ]

    tbl_shape = s9.shapes.add_table(len(rows) + 1, 4, Inches(0.75), Inches(1.42), Inches(11.83), Inches(5.52))
    tbl = tbl_shape.table
    col_widths = [Inches(2.15), Inches(3.15), Inches(3.55), Inches(2.98)]
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = w

    for j, h in enumerate(["改進項目", "原版 (etf_trend_anomaly.py)", "改進版 (claude改進版.py)", "實質影響"]):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY_TITLE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_ZH
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    for i, row in enumerate(rows, start=1):
        bg_c = C_WHITE if i % 2 == 1 else RGBColor(241, 245, 249)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_ZH
            p.font.size = Pt(10)
            p.font.bold = (j in (0, 3))
            p.font.color.rgb = C_BLUE if j == 0 else (C_EMERALD if j == 3 else C_TEXT)

    set_notes(s9, (
        "【第 9 頁講稿】\n"
        "這一頁彙整了 claude改進版.py 相對原版的 8 項修正，把前面三組數學公式跟實務除錯一一對齊。"
    ))

    # =========================================================================
    # SLIDE 10: 門檻敏感度與實證事件
    # =========================================================================
    s10 = prs.slides.add_slide(blank)
    set_bg(s10, prs, C_BG_LIGHT)
    add_header(s10, "09  THRESHOLD SENSITIVITY & EMPIRICAL EVENTS", "門檻敏感度分析 (P95–P99) 與盲測集真實黑天鵝捕捉", 10)

    make_card(s10, Inches(0.75), Inches(1.42), Inches(5.55), Inches(5.55), "為什麼選擇訓練集第 98 百分位 (P98)？", C_BLUE)
    add_pic_fit(s10, assets["sensitivity"], Inches(0.90), Inches(1.94), Inches(5.25), Inches(3.38))
    add_bullets(s10, Inches(0.95), Inches(5.40), Inches(5.15), Inches(1.45), [
        [("• P95 (6.9%)：", {"bold": True, "color": C_NAVY_TITLE}), ("一年約 17 天警報，雜訊偏多。", {})],
        [("• P98 (2.5%)：", {"bold": True, "color": C_BLUE}), ("門檻 20.793 抓出 30/1222 天，與理論 2% 尾端高度一致。", {})],
        [("• P99 (1.1%)：", {"bold": True, "color": C_NAVY_TITLE}), ("僅剩 13 天極端危機，容易漏掉前期破裂徵兆。", {})],
    ], font_size=10.2, space_after=4)

    make_card(s10, Inches(6.50), Inches(1.42), Inches(6.08), Inches(5.55), "盲測集異常分數 Top 事件對照 (2022–2026)", C_NAVY_TITLE)
    ev_rows = [
        ("2025-04-09", "246.57", "SPY 32.6% / QQQ 32.5%", "關稅暫緩暴漲反彈"),
        ("2025-04-04", "131.99", "全板塊均攤 (max 20.3%)", "對等關稅恐慌暴跌"),
        ("2025-04-03", "80.02", "XLE 23.7% / SPY 23.5%", "關稅宣布後首日重挫"),
        ("2022-11-10", "61.74", "QQQ 42.2% (科技主導)", "CPI 降溫暴力軋空"),
        ("2026-11-05", "57.17", "QQQ 44.7% (近期最新)", "科技板塊結構破裂"),
        ("2022-09-13", "47.40", "QQQ 33.7% / SPY 31.0%", "CPI 超預期升息恐慌"),
        ("2025-04-10", "46.21", "SPY 33.8% / XLE 27.1%", "關稅週劇烈餘震"),
        ("2022-05-09", "42.15", "XLE 46.8% (能源主導)", "能源獨立脫鉤修正"),
    ]
    etbl_s = s10.shapes.add_table(len(ev_rows) + 1, 4, Inches(6.70), Inches(1.98), Inches(5.68), Inches(4.25))
    etbl = etbl_s.table
    ecw = [Inches(1.12), Inches(0.82), Inches(1.88), Inches(1.86)]
    for j, w in enumerate(ecw):
        etbl.columns[j].width = w
    for j, h in enumerate(["日期", "分數", "主要貢獻", "對應市場事件"]):
        c = etbl.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = C_BLUE
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_ZH
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    for i, rdata in enumerate(ev_rows, start=1):
        bg = C_WHITE if i % 2 == 1 else RGBColor(241, 245, 249)
        for j, val in enumerate(rdata):
            c = etbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = bg
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_ZH
            p.font.size = Pt(9.2)
            p.font.bold = (j in (0, 1))
            p.font.color.rgb = C_CORAL if (j == 1 and float(val) > 70) else C_TEXT

    add_bullets(s10, Inches(6.72), Inches(6.34), Inches(5.65), Inches(0.55), [
        [("註：", {"bold": True, "color": C_MUTED}), ("分數高低反映『跨板塊結構偏離度』而非單純跌點，故 2025-04-09 暴漲脫鉤亦能被精準捕獲。", {})]
    ], font_size=9.2, space_after=0)

    set_notes(s10, (
        "【第 10 頁講稿】\n"
        "在門檻設定上，我們透過 P95–P99 敏感度分析驗證了 P98（20.793）最貼近理論 2% 尾端，共捕獲 30 天。\n"
        "右表顯示分數最高的日期全數對應真實重大事件，如 2025 年 4 月關稅衝擊、2022 年 CPI 公布日與近期 2026-11-05 的科技破裂。"
    ))

    # =========================================================================
    # SLIDE 11: XAI 找戰犯實證
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    set_bg(s11, prs, C_BG_LIGHT)
    add_header(s11, "10  EXPLAINABLE AI (XAI) SECTOR ATTRIBUTION", "XAI 找戰犯實證：結合「誤差集中度 (<35%)」區分風險類型", 11)

    make_card(s11, Inches(0.75), Inches(1.42), Inches(6.85), Inches(5.55), "實證案例：6 檔 ETF 異常貢獻度拆解", C_NAVY_TITLE)
    add_pic_fit(s11, assets["xai_cases"], Inches(0.88), Inches(1.94), Inches(6.60), Inches(4.85))

    make_card(s11, Inches(7.80), Inches(1.42), Inches(4.78), Inches(2.55), "為什麼原版診斷規則會誤判？[修正8]", C_CORAL, C_CORAL_BG, C_CORAL)
    add_bullets(s11, Inches(8.00), Inches(1.96), Inches(4.38), Inches(1.90), [
        [("• 原版規則：", {"bold": True, "color": C_NAVY_TITLE}), ("只看最高貢獻者是否為 SPY，不是就判為『單一板塊破裂』。", {})],
        [("• 實證反例 (2025-04-04)：", {"bold": True, "color": C_CORAL}), ("關稅暴跌當天 6 檔貢獻度極度平均 (XLE 20.3%, SPY 20.0%, XLF 18.2%, XLI 15.6%, QQQ 15.2%, XLV 10.7%)。原版會只因 XLE 多了 0.3% 而誤診為『能源外溢』！", {})],
    ], font_size=10.0, space_after=6)

    make_card(s11, Inches(7.80), Inches(4.15), Inches(4.78), Inches(2.82), "claude改進版.py 雙階診斷邏輯", C_EMERALD, C_WHITE, C_EMERALD)
    add_bullets(s11, Inches(8.00), Inches(4.68), Inches(4.38), Inches(2.18), [
        [("1. Max 佔比 < 35% ➔ ", {"bold": True, "color": C_BLUE}), ("【宏觀系統性風險】：全板塊同步偏離（正確識別 2025-04-04 與 04-09）。", {})],
        [("2. Max ≥ 35% 且為 SPY ➔ ", {"bold": True, "color": C_AMBER}), ("【大盤脫鉤】：大盤與板塊慣性明顯背離。", {})],
        [("3. Max ≥ 35% 且為特定板塊 ➔ ", {"bold": True, "color": C_CORAL}), ("【板塊結構性破裂】：如 2022-05-09 (XLE 46.8%)、2026-11-05 (QQQ 44.7%)。", {})],
    ], font_size=10.0, space_after=6)

    set_notes(s11, (
        "【第 11 頁講稿】\n"
        "將公式 ④ 應用到真實事件上，看 2025-04-04 關稅暴跌當天：六個板塊誤差非常平均，最大只有 20.3%。\n"
        "如果沒設 35% 集中度門檻，原版規則會只因為 XLE 多出 0.3% 而誤判成能源單一板塊問題；加上 35% 門檻後，系統就能正確區分『全市場系統性風險』與像 2026-11-05 (QQQ 44.7%) 這種『單一板塊破裂』。"
    ))

    # =========================================================================
    # SLIDE 12: 深入實驗洞察 — 對抗穩定性觀察
    # =========================================================================
    s12 = prs.slides.add_slide(blank)
    set_bg(s12, prs, C_BG_LIGHT)
    add_header(s12, "11  EMPIRICAL TRAINING INSIGHT", "模型訓練洞察：論文對抗公式發散現象與 3 種訓練模式對比", 12)

    make_card(s12, Inches(0.75), Inches(1.42), Inches(11.83), Inches(2.18), "重要實證發現：為什麼 Epoch 7 後驗證誤差會回升？", C_AMBER, C_AMBER_BG, C_AMBER)
    add_bullets(s12, Inches(0.98), Inches(1.94), Inches(11.35), Inches(1.55), [
        [("• 數學無下界現象：", {"bold": True, "color": C_NAVY_TITLE}), ("Epoch ≥ 3 起權重 w ≤ 0.33 < 0.5，L2 = w‖O2−W‖² − (1−w)‖Ô2−W‖² 的負項權重大於正項。Decoder 2 只要放大輸出就能無限壓低 L2（實測由 +0.74 降至 −2.15），並反拉共用 Encoder。", {})],
        [("• Early Stopping 發揮關鍵保護：", {"bold": True, "color": C_EMERALD}), ("正因驗證集及早介入，我們成功鎖住『對抗剛生效、常態重構尚未被破壞』的 Epoch 7 黃金權重 (val_rec = 0.5877)。", {})],
        [("• 對照 TranAD 官方 GitHub 實作：", {"bold": True, "color": C_BLUE}), ("有趣的是，論文官方開源碼其實省略了帶負號的 L2，只留下 L1 以維持訓練穩定；這也解釋了為什麼直接照論文公式寫會出現震盪。", {})],
    ], font_size=10.3, space_after=6)

    modes = [
        ("方案 A：嚴格論文公式 + 早停（本版採用）", C_BLUE, C_BLUE_BG, [
            [("• 機制：", {"bold": True}), ("雙優化器真實 Min-Max + 梯度裁剪 + Early Stopping。", {})],
            [("• 優點：", {"bold": True, "color": C_EMERALD}), ("完全忠於論文公式，已實證能穩定抓取 30 天關鍵異常。", {})],
            [("• 限制：", {"bold": True, "color": C_CORAL}), ("訓練輪數受限於早停點（約 Epoch 7）。", {})],
        ]),
        ("方案 B：對抗項加 Clamp 上限（待測消融）", C_EMERALD, C_EMERALD_BG, [
            [("• 機制：", {"bold": True}), ("對 L2 中的 mse(Ô2, W) 設上限 clamp(max=C)。", {})],
            [("• 優點：", {"bold": True, "color": C_EMERALD}), ("保留對抗放大異常的優勢，同時給定數值下界，允許安全訓練更久。", {})],
            [("• 目的：", {"bold": True, "color": C_BLUE}), ("觀察更充分訓練下對小異常的 Recall 是否再提升。", {})],
        ]),
        ("方案 C：官方開源碼版（L1-only 對照）", C_MUTED, C_WHITE, [
            [("• 機制：", {"bold": True}), ("移除負號項 L2，全參數共同最小化 L1。", {})],
            [("• 優點：", {"bold": True, "color": C_EMERALD}), ("訓練極度平穩、不發散。", {})],
            [("• 目的：", {"bold": True, "color": C_BLUE}), ("作為實驗對照組，量化『真對抗懲罰』究竟帶來多少增益。", {})],
        ]),
    ]
    mw, mgap = Inches(3.75), Inches(0.29)
    for i, (mt, mc, mbg, mitems) in enumerate(modes):
        mx = Inches(0.75) + i * (mw + mgap)
        make_card(s12, mx, Inches(3.82), mw, Inches(3.15), mt, title_color=mc, bg=mbg, border=mc)
        add_bullets(s12, mx + Inches(0.20), Inches(4.42), mw - Inches(0.40), Inches(2.40), mitems, font_size=10.2, space_after=8)

    set_notes(s12, (
        "【第 12 頁講稿】\n"
        "把公式 ③ 真的寫成雙優化器後，我們也觀察到一個重要現象：當 n >= 3（1/n < 0.5）時，L2 因為沒有下界會一路往負數掉。\n"
        "反觀 TranAD 官方開源碼其實把負號項拿掉只留 L1。我們靠 Early Stopping 成功停在 Epoch 7，接下來可以做 A/B/C 三組消融對比實驗。"
    ))

    # =========================================================================
    # SLIDE 13: 總結與向教授請益
    # =========================================================================
    s13 = prs.slides.add_slide(blank)
    set_bg(s13, prs, C_BG_LIGHT)
    add_header(s13, "12  CONCLUSION & NEXT STEPS", "總結與向教授請益：從公式落地到完整專題實驗矩陣", 13)

    make_card(s13, Inches(0.75), Inches(1.42), Inches(5.70), Inches(5.55), "claude改進版.py 三大核心交付成果", C_BLUE, C_WHITE, C_BLUE)
    add_bullets(s13, Inches(0.98), Inches(2.04), Inches(5.25), Inches(4.75), [
        [("1. 論文公式與實作完全對齊", {"bold": True, "size": 12, "color": C_BLUE}), ("\n透徹理解並落地 PE、Causal Mask、平方 Focus 與雙優化器 Min-Max 對抗訓練，不再被正負抵消 Bug 架空。", {})],
        [("2. 單日計分精準對齊真實市場", {"bold": True, "size": 12, "color": C_EMERALD}), ("\n改取 Day t 並統一可加性平方和，異常率從 40.0% 降為 2.5%，精準命中 2025 關稅劇震與 2022 CPI 衝擊。", {})],
        [("3. 可解釋金融風險診斷升級", {"bold": True, "size": 12, "color": C_CORAL}), ("\n透過誤差佔比與 35% 集中度門檻，能具體指導投資人區分『系統性避險』與『板塊調倉』。", {})],
    ], font_size=10.8, space_after=16)

    make_card(s13, Inches(6.75), Inches(1.42), Inches(5.83), Inches(5.55), "向 教授請益與下一步對齊事項", C_NAVY_TITLE, C_BLUE_BG, C_NAVY_TITLE)
    add_bullets(s13, Inches(6.98), Inches(2.04), Inches(5.38), Inches(4.75), [
        [("Q1. 對抗訓練模式的消融比較", {"bold": True, "size": 11.5, "color": C_NAVY_TITLE}), ("\n是否建議補齊『論文公式+早停 vs. Clamp 限幅 vs. 官方 L1-only』三組對比，量化對抗懲罰的實際效益？", {})],
        [("Q2. 門檻與量化評估指標", {"bold": True, "size": 11.5, "color": C_NAVY_TITLE}), ("\n在 P95–P99 敏感度外，是否建議實作論文中的 POT (極值理論) 動態門檻，或以大跌日做 Lead-Time 評分？", {})],
        [("Q3. 同步升級主專案架構", {"bold": True, "size": 11.5, "color": C_NAVY_TITLE}), ("\n確認方向後，我們將把這套修正同步套用至 etf_tranad_project/ 的 exp1~exp3，並整合 GARCH 與 EWS 預警系統。", {})],
    ], font_size=10.5, space_after=16)

    set_notes(s13, (
        "【第 13 頁講稿】\n"
        "總結來說，這份 claude改進版.py 讓我們徹底掌握了每一條論文公式在金融上的意義與用途，並修復了實作落差。\n"
        "接下來想請教授在三個方向給予建議：對抗模式消融實驗、POT 或事件評估指標、以及合回完整專案管線。謝謝教授！"
    ))

    prs.save(PPTX_OUT)
    print(f"Saved PPTX -> {PPTX_OUT}")
    return prs


def audit_pptx_geometry(prs):
    warnings = []
    W, H = prs.slide_width, prs.slide_height
    for s_idx, slide in enumerate(prs.slides, start=1):
        for sh in slide.shapes:
            if sh.left < 0 or sh.top < 0 or sh.left + sh.width > W + 5000 or sh.top + sh.height > H + 5000:
                warnings.append(f"Slide {s_idx}: shape '{sh.name}' exceeds slide bounds")
    if warnings:
        print("Geometry warnings:", warnings)
    else:
        print(f"Geometry audit passed: {len(prs.slides)} slides checked, 0 out-of-bounds shapes.")


# ---------------------------------------------------------------------------
# 4. 生成對應的 13 頁互動式 HTML 簡報
# ---------------------------------------------------------------------------
def build_html(assets):
    def rel(p):
        return "claude_assets/" + os.path.basename(p)

    slides_html = [
        # 1
        f"""
        <section class="slide active dark-slide">
          <div class="cover-wrap">
            <span class="pill-badge">US ETF ANOMALY RADAR · TRANAD CAPSTONE PROJECT</span>
            <h1 class="cover-h1">跨板塊美股 ETF 風險異常偵測與<br>成因診斷系統（基於 TranAD）</h1>
            <p class="cover-sub">「美股市場異常雷達」— 抓出板塊脫鉤變天訊號、找出出事產業，並剖析背後數學與程式改進</p>
            <div class="grid-3 cover-cards">
              <div class="dark-card border-blue">
                <h3>解決什麼痛點？</h3>
                <p>抓出「大盤看似沒事、底下板塊已脫鉤」的結構風險，避免只看單日漲跌幅漏報</p>
              </div>
              <div class="dark-card border-emerald">
                <h3>能幫使用者幹嘛？</h3>
                <p>亮燈時立刻判斷該「全市場降槓桿避險」還是「只針對單一板塊換倉」</p>
              </div>
              <div class="dark-card border-coral">
                <h3>理論剖析與工程除錯</h3>
                <p>拆解 4 組論文公式並修復抵消 / 拖尾 Bug；盲測異常率 40.0% ➔ 2.5%</p>
              </div>
            </div>
            <div class="cover-foot">主講檔案：<code>firstry/claude改進版.py</code> ｜ 對照文獻：Tuli et al., TranAD (VLDB 2022) ｜ 資料期間：2010-01-05 ~ 2026-11-13</div>
          </div>
        </section>""",

        # 2
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">01 PLAIN-ENGLISH PURPOSE &amp; WORKFLOW</div><h2>我們在做什麼？用「平靜期默契」抓出市場變天與出事產業</h2></div>
            <span class="page-pill">02 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-3" style="margin-bottom:14px;">
            <div class="card bg-coral border-coral">
              <h3 class="c-coral">1. 痛點：只看大盤漲跌會漏掉風險</h3>
              <ul>
                <li><strong>傳統看盤盲點：</strong>只盯 SPY 今天跌幾%，但危機剛冒頭時大盤看似風平浪靜，底下板塊已開始「各走各的」。</li>
                <li><strong>為何選 TranAD？</strong>一般 Autoencoder 太會「順手把小異常修成正常」；TranAD 用兩階段機制放大脫鉤裂縫。</li>
              </ul>
            </div>
            <div class="card bg-blue border-blue">
              <h3 class="c-blue">2. 做法：教 AI 記住平靜期的默契</h3>
              <ul>
                <li><strong>學習健康範本：</strong>拿 2010–2018 年平穩市場教模型記住 <code>SPY, QQQ, XLF, XLE, XLV, XLI</code> 誰跟誰一起動。</li>
                <li><strong>怎麼抓異常？</strong>2022–2026 盲測時模型照過去默契推算走勢，「猜錯越多 ＝ 今天市場越反常」。</li>
              </ul>
            </div>
            <div class="card bg-emerald border-emerald">
              <h3 class="c-emerald">3. 價值：不只亮燈，還給出避險處方</h3>
              <ul>
                <li><strong>過濾警報疲勞：</strong>只對比平靜期 98% 日子更反常的日子亮燈（2.5% 關鍵警報，不再天天亂叫）。</li>
                <li><strong>看佔比決定動作：</strong>誤差分散 (&lt;35%) ➔ 全市場降槓桿；集中 (≥35%) ➔ 抓出單一戰犯板塊換倉。</li>
              </ul>
            </div>
          </div>
          <div class="card kpi-row">
            <h3>claude改進版.py 大白話運作 4 步驟（昨晚收盤跑一次 ➔ 決定今天怎麼調整部位）</h3>
            <div class="grid-4">
              <div class="step-box bg-blue"><h4>Step 1 看這 6 檔怎麼動</h4><p>計算 6 大 ETF 每日漲跌幅<br>只用 2010–18 標準縮放（無前視）</p></div>
              <div class="step-box bg-emerald"><h4>Step 2 學平靜期連動默契</h4><p>看過去 30 天（因果遮罩防偷看）<br>雙優化器對抗放大可疑偏差</p></div>
              <div class="step-box bg-amber"><h4>Step 3 看今天猜錯多少</h4><p>只算最新一天 Day t 誤差總和<br>超過平靜期 98% 水準就亮燈</p></div>
              <div class="step-box bg-coral"><h4>Step 4 抓戰犯決定動作</h4><p>拆解 6 檔 ETF 誤差佔比 %<br>依 35% 門檻分系統風險或換板塊</p></div>
            </div>
          </div>
        </section>""",

        # 3
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">02 INPUT FORMULATION &amp; FINANCIAL ADAPTATIONS</div><h2>金融化輸入建構：對數報酬率、Z-score 與無前視時間切分</h2></div>
            <span class="page-pill">03 / {TOTAL_PAGES}</span>
          </div>
          <div class="card" style="padding:6px 10px; margin-bottom:10px;">
            <img src="{rel(assets['timeline'])}" class="fit-img" style="max-height:152px;" alt="timeline">
          </div>
          <div class="card bg-blue border-blue" style="flex-direction:row; align-items:center; gap:16px; padding:8px 16px; margin-bottom:10px;">
            <img src="{rel(assets['eq_input'])}" style="max-height:58px; max-width:68%; object-fit:contain;" alt="eq_input">
            <p style="font-size:12px; color:#334155;"><strong>公式意涵：</strong>將非平穩價格轉為平穩對數報酬率，再以訓練期標準差縮放成 30×6 輸入矩陣。</p>
          </div>
          <div class="grid-3">
            <div class="card border-blue">
              <h3 class="c-blue">為什麼公式長這樣 &amp; 改 Adj Close</h3>
              <ul>
                <li><strong>取對數 ln(P<sub>t</sub>/P<sub>t-1</sub>)：</strong>股價隨通膨長期走高，直接餵價格會把高價當異常；對數報酬平穩且可時間累加。</li>
                <li><strong>[修正3] Adj Close：</strong>修正原版 Close 在除息日跳空被誤算負報酬的假警報。</li>
              </ul>
            </div>
            <div class="card border-emerald">
              <h3 class="c-emerald">為何用 Z-score 並拿掉 Sigmoid</h3>
              <ul>
                <li><strong>原論文設定：</strong>感測器做 Min-Max 到 [0,1] 故接 Sigmoid；但金融厚尾下 Min-Max 會被極端值壓扁常態波動。</li>
                <li><strong>金融化改造：</strong>採 Z-score + 純 Linear 輸出，跌破 −5σ 也能完整重構。</li>
              </ul>
            </div>
            <div class="card border-amber">
              <h3 class="c-amber">怎麼用：防前視與 P98 健康錨定</h3>
              <ul>
                <li><strong>嚴格隔離統計量：</strong>測試期全部套用 2010–18 計算出的 μ 與 σ。</li>
                <li><strong>為何不用 Val 定門檻：</strong>2020 疫情會把門檻拉爆導致 2022+ 漏報，故門檻錨定訓練集 P98 (20.793)，Val 專職早停。</li>
              </ul>
            </div>
          </div>
        </section>""",

        # 4 [NEW]
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">03 PAPER FORMULA ANATOMY #1</div><h2>論文公式剖析 (一)：位置編碼、因果遮罩與自條件聚焦機制</h2></div>
            <span class="page-pill">04 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-2 kpi-row" style="margin-bottom:12px;">
            <div class="card bg-blue border-blue" style="padding:10px 14px;">
              <h3 class="c-blue" style="margin-bottom:4px;">公式 ① 正弦位置編碼 (Positional Encoding)</h3>
              <img src="{rel(assets['eq_pe'])}" class="eq-img" style="max-height:66px; margin:2px auto;" alt="eq_pe">
            </div>
            <div class="card bg-emerald border-emerald" style="padding:10px 14px;">
              <h3 class="c-emerald" style="margin-bottom:4px;">公式 ② 因果遮罩注意力與 Focus Score</h3>
              <img src="{rel(assets['eq_attn_focus'])}" class="eq-img" style="max-height:66px; margin:2px auto;" alt="eq_attn_focus">
            </div>
          </div>
          <div class="grid-3">
            <div class="card">
              <h3>1. 為什麼公式長這樣？</h3>
              <ul>
                <li><strong class="c-blue">PE 用 sin/cos 多頻率波：</strong>Attention 本質無序，多波長正餘弦讓「相對天數差 Δk」可被線性變換辨識，且無額外參數不易過擬合。</li>
                <li><strong class="c-emerald">為什麼加 M (上三角 −∞)？</strong>經 softmax 後 e^(−∞)=0，強制第 t 天完全屏蔽未來 t+1…30 天。</li>
                <li><strong>為什麼 F = (O1 − W)²？</strong>平方把誤差轉正並放大顯著偏離，成為指導 Phase 2 的「錯誤焦點圖」。</li>
              </ul>
            </div>
            <div class="card border-blue">
              <h3 class="c-blue">2. 我們為何選用這組機制？</h3>
              <ul>
                <li><strong class="c-blue">金融時序具方向性：</strong>「先跌後彈」與「先彈後跌」風險完全不同，必須靠 PE 鎖定 30 天內的先後順序。</li>
                <li><strong class="c-emerald">嚴防偷看未來：</strong>Window Encoder 透過因果遮罩確保推論 t 時不洩漏未來報酬。</li>
                <li><strong>克服一般 AE 漏報：</strong>靠 F 顯影微小裂縫，經 Cross-Attention (Q=視窗, K/V=含 F 編碼) 放大檢視。</li>
              </ul>
            </div>
            <div class="card bg-emerald border-emerald">
              <h3 class="c-emerald">3. 在 claude改進版.py 怎麼用？</h3>
              <ul>
                <li><strong>PositionalEncoding (L82)：</strong>接在 <code>proj1</code> 與 <code>proj2</code> 後，注入 30×64 位置向量。</li>
                <li><strong>因果遮罩 M (L132)：</strong><code>torch.triu(full((K,K),-inf), 1)</code> 傳入 <code>self_attn2</code>。</li>
                <li><strong>兩階段推論 (L154)：</strong><br>① <code>o1, o2 = model(w)</code> (F=0)<br>② <code>focus = (o1 - w) ** 2</code> [修正5]<br>③ <code>_, o2_hat = model(w, focus)</code></li>
              </ul>
            </div>
          </div>
        </section>""",

        # 5 [NEW]
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">04 PAPER FORMULA ANATOMY #2</div><h2>論文公式剖析 (二)：演化式雙解碼器對抗損失 (L1, L2)</h2></div>
            <span class="page-pill">05 / {TOTAL_PAGES}</span>
          </div>
          <div class="card bg-blue border-blue kpi-row" style="margin-bottom:12px; padding:10px 16px;">
            <h3 style="margin-bottom:6px;">公式 ③ 演化對抗損失函數（Adversarial Loss，其中 n = epoch）</h3>
            <div style="display:grid; grid-template-columns:1.25fr 1fr; gap:16px; align-items:center;">
              <img src="{rel(assets['eq_adv_losses'])}" style="max-height:94px; max-width:100%; object-fit:contain;" alt="eq_adv_losses">
              <ul style="margin:0;">
                <li><strong class="c-blue">左半項 1/n · ‖O−W‖²：</strong>基本功項，讓兩解碼器無提示時都會正常重構。</li>
                <li><strong class="c-coral">右半項 ±(1−1/n) · ‖Ô2−W‖²：</strong>核心博弈！L1 用「+」壓小 Ô2；L2 用「−」放大 Ô2。</li>
                <li><strong class="c-emerald">動態權重 w=1/n：</strong>由 1.0 → 0.5 → 0.1 遞減，先練基本功再逐輪開啟對抗。</li>
              </ul>
            </div>
          </div>
          <div class="grid-3">
            <div class="card">
              <h3>1. 為什麼公式長這樣？(Min-Max)</h3>
              <ul>
                <li><strong class="c-blue">借鑑 GAN 雙人博弈：</strong>Decoder 1 像生成器（想做到給提示也重構完美）；Decoder 2 像判別器（看到 Focus 就刻意把誤差放大）。</li>
                <li><strong class="c-amber">為什麼要有 1/n 衰減？</strong>Epoch 1 時 O1 還很爛，Focus 全是雜訊；n=1 時 1−1/n=0 純學重構，n≥2 才交棒給對抗。</li>
              </ul>
            </div>
            <div class="card border-coral">
              <h3 class="c-coral">2. 我們為何在金融上使用它？</h3>
              <ul>
                <li><strong class="c-coral">解決「異常被平滑化」：</strong>正常市場 Focus 趨近 0，Decoder 2 無從放大；一旦脫鉤，Focus 亮起後 Decoder 2 會把誤差成倍放大，拉開正常與異常分數。</li>
                <li><strong>解釋原版為何失效：</strong>正負號正是為了對立，絕不能把 L1+L2 相加後一起求導！</li>
              </ul>
            </div>
            <div class="card bg-blue border-blue">
              <h3 class="c-blue">3. 在 claude改進版.py 怎麼用？</h3>
              <ul>
                <li><strong>參數分組 (L164)：</strong><code>main_params</code> 配 <code>opt_main</code> (1e-3)；<code>dec2_params</code> 配 <code>opt_dec2</code> (5e-4)。</li>
                <li><strong>獨立求導 (L195)：</strong><code>autograd.grad</code> 分別求導並加 <code>grad_clip=1.0</code> 防發散。</li>
                <li><strong class="c-emerald">早停指標巧思：</strong>只看 Phase 1 的 O1 驗證誤差，因為 Ô2 被要求放大誤差不適合作收斂指標。</li>
              </ul>
            </div>
          </div>
        </section>""",

        # 6
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">05 CRITICAL BUG FIX #1</div><h2>實作驗證 (一)：修復原版 Loss 相加抵消，實測對抗真正生效</h2></div>
            <span class="page-pill">06 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-2">
            <div style="display:flex; flex-direction:column; gap:14px;">
              <div class="card bg-coral border-coral">
                <h3 class="c-coral">原版 Bug：L1 + L2 合併求導，對抗項精確抵消為 0</h3>
                <img src="{rel(assets['eq_cancel'])}" class="eq-img" alt="cancel">
                <ul>
                  <li><strong>問題 (L156)：</strong><code>(loss1 + loss2).backward()</code> 使 ±(1−w)‖Ô2−W‖² 互相抵消。</li>
                  <li><strong>後果：</strong>Phase 2 梯度恆為 0，<code>proj1</code> Focus 權重停在隨機亂數；盲跑 40 輪嚴重過擬合 (Train P98=1.71 ➔ 誤報 40%)。</li>
                </ul>
              </div>
              <div class="card bg-blue border-blue">
                <h3 class="c-blue">claude改進版.py 實作效果 [修正1][4][5]</h3>
                <img src="{rel(assets['eq_split'])}" class="eq-img" alt="split">
                <ul>
                  <li><strong>參數分組：</strong>Encoder+Dec1 只吃 L1 梯度；Dec2 只吃 L2 梯度 (lr=5e-4, grad_clip=1.0)。</li>
                  <li><strong>對齊論文 + 早停：</strong>Focus 改平方誤差 <code>(O1-W)**2</code>，依 <code>val_rec</code> 鎖定最佳 Epoch 7 (0.5877)。</li>
                </ul>
              </div>
            </div>
            <div class="card">
              <h3>實測訓練動態 (Epoch 1–15 真實數據)</h3>
              <img src="{rel(assets['train_dyn'])}" class="fit-img" style="max-height:385px;" alt="training dynamics">
              <p class="caption">Epoch 3 起 L2 如公式預測轉負（w &lt; 0.5），證實對抗啟動；Epoch 7 達最低驗證誤差並自動保留。</p>
            </div>
          </div>
        </section>""",

        # 7 [NEW]
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">06 PAPER FORMULA ANATOMY #3</div><h2>論文公式剖析 (三)：綜合異常分數 s<sub>t</sub> 與 XAI 可加性歸因分解</h2></div>
            <span class="page-pill">07 / {TOTAL_PAGES}</span>
          </div>
          <div class="card bg-blue border-blue kpi-row" style="margin-bottom:12px; padding:10px 16px;">
            <h3 style="margin-bottom:4px;">公式 ④ 單日異常分數與逐維度誤差歸因（t = 視窗最後一天，i = 第 i 檔 ETF）</h3>
            <img src="{rel(assets['eq_score_xai'])}" class="eq-img" style="max-height:74px; margin:2px auto;" alt="eq_score_xai">
          </div>
          <div class="grid-3">
            <div class="card">
              <h3>1. 為什麼公式長這樣？(算子設計)</h3>
              <ul>
                <li><strong class="c-blue">為什麼 ½ O<sub>1</sub> + ½ Ô<sub>2</sub>？</strong>同時融合 Phase 1 基礎重構偏離與 Phase 2 被對抗放大的焦點偏離，兼顧穩定性與靈敏度。</li>
                <li><strong class="c-emerald">為何用平方和而非 Mean？</strong>平方和具備嚴格「可加性」——總分剛好等於 6 檔 ETF 個別誤差之和，不稀釋總衝擊。</li>
                <li><strong class="c-coral">為何只取最後一天 t？</strong>論文僅評當下向量，前 29 天只供脈絡。</li>
              </ul>
            </div>
            <div class="card border-emerald">
              <h3 class="c-emerald">2. 我們為何這樣改造？(金融需求)</h3>
              <ul>
                <li><strong class="c-coral">[修正2] 根除 30 天高台：</strong>原版加總 30×6=180 點，導致一次衝擊往後誤響 30 天，當日訊號還被稀釋 30 倍。</li>
                <li><strong class="c-emerald">[修正6] 統一計分與 XAI 口徑：</strong>讓 s<sub>t</sub> = Σ Attr<sub>t,i</sub> 嚴格成立，6 檔佔比加起來等於 100%。</li>
                <li><strong>[修正8] 解開黑盒子：</strong>直接給出該賣特定板塊還是全面避險的依據。</li>
              </ul>
            </div>
            <div class="card bg-blue border-blue">
              <h3 class="c-blue">3. 在 claude改進版.py 怎麼用？</h3>
              <ul>
                <li><strong>單日切片 (L252)：</strong><code>feat_err = 0.5*(o1[:,-1,:]-w[:,-1,:])**2 + 0.5*(o2_hat[:,-1,:]-w[:,-1,:])**2</code>，<code>scores = feat_err.sum(dim=1)</code>。</li>
                <li><strong>健康常態門檻 (L309)：</strong>取訓練集 s_t 的第 98 百分位 (20.793) 判異常。</li>
                <li><strong class="c-coral">集中度診斷 (L258)：</strong>佔比 &lt; 35% 判系統性風險；≥ 35% 判大盤脫鉤或單一板塊破裂。</li>
              </ul>
            </div>
          </div>
        </section>""",

        # 8
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">07 CRITICAL BUG FIX #2</div><h2>實作驗證 (二)：單日 (Day t) 計分對照，異常率由 40.0% 收斂至 2.5%</h2></div>
            <span class="page-pill">08 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-3 kpi-row" style="margin-bottom:12px;">
            <div class="kpi bg-blue border-blue"><span>測試集異常天數 (1,222 天)</span><strong>489 天 (40.0%) ➔ 30 天 (2.5%)</strong></div>
            <div class="kpi bg-coral border-coral"><span>2025/04 關稅衝擊峰值日</span><strong>延遲至 05-15 ➔ 精準 04-09</strong></div>
            <div class="kpi bg-emerald border-emerald"><span>分數與歸因口徑 [修正2][6]</span><strong>統一只取 Day t，總和完全一致</strong></div>
          </div>
          <div class="grid-2" style="margin-bottom:12px; flex:1; min-height:0;">
            <div class="card border-coral" style="padding:10px 12px;">
              <h3 class="c-coral" style="font-size:13px; margin-bottom:4px;">Before：原版 (30 天加總 → 矩形拖尾 + 40% 條碼)</h3>
              <img src="{rel(assets.get('orig_trend', assets['claude_trend']))}" class="fit-img" style="max-height:295px;" alt="original plot">
            </div>
            <div class="card border-blue" style="padding:10px 12px;">
              <h3 class="c-blue" style="font-size:13px; margin-bottom:4px;">After：claude改進版 (只取 Day t → 精準釘住衝擊日)</h3>
              <img src="{rel(assets.get('claude_trend', assets['orig_trend']))}" class="fit-img" style="max-height:295px;" alt="improved plot">
            </div>
          </div>
          <div class="card" style="flex-direction:row; align-items:center; gap:18px; padding:10px 18px;">
            <img src="{rel(assets['eq_score_dayt'])}" style="height:42px;" alt="score eq">
            <p style="font-size:12.5px; color:#334155;"><strong>為什麼會有 30 天高台？</strong> 原版把 30 天×6 檔共 180 點全加起來，只要 4/9 暴衝，往後 29 天視窗都帶著這一天！改取最後一天 Day t 後，前 29 天純作歷史脈絡，訊號立即乾淨銳利。</p>
          </div>
        </section>""",

        # 9
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">08 SUMMARY OF 8 IMPROVEMENTS</div><h2>架構進化對照總表：8 項工程除錯與金融邏輯升級</h2></div>
            <span class="page-pill">09 / {TOTAL_PAGES}</span>
          </div>
          <div class="card" style="padding:14px;">
            <table class="clean-table">
              <thead>
                <tr><th>改進項目</th><th>原版 (etf_trend_anomaly.py)</th><th>改進版 (claude改進版.py)</th><th>實質影響</th></tr>
              </thead>
              <tbody>
                <tr><td class="c-blue bold">[修正1] 對抗訓練更新</td><td>(loss1 + loss2).backward() 抵消</td><td>參數分組 + autograd 雙優化器 + 梯度裁剪</td><td class="c-emerald bold">Phase 2 與 Focus 權重真正活化</td></tr>
                <tr><td class="c-blue bold">[修正2] 異常計分範圍</td><td>整個 30 天視窗 180 點加總</td><td>僅取視窗最後一天 Day t ([:, -1, :])</td><td class="c-emerald bold">異常率 40.0% ➔ 2.5%，消除拖尾</td></tr>
                <tr><td class="c-blue bold">[修正3] ETF 價格欄位</td><td>未還原股利的 Close</td><td>還原股利與分割的 Adj Close</td><td class="c-emerald bold">避免除息跳空被誤判為崩跌</td></tr>
                <tr><td class="c-blue bold">[修正4] 驗證集與早停</td><td>切出 win_val 後閒置，盲跑 40 輪</td><td>以 val_rec 早停並載回最佳 Epoch 7</td><td class="c-emerald bold">防止過擬合與對抗後期發散</td></tr>
                <tr><td class="c-blue bold">[修正5] Focus Score</td><td>絕對誤差 |O1 − W|</td><td>論文原始平方誤差 (O1 − W)²</td><td class="c-emerald bold">對劇烈異常偏差更敏銳</td></tr>
                <tr><td class="c-blue bold">[修正6] 分數歸因口徑</td><td>分數看兩解碼器30天；歸因只看 Ô2 1天</td><td>統一為 ½err(O1) + ½err(Ô2) 的 Day t</td><td class="c-emerald bold">6 檔歸因總和嚴格等於總異常分</td></tr>
                <tr><td class="c-blue bold">[修正7] 門檻敏感度</td><td>只算單一 P98 門檻 (1.71)</td><td>輸出 P95 / P97 / P98 / P99 完整對照</td><td class="c-emerald bold">提供客觀門檻選取依據</td></tr>
                <tr><td class="c-blue bold">[修正8] 風險診斷規則</td><td>僅看最大貢獻者是否為 SPY</td><td>先檢查最大佔比是否 &lt; 35%（集中度）</td><td class="c-emerald bold">避免全市場同跌被誤判為單一板塊</td></tr>
              </tbody>
            </table>
          </div>
        </section>""",

        # 10
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">09 THRESHOLD SENSITIVITY &amp; EMPIRICAL EVENTS</div><h2>門檻敏感度分析 (P95–P99) 與盲測集真實黑天鵝捕捉</h2></div>
            <span class="page-pill">10 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-2">
            <div class="card">
              <h3 class="c-blue">為什麼選擇訓練集第 98 百分位 (P98)？</h3>
              <img src="{rel(assets['sensitivity'])}" class="fit-img" style="max-height:285px;" alt="sensitivity">
              <ul>
                <li><strong>P95 (6.9%)：</strong>一年約 17 天警報，雜訊偏多。</li>
                <li><strong class="c-blue">P98 (2.5%)：</strong>門檻 20.793 抓出 30/1222 天，最貼近理論 2% 尾端。</li>
                <li><strong>P99 (1.1%)：</strong>僅剩 13 天極端危機，恐漏抓早期破裂徵兆。</li>
              </ul>
            </div>
            <div class="card">
              <h3>盲測集異常分數 Top 事件對照 (2022–2026)</h3>
              <table class="clean-table compact">
                <thead><tr><th>日期</th><th>分數</th><th>主要貢獻</th><th>對應市場事件</th></tr></thead>
                <tbody>
                  <tr><td class="bold">2025-04-09</td><td class="c-coral bold">246.57</td><td>SPY 32.6% / QQQ 32.5%</td><td>關稅暫緩暴漲反彈</td></tr>
                  <tr><td class="bold">2025-04-04</td><td class="c-coral bold">131.99</td><td>全板塊均攤 (max 20.3%)</td><td>對等關稅恐慌暴跌</td></tr>
                  <tr><td class="bold">2025-04-03</td><td class="c-coral bold">80.02</td><td>XLE 23.7% / SPY 23.5%</td><td>關稅宣布首日重挫</td></tr>
                  <tr><td class="bold">2022-11-10</td><td class="bold">61.74</td><td>QQQ 42.2% (科技主導)</td><td>CPI 降溫暴力軋空</td></tr>
                  <tr><td class="bold">2026-11-05</td><td class="bold">57.17</td><td>QQQ 44.7% (近期最新)</td><td>科技板塊結構破裂</td></tr>
                  <tr><td class="bold">2022-09-13</td><td class="bold">47.40</td><td>QQQ 33.7% / SPY 31.0%</td><td>CPI 超預期升息恐慌</td></tr>
                  <tr><td class="bold">2025-04-10</td><td class="bold">46.21</td><td>SPY 33.8% / XLE 27.1%</td><td>關稅週劇烈餘震</td></tr>
                  <tr><td class="bold">2022-05-09</td><td class="bold">42.15</td><td>XLE 46.8% (能源主導)</td><td>能源獨立脫鉤修正</td></tr>
                </tbody>
              </table>
              <p class="caption">註：異常分數衡量「定價慣性偏離度」，故極端暴漲與暴跌脫鉤皆可精準捕獲。</p>
            </div>
          </div>
        </section>""",

        # 11
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">10 EXPLAINABLE AI (XAI) SECTOR ATTRIBUTION</div><h2>XAI 找戰犯實證：結合「誤差集中度 (&lt;35%)」區分風險類型</h2></div>
            <span class="page-pill">11 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-2-asym">
            <div class="card">
              <h3>實證案例：6 檔 ETF 異常貢獻度拆解</h3>
              <img src="{rel(assets['xai_cases'])}" class="fit-img" style="max-height:420px;" alt="xai cases">
            </div>
            <div style="display:flex; flex-direction:column; gap:14px;">
              <div class="card bg-coral border-coral">
                <h3 class="c-coral">為什麼原版診斷規則會誤判？[修正8]</h3>
                <ul>
                  <li><strong>原版規則：</strong>只看第一名是不是 SPY，不是就判單一板塊破裂。</li>
                  <li><strong>實證反例 (2025-04-04)：</strong>當天 6 檔貢獻極度平均 (XLE 20.3%、SPY 20.0%、XLF 18.2%、XLI 15.6%、QQQ 15.2%、XLV 10.7%)。原版會只因 XLE 多了 0.3% 誤診為能源外溢！</li>
                </ul>
              </div>
              <div class="card border-emerald">
                <h3 class="c-emerald">claude改進版.py 雙階診斷邏輯</h3>
                <ul>
                  <li><strong class="c-blue">1. Max &lt; 35% ➔ 【宏觀系統性風險】：</strong>多板塊齊崩（正確識別 2025-04-04 與 04-09）。</li>
                  <li><strong class="c-amber">2. Max ≥ 35% 且為 SPY ➔ 【大盤脫鉤】：</strong>大盤與板塊慣性背離。</li>
                  <li><strong class="c-coral">3. Max ≥ 35% 且為特定板塊 ➔ 【板塊結構破裂】：</strong>如 2022-05-09 (XLE 46.8%)、2026-11-05 (QQQ 44.7%)。</li>
                </ul>
              </div>
            </div>
          </div>
        </section>""",

        # 12
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">11 EMPIRICAL TRAINING INSIGHT</div><h2>模型訓練洞察：論文對抗公式發散現象與 3 種訓練模式對比</h2></div>
            <span class="page-pill">12 / {TOTAL_PAGES}</span>
          </div>
          <div class="card bg-amber border-amber kpi-row" style="margin-bottom:14px;">
            <h3 class="c-amber">重要實證發現：為什麼 Epoch 7 後驗證誤差會回升？</h3>
            <ul>
              <li><strong>數學無下界現象：</strong>Epoch ≥ 3 起 w ≤ 0.33 &lt; 0.5，L2 負項權重大於正項，Decoder 2 可透過放大輸出無限拉低 L2（實測 +0.74 降至 −2.15）並反拉共用 Encoder。</li>
              <li><strong>Early Stopping 保護：</strong>驗證集早停鎖住「對抗剛生效、常態重構尚未受損」的 Epoch 7 黃金權重 (val_rec = 0.5877)。</li>
              <li><strong>對照官方實作：</strong>TranAD 官方 GitHub 其實省略了負號項 L2 只留 L1 以換取穩定，這解釋了照論文公式寫為何出現震盪。</li>
            </ul>
          </div>
          <div class="grid-3">
            <div class="card bg-blue border-blue">
              <h3 class="c-blue">方案 A：論文公式+早停（本版）</h3>
              <ul>
                <li><strong>機制：</strong>雙優化器 Min-Max + 梯度裁剪 + Early Stopping。</li>
                <li><strong>優點：</strong>忠於論文公式，已實證能穩定抓取 30 天關鍵異常。</li>
                <li><strong>限制：</strong>訓練輪數受限於早停點（Epoch 7）。</li>
              </ul>
            </div>
            <div class="card bg-emerald border-emerald">
              <h3 class="c-emerald">方案 B：對抗項加 Clamp（待測）</h3>
              <ul>
                <li><strong>機制：</strong>對 L2 中的 mse(Ô2, W) 設上限 clamp(max=C)。</li>
                <li><strong>優點：</strong>保留對抗放大異常特性，給定下界可訓練更久。</li>
                <li><strong>目的：</strong>觀察充分訓練後對小異常的 Recall 提升。</li>
              </ul>
            </div>
            <div class="card">
              <h3>方案 C：官方開源版（L1-only）</h3>
              <ul>
                <li><strong>機制：</strong>移除負號項 L2，全參數共同最小化 L1。</li>
                <li><strong>優點：</strong>訓練平穩不發散。</li>
                <li><strong>目的：</strong>作為消融對照組，量化對抗懲罰的實際增益。</li>
              </ul>
            </div>
          </div>
        </section>""",

        # 13
        f"""
        <section class="slide">
          <div class="s-head">
            <div><div class="s-tag">12 CONCLUSION &amp; NEXT STEPS</div><h2>總結與向教授請益：從公式落地到完整專題實驗矩陣</h2></div>
            <span class="page-pill">13 / {TOTAL_PAGES}</span>
          </div>
          <div class="grid-2">
            <div class="card border-blue">
              <h3 class="c-blue">claude改進版.py 三大核心交付成果</h3>
              <ul class="big-list">
                <li><strong class="c-blue">1. 論文公式與實作完全對齊</strong><br>透徹理解並落地 PE、Causal Mask、平方 Focus 與雙優化器 Min-Max 對抗訓練，不再被正負抵消 Bug 架空。</li>
                <li><strong class="c-emerald">2. 單日計分精準對齊真實市場</strong><br>改取 Day t 並統一可加性平方和，異常率由 40.0% 降為 2.5%，精準命中 2025 關稅劇震與 2022 CPI。</li>
                <li><strong class="c-coral">3. 可解釋金融風險診斷升級</strong><br>透過誤差佔比與 35% 集中度門檻，具體指導投資人區分「系統性避險」與「板塊調倉」。</li>
              </ul>
            </div>
            <div class="card bg-blue border-navy">
              <h3>向 教授請益與下一步對齊事項</h3>
              <ul class="big-list">
                <li><strong>Q1. 對抗訓練模式的消融比較</strong><br>是否建議補齊「論文公式+早停 vs. Clamp 限幅 vs. 官方 L1-only」三組對比，量化對抗懲罰效益？</li>
                <li><strong>Q2. 門檻與量化評估指標</strong><br>在 P95–P99 敏感度分析外，是否建議實作 POT 極值理論門檻或以歷史大跌日計算 Lead-Time？</li>
                <li><strong>Q3. 同步升級主專案架構</strong><br>確認方向後，即可將這套修正同步套用至 <code>etf_tranad_project/</code> 的 <code>exp1~exp3</code> 並整合 GARCH 與 EWS。</li>
              </ul>
            </div>
          </div>
        </section>"""
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>claude改進版.py 教授說明簡報 (含論文公式剖析)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Microsoft JhengHei", -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0F172A; color: #334155;
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }}
  #stage {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 16px; }}
  .slide {{
    display: none; width: min(1240px, 96vw); height: min(698px, 88vh);
    background: #F8FAFC; border-radius: 14px; padding: 26px 34px 22px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.45); flex-direction: column; overflow: hidden;
  }}
  .slide.active {{ display: flex; }}
  .dark-slide {{ background: #0F172A; color: #F8FAFC; border: 1px solid #1E293B; }}
  .cover-wrap {{ flex: 1; display: flex; flex-direction: column; justify-content: center; padding: 0 18px; }}
  .pill-badge {{
    display: inline-block; width: fit-content; padding: 6px 14px; border-radius: 6px;
    background: #2563EB; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 22px;
  }}
  .cover-h1 {{ font-size: 36px; line-height: 1.3; font-weight: 800; color: #fff; margin-bottom: 14px; }}
  .cover-sub {{ font-size: 18px; color: #38BDF8; margin-bottom: 34px; }}
  .cover-cards {{ margin-bottom: 34px; }}
  .dark-card {{ background: #1E293B; border: 1.5px solid #334155; border-radius: 10px; padding: 16px 18px; }}
  .dark-card h3 {{ color: #38BDF8; font-size: 15px; margin-bottom: 8px; }}
  .dark-card p {{ color: #E2E8F0; font-size: 13px; line-height: 1.55; }}
  .cover-foot {{ color: #94A3B8; font-size: 12.5px; border-top: 1px solid #334155; padding-top: 14px; }}
  .s-head {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; margin-bottom: 14px; }}
  .s-tag {{ font-size: 11px; font-weight: 800; color: #2563EB; letter-spacing: 0.8px; margin-bottom: 4px; }}
  .s-head h2 {{ font-size: 21.5px; color: #1E293B; font-weight: 800; }}
  .page-pill {{ background: #EFF6FF; border: 1px solid #93C5FD; color: #2563EB; font-size: 11px; font-weight: 800; padding: 4px 12px; border-radius: 999px; white-space: nowrap; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; flex: 1; min-height: 0; }}
  .grid-2-asym {{ display: grid; grid-template-columns: 1.38fr 1fr; gap: 16px; flex: 1; min-height: 0; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; flex: 1; min-height: 0; }}
  .kpi-row, .cover-cards {{ flex: 0 0 auto !important; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .card {{ background: #fff; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 13px 15px; display: flex; flex-direction: column; min-height: 0; }}
  .card h3 {{ font-size: 14px; font-weight: 800; color: #1E293B; margin-bottom: 8px; }}
  .card ul {{ list-style: none; font-size: 12.2px; line-height: 1.55; color: #334155; }}
  .card li {{ margin-bottom: 7px; }}
  .big-list li {{ margin-bottom: 16px; font-size: 13.5px; line-height: 1.6; }}
  .step-box {{ border-radius: 8px; padding: 12px; border: 1.5px solid #CBD5E1; }}
  .step-box h4 {{ font-size: 13px; margin-bottom: 6px; color: #0F172A; }}
  .step-box p {{ font-size: 11.8px; line-height: 1.5; color: #334155; }}
  .kpi {{ border-radius: 8px; padding: 10px 14px; border: 1.5px solid; display: flex; flex-direction: column; gap: 4px; }}
  .kpi span {{ font-size: 11.5px; font-weight: 700; color: #475569; }}
  .kpi strong {{ font-size: 13.5px; color: #0F172A; }}
  .fit-img {{ width: 100%; object-fit: contain; display: block; margin: auto; }}
  .eq-img {{ max-width: 100%; max-height: 46px; object-fit: contain; margin: 4px auto 8px; display: block; }}
  .caption {{ font-size: 11.5px; color: #475569; margin-top: 6px; line-height: 1.45; }}
  .clean-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .clean-table th {{ background: #1E293B; color: #fff; text-align: left; padding: 8px 10px; }}
  .clean-table td {{ padding: 8px 10px; border-bottom: 1px solid #E2E8F0; color: #334155; }}
  .clean-table tr:nth-child(even) td {{ background: #F1F5F9; }}
  .clean-table.compact th {{ background: #2563EB; padding: 6px 8px; font-size: 11.5px; }}
  .clean-table.compact td {{ padding: 5.5px 8px; font-size: 11.5px; }}
  .border-blue {{ border-color: #2563EB !important; }}
  .border-emerald {{ border-color: #059669 !important; }}
  .border-coral {{ border-color: #DC2626 !important; }}
  .border-amber {{ border-color: #D97706 !important; }}
  .border-navy {{ border-color: #1E293B !important; }}
  .bg-blue {{ background: #EFF6FF !important; }}
  .bg-emerald {{ background: #ECFDF5 !important; }}
  .bg-coral {{ background: #FEF2F2 !important; }}
  .bg-amber {{ background: #FFFBEB !important; }}
  .c-blue {{ color: #2563EB !important; }}
  .c-emerald {{ color: #059669 !important; }}
  .c-coral {{ color: #DC2626 !important; }}
  .c-amber {{ color: #D97706 !important; }}
  .bold {{ font-weight: 700; }}
  code {{ background: rgba(148,163,184,0.2); padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }}
  #bar {{ height: 48px; background: #090D16; border-top: 1px solid #1E293B; display: flex; align-items: center; justify-content: space-between; padding: 0 28px; color: #94A3B8; font-size: 13px; }}
  button {{ background: #1E293B; color: #F8FAFC; border: 1px solid #334155; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  button:hover {{ background: #2563EB; border-color: #2563EB; }}
</style>
</head>
<body>
  <div id="stage">{''.join(slides_html)}</div>
  <div id="bar">
    <div><code>claude改進版.py</code> 教授說明簡報（支援方向鍵 ← / → 換頁）</div>
    <div style="display:flex; gap:10px; align-items:center;">
      <button onclick="go(-1)">← 上一頁</button>
      <span id="counter">1 / {TOTAL_PAGES}</span>
      <button onclick="go(1)">下一頁 →</button>
    </div>
  </div>
<script>
  const slides = Array.from(document.querySelectorAll('.slide'));
  let cur = 0;
  function show(n) {{
    cur = (n + slides.length) % slides.length;
    slides.forEach((s, i) => s.classList.toggle('active', i === cur));
    document.getElementById('counter').textContent = (cur + 1) + ' / ' + slides.length;
  }}
  function go(d) {{ show(cur + d); }}
  const params = new URLSearchParams(window.location.search);
  const initSlide = parseInt(params.get('slide'), 10);
  if (!isNaN(initSlide) && initSlide >= 1 && initSlide <= slides.length) {{
    show(initSlide - 1);
  }}
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === ' ') go(1);
    if (e.key === 'ArrowLeft') go(-1);
  }});
</script>
</body>
</html>"""

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved HTML -> {HTML_OUT}")


if __name__ == "__main__":
    assets = prepare_assets()
    prs = build_pptx(assets)
    audit_pptx_geometry(prs)
    build_html(assets)
