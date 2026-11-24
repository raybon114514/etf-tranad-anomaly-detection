import os
import matplotlib.pyplot as plt
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Configure Matplotlib for crisp LaTeX / Computer Modern math rendering
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


def render_formula_png(latex_str, out_path, fontsize=22, color="#0F172A", figsize=(6.5, 0.7)):
    fig = plt.figure(figsize=figsize, dpi=600)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, color=color, ha="center", va="center")
    fig.savefig(out_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_path


def set_script(run, mode):
    """Set OpenXML true superscript ('sup') or subscript ('sub') on a PowerPoint run."""
    rPr = run._r.get_or_add_rPr()
    if mode == "sup":
        rPr.set("baseline", "30000")
    elif mode == "sub":
        rPr.set("baseline", "-25000")


def add_runs(p, segments, default_font="Microsoft JhengHei", default_size=11, default_color=RGBColor(51, 65, 85)):
    """
    segments: list of tuples:
      (text, style_dict) or plain str
      style_dict keys: bold, italic, size, color, font, script ('sub'/'sup'), math (True sets Cambria Math + italic)
    """
    for seg in segments:
        if isinstance(seg, str):
            text, style = seg, {}
        else:
            text, style = seg
        r = p.add_run()
        r.text = text
        is_math = style.get("math", False)
        r.font.name = style.get("font", "Cambria Math" if is_math else default_font)
        r.font.size = Pt(style.get("size", default_size))
        r.font.bold = style.get("bold", False)
        r.font.italic = style.get("italic", True if is_math else False)
        r.font.color.rgb = style.get("color", default_color)
        if "script" in style:
            set_script(r, style["script"])


def build_presentation(output_path):
    out_dir = os.path.dirname(output_path)
    eq_dir = os.path.join(out_dir, "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # Pre-render high-res (600 DPI) equations for Slide 5, Slide 7, Slide 8
    eq_files = {
        "s5_focus": render_formula_png(
            r"$\mathrm{Focus} = |O_1 - W|$",
            os.path.join(eq_dir, "s5_focus.png"),
            fontsize=20,
            color="#1E3A8A",
            figsize=(4.5, 0.5),
        ),
        "s5_score": render_formula_png(
            r"$\mathrm{Score}_t = \frac{1}{2}\|O_1 - W\|_2^2 + \frac{1}{2}\|\hat{O}_2 - W\|_2^2$",
            os.path.join(eq_dir, "s5_score.png"),
            fontsize=19,
            color="#1E3A8A",
            figsize=(5.2, 0.6),
        ),
        "s5_loss": render_formula_png(
            r"$\mathcal{L}_1 = w \cdot \mathrm{MSE}(O_1, W) + (1 - w) \cdot \mathrm{MSE}(\hat{O}_2, W)$"
            + "\n"
            + r"$\mathcal{L}_2 = w \cdot \mathrm{MSE}(O_2, W) - (1 - w) \cdot \mathrm{MSE}(\hat{O}_2, W)$",
            os.path.join(eq_dir, "s5_loss.png"),
            fontsize=17,
            color="#0F172A",
            figsize=(5.5, 1.0),
        ),
        "s7_attr": render_formula_png(
            r"$\mathrm{Attr}_i = \left(\hat{O}_{2, T, i} - W_{T, i}\right)^2, \quad \mathrm{Contribution}_i = \frac{\mathrm{Attr}_i}{\sum_{j=1}^{M} \mathrm{Attr}_j} \times 100\%$",
            os.path.join(eq_dir, "s7_attr.png"),
            fontsize=16,
            color="#1E3A8A",
            figsize=(5.6, 0.75),
        ),
        "s8_garch": render_formula_png(
            r"$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$",
            os.path.join(eq_dir, "s8_garch.png"),
            fontsize=20,
            color="#065F46",
            figsize=(5.0, 0.55),
        ),
        "s8_gjr": render_formula_png(
            r"$\sigma_t^2 = \omega + \left(\alpha + \gamma \mathbb{I}_{\{r_{t-1} < 0\}}\right) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$",
            os.path.join(eq_dir, "s8_gjr.png"),
            fontsize=18.5,
            color="#065F46",
            figsize=(5.4, 0.6),
        ),
    }

    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Professional Theme Palette
    COLOR_NAVY_DARK   = RGBColor(15, 23, 42)     # #0F172A Dark background
    COLOR_NAVY_TITLE  = RGBColor(30, 41, 59)     # #1E293B Title dark
    COLOR_BLUE_ACCENT = RGBColor(37, 99, 235)    # #2563EB Royal Blue
    COLOR_BLUE_LIGHT  = RGBColor(239, 246, 255)  # #EFF6FF Tint
    COLOR_CYAN_ACCENT = RGBColor(14, 165, 233)   # #0EA5E9
    COLOR_EMERALD     = RGBColor(16, 185, 129)   # #10B981 Green
    COLOR_EMERALD_BG  = RGBColor(236, 253, 245)  # #ECFDF5 Soft Emerald Tint
    COLOR_CORAL       = RGBColor(239, 68, 68)    # #EF4444 Red/Coral
    COLOR_BG_LIGHT    = RGBColor(248, 250, 252)  # #F8FAFC Slide bg
    COLOR_CARD_BG     = RGBColor(255, 255, 255)  # White Card
    COLOR_CARD_BORDER = RGBColor(226, 232, 240)  # #E2E8F0 Border
    COLOR_TEXT_MAIN   = RGBColor(51, 65, 85)     # #334155 Body text
    COLOR_TEXT_MUTED  = RGBColor(100, 116, 139)  # #64748B Secondary text
    COLOR_WHITE       = RGBColor(255, 255, 255)

    FONT_TITLE = "Microsoft JhengHei"
    FONT_BODY  = "Microsoft JhengHei"

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, tag_text, title_text, category_num=""):
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(8), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_right = tf_tag.margin_top = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"{category_num}  {tag_text}".strip().upper()
        p_tag.font.name = FONT_BODY
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_BLUE_ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.5), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_TITLE
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_NAVY_TITLE

        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.55), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_CARD_BORDER
        line.line.fill.background()

    def create_card(slide, left, top, width, height, title="", title_color=COLOR_NAVY_TITLE, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.45))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = FONT_TITLE
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = title_color
        return card

    def add_formula_box(slide, img_path, left, top, box_w, box_h, img_w, bg_color=COLOR_BLUE_LIGHT, border_color=RGBColor(191, 219, 254)):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, box_w, box_h)
        box.fill.solid()
        box.fill.fore_color.rgb = bg_color
        box.line.color.rgb = border_color
        box.line.width = Pt(1)
        pic = slide.shapes.add_picture(img_path, left, top, width=img_w)
        # Center image inside box
        pic.left = int(left + (box_w - pic.width) / 2)
        pic.top = int(top + (box_h - pic.height) / 2)
        return box, pic

    # =========================================================================
    # SLIDE 1: 封面 (Title Slide - Dark Mode)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_NAVY_DARK)

    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(3.2), Inches(0.4))
    badge.fill.solid()
    badge.fill.fore_color.rgb = COLOR_BLUE_ACCENT
    badge.line.fill.background()
    tf_b = badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "DEEP LEARNING CAPSTONE PROJECT"
    p_b.font.name = FONT_BODY
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_WHITE
    p_b.alignment = PP_ALIGN.CENTER

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(2.0))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "基於 TranAD 之多變量 ETF 時間序列\n異常趨勢偵測與板塊歸因分析"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p2 = tf.add_paragraph()
    p2.text = "雙階段自條件 Transformer 模型實作、無前視切分與市場黑天鵝 XAI 實證進度"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLOR_CYAN_ACCENT
    p2.space_before = Pt(16)

    info_box = s1.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.0))
    tf_info = info_box.text_frame
    p_info = tf_info.paragraphs[0]
    p_info.text = "報告人：專題研究團隊  |  報告類型：期中進度審查報告  |  日期：2026 年 11 月"
    p_info.font.name = FONT_BODY
    p_info.font.size = Pt(13)
    p_info.font.color.rgb = COLOR_TEXT_MUTED

    s1.notes_slide.notes_text_frame.text = (
        "【第 1 頁講稿 - 開場】\n"
        "教授好，各位組員好。今天由我代表我們團隊報告專題的最新進展。\n"
        "我們專題的主題是『基於 TranAD 之多變量 ETF 時間序列異常趨勢偵測與板塊歸因分析』。\n"
        "目前我們已經完成了多檔核心 ETF 的特徵工程、嚴格杜絕前視偏差的資料分割、"
        "TranAD 雙階段對抗神經網路的完整實作，以及在 2022 到 2026 年盲測集上的異常偵測與板塊崩潰歸因分析。"
    )

    # =========================================================================
    # SLIDE 2: 研究背景與核心問題 (Background & Research Questions)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, COLOR_BG_LIGHT)
    add_header(s2, "RESEARCH MOTIVATION & PROBLEM FORMULATION", "研究背景與核心問題定義", "01")

    card_w = Inches(3.64)
    card_h = Inches(5.1)
    gap = Inches(0.4)

    create_card(s2, Inches(0.8), Inches(1.8), card_w, card_h, "1. 金融黑天鵝頻發")
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(2.4), card_w - Inches(0.4), card_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    bullet_texts_1 = [
        ("極端衝擊常態化：", "自 2020 年新冠疫情崩盤、2022 年激進升息引發股債雙殺，到 2023 年矽谷銀行破產與 2024 年科技股板塊劇烈輪動，黑天鵝事件頻率顯著上升。"),
        ("跨板塊外溢效應：", "現代金融市場高度連動，單一產業（如科技股 XLK 或金融股 XLF）的結構性風險常迅速外溢為全市場系統性風險。"),
        ("防禦與避險需求：", "量化基金與投資機構迫切需要能在早期發出『市場慣性破裂』預警的偵測系統。")
    ]
    for i, (b_title, b_desc) in enumerate(bullet_texts_1):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, [("• " + b_title + " ", {"bold": True, "size": 12, "color": COLOR_NAVY_TITLE}), (b_desc, {"size": 12})])
        p.space_after = Pt(12)

    create_card(s2, Inches(0.8) + card_w + gap, Inches(1.8), card_w, card_h, "2. 傳統偵測模型的瓶頸")
    tb = s2.shapes.add_textbox(Inches(1.0) + card_w + gap, Inches(2.4), card_w - Inches(0.4), card_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    bullet_texts_2 = [
        ("單變量統計法侷限：", "傳統 Z-score、布林通道僅能檢驗單一標的，忽略了 ETF 之間的『相對偏離』與共變協同關係。"),
        ("傳統 ML 忽略時序：", "孤立森林（Isolation Forest）與 OCSVM 等方法將時間切片獨立看待，無法建模數週的歷史上下文與時間動態依賴。"),
        ("RNN / LSTM 缺點：", "序列式遞迴運算緩慢，在長歷史視窗下容易發生梯度消失，且難以捕捉多維度之間的非線性交叉注意力。")
    ]
    for i, (b_title, b_desc) in enumerate(bullet_texts_2):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, [("• " + b_title + " ", {"bold": True, "size": 12, "color": COLOR_NAVY_TITLE}), (b_desc, {"size": 12})])
        p.space_after = Pt(12)

    create_card(s2, Inches(0.8) + (card_w + gap) * 2, Inches(1.8), card_w, card_h, "3. 本專題的核心解答", border_color=COLOR_BLUE_ACCENT)
    tb = s2.shapes.add_textbox(Inches(1.0) + (card_w + gap) * 2, Inches(2.4), card_w - Inches(0.4), card_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    bullet_texts_3 = [
        ("架構選用 TranAD：", "採用前沿的雙階段對抗自條件 Transformer，實現高階時間依賴與特徵維度同步建模。"),
        ("點異常 ➔ 趨勢異常：", "不單看單日（Point Anomaly），進一步探討多日持續性（Persistent Anomaly）與市場狀態切換（Regime Shift）。"),
        ("XAI 戰犯歸因分析：", "異常觸發時，透過重構誤差矩陣自動拆解各板塊貢獻度，明確診斷是『大盤系統性風險』還是『特定板塊外溢』。")
    ]
    for i, (b_title, b_desc) in enumerate(bullet_texts_3):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, [("• " + b_title + " ", {"bold": True, "size": 12, "color": COLOR_BLUE_ACCENT}), (b_desc, {"size": 12})])
        p.space_after = Pt(12)

    s2.notes_slide.notes_text_frame.text = (
        "【第 2 頁講稿 - 動機與痛點】\n"
        "首先向教授說明研究動機。近年全球金融市場黑天鵝事件越來越頻繁，從 2020 年疫情崩盤、2022 年通膨升息，到 2024 年科技股大幅回檔。\n"
        "但在傳統量化金融上，常常只依賴單一指數的 Z-score 或布林通道，這有兩個大缺陷：第一，忽略了不同產業板塊間的連動；第二，傳統機器學習模型如孤立森林無法處理時間序列的長程依賴。\n"
        "因此，我們專題跳脫出傳統『只找哪一天是異常點』的框架，利用深度 Transformer 架構結合板塊 ETF，進一步回答：異常是否具有持續趨勢？到底是哪一個產業板塊引發了這次的市場崩潰？"
    )

    # =========================================================================
    # SLIDE 3: 文獻回顧與學術理論依據 (Literature Foundations)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, COLOR_BG_LIGHT)
    add_header(s3, "THEORETICAL FOUNDATION & LITERATURE", "理論基石：核心文獻與學術借鑑", "02")

    top_w = Inches(5.66)
    top_h = Inches(5.1)

    create_card(s3, Inches(0.8), Inches(1.8), top_w, top_h, "核心架構文獻：TranAD (VLDB 2022)", border_color=COLOR_BLUE_ACCENT)
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(2.4), top_w - Inches(0.4), top_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "《TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series》"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_TITLE
    p.space_after = Pt(10)

    tranad_points = [
        ("兩階段自條件機制（Self-Conditioning）：", "傳統自編碼器在面對輕微異常時容易『過度擬合正常』而漏報。TranAD 在 Phase 1 先產生初級重構誤差 Focus Score，並在 Phase 2 反饋給模型進行二次細化，有效放大異常信號。"),
        ("對抗訓練（Adversarial Training）：", "借鑑 GAN 思想，訓練階段透過負損失強制擴大模型在異常偏差區域的敏感度。"),
        ("超高運算效率：", "基於自注意力機制，在 GPU 上可高度平行化運算，推論速度比傳統 LSTM 架構快數十倍。")
    ]
    for b_title, b_desc in tranad_points:
        p = tf.add_paragraph()
        add_runs(p, [("✔ " + b_title + " ", {"bold": True, "size": 12, "color": COLOR_BLUE_ACCENT}), (b_desc, {"size": 12})])
        p.space_after = Pt(10)

    create_card(s3, Inches(6.86), Inches(1.8), top_w, top_h, "領域應用與可解釋性：JRFM 2026 & XAI")
    tb = s3.shapes.add_textbox(Inches(7.06), Inches(2.4), top_w - Inches(0.4), top_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "《Volatility Specification & Anomaly Detection Robustness...》(JRFM 2026)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_TITLE
    p.space_after = Pt(10)

    jrfm_points = [
        ("金融波動率特徵整合：", "該文獻深入探討將 GARCH、EGARCH 等條件波動率作為特徵輸入 Transformer，證明了金融數據厚尾特性對異常邊界設定的關鍵影響。"),
        ("常態邊界閥值設定：", "提出以健康時期的分位數（如 95% 或 98%）作為嚴格容忍門檻，避免單純固定閥值造成的誤判。"),
        ("根因診斷學理（InterFusion KDD 2021 & GDN AAAI 2021）：", "借鑑跨變量相依結構與多維度誤差分解（Root Cause Attribution），使模型具備『指出哪一檔資產出問題』的可解釋性（XAI）。")
    ]
    for b_title, b_desc in jrfm_points:
        p = tf.add_paragraph()
        add_runs(p, [("✔ " + b_title + " ", {"bold": True, "size": 12, "color": COLOR_EMERALD}), (b_desc, {"size": 12})])
        p.space_after = Pt(10)

    s3.notes_slide.notes_text_frame.text = (
        "【第 3 頁講稿 - 理論基礎】\n"
        "在文獻方面，我們專題不是隨意拼裝，而是建立在頂級學術研究的堅實基礎上。\n"
        "左側是我們的核心骨幹模型：發表於資料庫頂會 VLDB 2022 的 TranAD。它提出兩階段自條件回饋與對抗訓練機制，專門解決多變量時序中細微異常容易被自編碼器平滑掉的難題。\n"
        "右側則是領域應用依據：我們參考了 2026 年最新發表於 JRFM 的研究，探討金融波動率 GARCH 規格與 Transformer 的結合，並引入 KDD 2021 InterFusion 的根因分析理論，讓我們的模型具備明確的 XAI 可解釋性。"
    )

    # =========================================================================
    # SLIDE 4: 資料集與特徵工程 (Dataset & Feature Engineering)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, COLOR_BG_LIGHT)
    add_header(s4, "DATA PIPELINE & FEATURE ENGINEERING", "資料集挑選、嚴格防前視切分與特徵建構", "03")

    col_w = Inches(3.64)
    col_h = Inches(5.1)

    # Box 1: 標的資產
    create_card(s4, Inches(0.8), Inches(1.8), col_w, col_h, "1. 多變量資產標的池")
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(2.4), col_w - Inches(0.4), col_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "涵蓋美股全市場與 9 大代表板塊："
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_TITLE
    p.space_after = Pt(8)

    etf_items = [
        "SPY：標普 500 大盤基準",
        "QQQ：那斯達克 100 科技核心",
        "XLK：科技資訊板塊",
        "XLF：金融與銀行板塊",
        "XLE：能源與原油板塊",
        "XLI：工業製造板塊",
        "XLP / XLY：民生消費與可選消費",
        "XLV / XLB：醫療保健與原物料"
    ]
    for item in etf_items:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_after = Pt(4)
    p = tf.add_paragraph()
    p.text = "透過板塊輪動的多維交集，建立完整的市場微結構連動關係。"
    p.font.size = Pt(11)
    p.font.italic = True
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(8)

    # Box 2: 資料切分
    create_card(s4, Inches(0.8) + col_w + gap, Inches(1.8), col_w, col_h, "2. 嚴格杜絕前視偏差切分", border_color=COLOR_CORAL)
    tb = s4.shapes.add_textbox(Inches(1.0) + col_w + gap, Inches(2.4), col_w - Inches(0.4), col_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    splits = [
        ("訓練集 (2010–2018)：", "長達 9 年的純淨學習期。學習牛市與正常回檔下的多資產均衡協方差與重構慣性。"),
        ("驗證集 (2019–2021)：", "3 年驗證期，涵蓋 2020 疫情崩盤的劇烈震盪，專門用於超參數微調。"),
        ("盲測集 (2022–2026)：", "4 年未接觸測試期，包含 2022 升息暴跌、2023 銀行危機與 2024 科技股回檔，完全檢驗實際泛化能力。"),
        ("杜絕資料洩漏原則：", "標準化（Z-score）之 Mean 與 Std 嚴格只採用訓練集計算，測試集絕不提前獲取全域統計量。")
    ]
    for i, (b_title, b_desc) in enumerate(splits):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(
            p,
            [
                ("• " + b_title + " ", {"bold": True, "size": 11, "color": COLOR_CORAL if i == 3 else COLOR_NAVY_TITLE}),
                (b_desc, {"size": 11}),
            ],
        )
        p.space_after = Pt(8)

    # Box 3: 特徵工程 (With true Subscripts & Math Typography)
    create_card(s4, Inches(0.8) + (col_w + gap) * 2, Inches(1.8), col_w, col_h, "3. 特徵構造與時序視窗")
    tb = s4.shapes.add_textbox(Inches(1.0) + (col_w + gap) * 2, Inches(2.4), col_w - Inches(0.4), col_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True

    # Item 1: r_t = ln(P_t / P_{t-1})
    p = tf.paragraphs[0]
    add_runs(
        p,
        [
            ("• 對數日報酬率：", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("r", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" = ln(", {"font": "Cambria Math", "italic": False}),
            ("P", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" / ", {"font": "Cambria Math", "italic": False}),
            ("P", {"math": True}),
            ("t−1", {"math": True, "script": "sub"}),
            (")", {"font": "Cambria Math", "italic": False}),
            ("，消除非平穩價格趨勢，保留純淨收益波動。", {"size": 11}),
        ],
    )
    p.space_after = Pt(10)

    # Item 2: h_t, l_t
    p2 = tf.add_paragraph()
    add_runs(
        p2,
        [
            ("• 日內波動特徵 (", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("h", {"math": True, "bold": True, "color": COLOR_NAVY_TITLE}),
            ("t", {"math": True, "script": "sub", "bold": True, "color": COLOR_NAVY_TITLE}),
            (", ", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("l", {"math": True, "bold": True, "color": COLOR_NAVY_TITLE}),
            ("t", {"math": True, "script": "sub", "bold": True, "color": COLOR_NAVY_TITLE}),
            (")：", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("h", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" = (High − Close)/Close 與 ", {"size": 11}),
            ("l", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" = (Low − Close)/Close，捕捉單日盤中極端震盪幅度。", {"size": 11}),
        ],
    )
    p2.space_after = Pt(10)

    # Item 3: Volume Shock
    p3 = tf.add_paragraph()
    add_runs(
        p3,
        [
            ("• 成交量衝擊 (Volume Shock)：", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("Volume", {"math": True, "italic": False}),
            ("t", {"math": True, "script": "sub"}),
            (" / MA", {"font": "Cambria Math", "italic": False}),
            ("20", {"font": "Cambria Math", "script": "sub"}),
            ("(Volume)，監控異常爆量對定價慣性的衝擊。", {"size": 11}),
        ],
    )
    p3.space_after = Pt(10)

    # Item 4: Sliding Window
    p4 = tf.add_paragraph()
    add_runs(
        p4,
        [
            ("• 滾動滑動視窗 (", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("K = 30", {"font": "Cambria Math", "bold": True, "color": COLOR_NAVY_TITLE}),
            (")：", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("每個輸入樣本包含過去 30 個交易日的歷史時序，捕捉約一個半月的短期記憶與週期性。", {"size": 11}),
        ],
    )

    s4.notes_slide.notes_text_frame.text = (
        "【第 4 頁講稿 - 資料集與特徵】\n"
        "接下來向教授報告我們的資料處置方式。我們選取了美股大盤 SPY、科技 QQQ 以及 9 大板塊 ETF 作為標的池。\n"
        "在量化深度學習中，最容易犯的錯誤就是『前視偏差（Lookahead Bias）』與『資料洩漏』。\n"
        "我們嚴格將時間軸切為三段：2010 到 2018 年為訓練集，學習常態市場結構；2019 到 2021 年為驗證集；2022 到 2026 年則作為純粹的盲測集。\n"
        "而且我們所有的標準化參數（Mean 和 Std），都堅持只用 2010–2018 年的資料計算，絕對不讓未來的統計量提早洩漏給模型。\n"
        "特徵方面除了對數日報酬率，我們還建構了日內極值、爆量倍數以及 30 天的滑動視窗。"
    )

    # =========================================================================
    # SLIDE 5: 模型架構：TranAD 的雙階段對抗機制 (Architecture Details)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, COLOR_BG_LIGHT)
    add_header(s5, "MODEL ARCHITECTURE & TRAINING MECHANISM", "模型核心：TranAD 雙階段對抗自條件神經網路", "04")

    arch_w = Inches(5.66)
    arch_h = Inches(5.1)

    # Left Card: Self-Conditioning with embedded 600 DPI formula boxes
    create_card(s5, Inches(0.8), Inches(1.8), arch_w, arch_h, "雙階段自條件機制（Self-Conditioning）", border_color=COLOR_BLUE_ACCENT)

    # Phase 1 text
    tb_p1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.35), arch_w - Inches(0.4), Inches(0.95))
    tf_p1 = tb_p1.text_frame
    tf_p1.word_wrap = True
    tf_p1.margin_top = tf_p1.margin_bottom = 0
    p = tf_p1.paragraphs[0]
    add_runs(p, [("階段一：粗重構與注意力聚焦 (Phase 1)", {"bold": True, "size": 12, "color": COLOR_BLUE_ACCENT})])
    p_sub = tf_p1.add_paragraph()
    add_runs(
        p_sub,
        [
            ("輸入滑動視窗 ", {"size": 11}),
            ("W", {"math": True}),
            ("，加入位置編碼後經由 Transformer Encoder 產出初階重構 ", {"size": 11}),
            ("O", {"math": True}),
            ("1", {"math": True, "script": "sub"}),
            ("，計算各維度絕對誤差作為聚焦分數：", {"size": 11}),
        ],
    )

    # Phase 1 Equation Pill Box
    add_formula_box(s5, eq_files["s5_focus"], Inches(1.1), Inches(3.22), Inches(5.06), Inches(0.48), Inches(2.5))

    # Phase 2 text
    tb_p2 = s5.shapes.add_textbox(Inches(1.0), Inches(3.82), arch_w - Inches(0.4), Inches(1.15))
    tf_p2 = tb_p2.text_frame
    tf_p2.word_wrap = True
    tf_p2.margin_top = tf_p2.margin_bottom = 0
    p = tf_p2.paragraphs[0]
    add_runs(p, [("階段二：因果遮罩與細化重構 (Phase 2)", {"bold": True, "size": 12, "color": COLOR_BLUE_ACCENT})])
    p_sub = tf_p2.add_paragraph()
    add_runs(
        p_sub,
        [
            ("輸入原始視窗 ", {"size": 11}),
            ("W", {"math": True}),
            (" 進入解碼端，施加因果遮罩（Causal Mask）防止看到未來時間戳。透過交叉注意力將 Phase 1 的 Focus Score 融合回特徵流，專注放大偏離正常的微小異常點，輸出 ", {"size": 11}),
            ("Ô", {"math": True}),
            ("2", {"math": True, "script": "sub"}),
            ("。", {"size": 11}),
        ],
    )

    # Anomaly Score Header + Pill Box + Caption
    tb_p3 = s5.shapes.add_textbox(Inches(1.0), Inches(5.05), arch_w - Inches(0.4), Inches(0.32))
    tf_p3 = tb_p3.text_frame
    tf_p3.margin_top = tf_p3.margin_bottom = 0
    add_runs(tf_p3.paragraphs[0], [("推論判定公式 (Anomaly Score)", {"bold": True, "size": 12, "color": COLOR_BLUE_ACCENT})])

    add_formula_box(s5, eq_files["s5_score"], Inches(1.1), Inches(5.40), Inches(5.06), Inches(0.55), Inches(4.5))

    tb_p3_cap = s5.shapes.add_textbox(Inches(1.0), Inches(6.02), arch_w - Inches(0.4), Inches(0.6))
    tf_p3_cap = tb_p3_cap.text_frame
    tf_p3_cap.word_wrap = True
    tf_p3_cap.margin_top = 0
    add_runs(tf_p3_cap.paragraphs[0], [("若某日的綜合重構誤差顯著放大，即代表市場行為脫離了模型學會的常態結構。", {"size": 11})])

    # Right Card: Adversarial Loss & Code Implementation
    create_card(s5, Inches(6.86), Inches(1.8), arch_w, arch_h, "對抗式損失函數與程式實作重點")

    tb_r1 = s5.shapes.add_textbox(Inches(7.06), Inches(2.35), arch_w - Inches(0.4), Inches(0.32))
    tf_r1 = tb_r1.text_frame
    tf_r1.margin_top = tf_r1.margin_bottom = 0
    add_runs(tf_r1.paragraphs[0], [("• 對抗性損失公式設計：", {"bold": True, "size": 12, "color": COLOR_NAVY_TITLE})])

    add_formula_box(s5, eq_files["s5_loss"], Inches(7.16), Inches(2.70), Inches(5.06), Inches(1.05), Inches(4.85))

    tb_r2 = s5.shapes.add_textbox(Inches(7.06), Inches(3.85), arch_w - Inches(0.4), Inches(2.85))
    tf_r2 = tb_r2.text_frame
    tf_r2.word_wrap = True
    tf_r2.margin_top = 0

    p_w = tf_r2.paragraphs[0]
    add_runs(
        p_w,
        [
            ("權重因子 ", {"size": 11}),
            ("w", {"math": True}),
            (" = 1 / Epoch", {"font": "Cambria Math", "size": 11}),
            (" 隨訓練次數動態衰減。", {"size": 11}),
        ],
    )
    p_w.space_after = Pt(10)

    p_m_title = tf_r2.add_paragraph()
    add_runs(p_m_title, [("• 對抗懲罰的作用機制：", {"bold": True, "size": 12, "color": COLOR_NAVY_TITLE})])
    p_m_title.space_after = Pt(3)

    p_m_desc = tf_r2.add_paragraph()
    add_runs(
        p_m_desc,
        [
            ("在訓練初期 ", {"size": 11}),
            ("w", {"math": True}),
            (" 較大，模型快速學習常態重構；隨訓練推進，負號項 ", {"size": 11}),
            ("−(1 − ", {"font": "Cambria Math", "size": 11}),
            ("w", {"math": True}),
            (") · MSE(", {"font": "Cambria Math", "size": 11}),
            ("Ô", {"math": True}),
            ("2", {"math": True, "script": "sub"}),
            (", ", {"font": "Cambria Math", "size": 11}),
            ("W", {"math": True}),
            (")", {"font": "Cambria Math", "size": 11}),
            (" 開始發力，強制拉開正常信號與微小異常點的邊界，避免自編碼器將異常點『自動合理化修復』。", {"size": 11}),
        ],
    )
    p_m_desc.space_after = Pt(10)

    p_c_title = tf_r2.add_paragraph()
    add_runs(p_c_title, [("• 程式碼落地狀態：", {"bold": True, "size": 12, "color": COLOR_NAVY_TITLE})])
    p_c_title.space_after = Pt(3)

    p_c_desc = tf_r2.add_paragraph()
    add_runs(
        p_c_desc,
        [("已於 `firstry/etf_trend_anomaly.py` 與 `firstry/vsd.py` 完整實作 PyTorch 模組，包含自訂 Causal Mask 與多頭注意力層。", {"size": 11})],
    )

    s5.notes_slide.notes_text_frame.text = (
        "【第 5 頁講稿 - 模型細節】\n"
        "這一頁是我們實作的核心技術：TranAD 雙階段對抗自條件神經網路。\n"
        "傳統的 Autoencoder 最容易發生的問題，就是連異常點都被它『順手修復成正常』，導致漏報。\n"
        "TranAD 巧妙地設計了兩階段：第一階段算出的重構誤差向量被當成 Focus Score，在第二階段透過 Cross-Attention 強制反饋給解碼器。\n"
        "同時在右側的 Loss 設計上，隨 Epoch 衰減的對抗懲罰項，會故意擴大微小異常與正常的距離。目前這套架構已在我們的 PyTorch 程式碼中完整落地運行。"
    )

    # =========================================================================
    # SLIDE 6: 實驗成果與門檻校正 (Results & Threshold Calibration)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, COLOR_BG_LIGHT)
    add_header(s6, "EMPIRICAL EXPERIMENTS & DETECTION METRICS", "實證結果：常態門檻錨定與盲測集異常觸發", "05")

    m_w = Inches(3.64)
    m_h = Inches(1.2)

    create_card(s6, Inches(0.8), Inches(1.8), m_w, m_h, bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(1.85), m_w - Inches(0.4), m_h)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "745 交易日"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT
    p2 = tf.add_paragraph()
    p2.text = "測試集盲測跨度（2022–2026）"
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_TEXT_MUTED

    create_card(s6, Inches(0.8) + m_w + gap, Inches(1.8), m_w, m_h, bg_color=RGBColor(254, 242, 242), border_color=COLOR_CORAL)
    tb = s6.shapes.add_textbox(Inches(1.0) + m_w + gap, Inches(1.85), m_w - Inches(0.4), m_h)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "6 次重大警示"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_CORAL
    p2 = tf.add_paragraph()
    p2.text = "嚴格常態門檻（98th Percentile）"
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_TEXT_MUTED

    create_card(s6, Inches(0.8) + (m_w + gap) * 2, Inches(1.8), m_w, m_h, bg_color=COLOR_EMERALD_BG, border_color=COLOR_EMERALD)
    tb = s6.shapes.add_textbox(Inches(1.0) + (m_w + gap) * 2, Inches(1.85), m_w - Inches(0.4), m_h)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "9 檔板塊連動"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p2 = tf.add_paragraph()
    p2.text = "多變量特徵跨資產協同監控"
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_TEXT_MUTED

    b_w = Inches(5.66)
    b_h = Inches(3.6)

    create_card(s6, Inches(0.8), Inches(3.3), b_w, b_h, "關鍵修正：常態門檻錨定機制")
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(3.9), b_w - Inches(0.4), b_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    fix_points = [
        ("修正前盲點：", "原先若直接拿包含 2020 年疫情劇烈崩盤的驗證集來切分閾值，會導致門檻被極端值嚴重拉高，使 2022 年以後許多嚴重的結構性異常全部漏報。"),
        ("修正後策略：", "改用『純淨訓練集 (2010–2018)』的分數分佈來定義常態市場的容忍邊界，取訓練集 98 百分位數作為嚴格門檻。"),
        ("效果：", "門檻標準化穩定，成功過濾掉日常雜訊，僅在市場結構真正失衡時亮起紅燈。")
    ]
    for i, (b_title, b_desc) in enumerate(fix_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, [("• " + b_title + " ", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}), (b_desc, {"size": 11})])
        p.space_after = Pt(8)

    create_card(s6, Inches(6.86), Inches(3.3), b_w, b_h, "實測捕捉到的重大異常事件清單")
    tb = s6.shapes.add_textbox(Inches(7.06), Inches(3.9), b_w - Inches(0.4), b_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    anom_list = [
        ("2023-11-14：", [("CPI 通膨數據降溫引發全市場暴力軋空與科技股急漲脫鉤。", {"size": 11})]),
        ("2024-11-06：", [("美國大選後政策預期重塑，板塊間出現極端資金大輪動。", {"size": 11})]),
        (
            "2025-03-14 / 2025-04-09：",
            [
                ("春季美債殖利率跳升引發防禦性板塊與成長股劇烈背離（Score 飆破 ", {"size": 11}),
                ("1.269 × 10", {"font": "Cambria Math", "size": 11}),
                ("8", {"font": "Cambria Math", "size": 11, "script": "sup"}),
                ("）。", {"size": 11}),
            ],
        ),
        ("2025-05-12：", [("科技板塊動能回落引發之連鎖定價異常。", {"size": 11})]),
        ("實證結論：", [("模型觸發之異常日均與真實金融市場重大宏觀衝擊點精準吻合。", {"size": 11})]),
    ]
    for i, (b_title, b_segs) in enumerate(anom_list):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, [("🚩 " + b_title + " ", {"bold": True, "size": 11, "color": COLOR_CORAL if i < 4 else COLOR_EMERALD})] + b_segs)
        p.space_after = Pt(6)

    s6.notes_slide.notes_text_frame.text = (
        "【第 6 頁講稿 - 實證成果】\n"
        "在實證結果部分，我們在 2022 到 2026 年的 745 個盲測交易日中，進行了嚴格檢驗。\n"
        "這裡要向教授特別報告我們做的一個『關鍵架構修正』：一開始如果用驗證集的極值當門檻，2020 年疫情崩盤會把門檻拉得太高，導致後續異常全都抓不到。\n"
        "因此我們修正為：以 2010–2018 年健康常態市場的第 98 百分位數作為嚴格容忍邊界。\n"
        "修正後，模型在測試集成功抓出 6 次關鍵重大異常日，包括 2023 年底的通膨轉折、2024 年底大選後的板塊極端輪動，以及 2025 年春季的多波極端震盪，完全對應到了真實世界的金融衝擊。"
    )

    # =========================================================================
    # SLIDE 7: 專題核心亮點：板塊異常歸因分析 (XAI)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, COLOR_BG_LIGHT)
    add_header(s7, "CORE INNOVATION & EXPLAINABILITY (XAI)", "專題核心亮點：板塊異常歸因分析（找戰犯）", "06")

    card_w = Inches(5.66)
    card_h = Inches(5.1)

    # Left: Methodology with embedded 600 DPI formula box
    create_card(s7, Inches(0.8), Inches(1.8), card_w, card_h, "歸因分析機制：特徵層級重構誤差分解", border_color=COLOR_BLUE_ACCENT)

    tb_x1 = s7.shapes.add_textbox(Inches(1.0), Inches(2.35), card_w - Inches(0.4), Inches(1.65))
    tf_x1 = tb_x1.text_frame
    tf_x1.word_wrap = True
    tf_x1.margin_top = 0

    p0 = tf_x1.paragraphs[0]
    add_runs(
        p0,
        [
            ("• 傳統黑盒子痛點：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("一般深度學習只給出單一純量 Anomaly Score，投資人只知道『今天市場異常』，卻不知道『為什麼異常』與『誰引發異常』。", {"size": 11}),
        ],
    )
    p0.space_after = Pt(8)

    p1 = tf_x1.add_paragraph()
    add_runs(
        p1,
        [
            ("• 重構誤差分解算式：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("在異常觸發日，擷取最後一個時間戳 (", {"size": 11}),
            ("t = T", {"math": True}),
            (") 各資產的重構誤差與百分比貢獻度：", {"size": 11}),
        ],
    )

    # Formula Pill Box for Attr_i and Contribution_i
    add_formula_box(s7, eq_files["s7_attr"], Inches(1.05), Inches(4.02), Inches(5.16), Inches(0.82), Inches(4.95))

    tb_x2 = s7.shapes.add_textbox(Inches(1.0), Inches(4.98), card_w - Inches(0.4), Inches(1.75))
    tf_x2 = tb_x2.text_frame
    tf_x2.word_wrap = True
    tf_x2.margin_top = 0
    p2 = tf_x2.paragraphs[0]
    add_runs(
        p2,
        [
            ("• 雙層風險診斷決策樹：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("• 情況 A：若大盤 SPY 貢獻度最高 ➔ 診斷為【宏觀系統性風險】，市場全面流動性緊縮。\n", {"size": 11}),
            ("• 情況 B：若特定板塊（如 XLK 科技或 XLF 金融）居首 ➔ 診斷為【板塊結構性破裂】，衝擊由該產業向外擴散。", {"size": 11}),
        ],
    )

    # Right: Real Ranking from CSV (with true superscript 10^8)
    create_card(s7, Inches(6.86), Inches(1.8), card_w, card_h, "實測 9 檔 ETF 平均異常分數與敏感度排名")
    tb = s7.shapes.add_textbox(Inches(7.06), Inches(2.4), card_w - Inches(0.4), card_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "根據盲測集（2022–2026）實測 CSV 導出之板塊敏感度："
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY_TITLE
    p.space_after = Pt(8)

    rankings = [
        ("1. XLK (科技資訊)", "1.307", "【最高戰犯】科技股高估值，對利率與極端情緒最敏感"),
        ("2. XLP (必需消費)", "1.306", "【避險異動】資金恐慌撤出或湧入防禦資產的背離信號"),
        ("3. SPY (大盤基準)", "1.301", "【系統性脫鉤】全市場定價慣性瓦解"),
        ("4. QQQ (那斯達克)", "1.289", "科技大型權值股波動連動"),
        ("5. XLY (可選消費)", "1.274", "景氣循環消費信心波動"),
        ("6. XLU (公用事業)", "1.258", "防禦資產相對穩定"),
        ("7. XLE (能源原油)", "1.237", "受地緣政治獨立驅動"),
        ("8. XLB / XLF", "1.222", "金融與原物料於本次測試期相對中性")
    ]
    for r_name, r_mantissa, r_desc in rankings:
        p = tf.add_paragraph()
        name_color = COLOR_CORAL if "XLK" in r_name else (COLOR_BLUE_ACCENT if "SPY" in r_name else COLOR_NAVY_TITLE)
        add_runs(
            p,
            [
                (r_name + " : ", {"bold": True, "size": 10.5, "color": name_color}),
                (f"{r_mantissa} × 10", {"font": "Cambria Math", "size": 10.5}),
                ("8", {"font": "Cambria Math", "size": 10.5, "script": "sup"}),
                (f"  — {r_desc}", {"size": 10}),
            ],
        )
        p.space_after = Pt(4)

    s7.notes_slide.notes_text_frame.text = (
        "【第 7 頁講稿 - 歸因 XAI 亮點】\n"
        "這一頁是我們專題最受矚目的創新亮點：『板塊異常歸因分析』，俗稱『找戰犯』。\n"
        "過去深度學習常被詬病是黑盒子，只告訴你今天有問題，但講不出原因。我們透過拆解 Transformer 的重構誤差矩陣，精確算出每個板塊對總異常的貢獻百分比。\n"
        "在右邊的盲測數據中，科技板塊 XLK 奪得平均異常分數第一名，證實了近幾年科技股的高估值是引發市場動盪的最大震源；而必需消費 XLP 緊追在後，反映了資金避險造成的反向脫鉤。\n"
        "透過這套機制，我們能清楚判斷市場究竟是『全體大盤系統性崩盤』，還是『特定板塊失衡引發的骨牌外溢』。"
    )

    # =========================================================================
    # SLIDE 8: 延伸實驗：GARCH 波動度模組 (vsd.py)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, COLOR_BG_LIGHT)
    add_header(s8, "ECONOMETRIC EXTENSION & VOLATILITY SPECIFICATION", "前沿複現與延伸：GARCH 條件波動率對接實驗", "07")

    card_w = Inches(5.66)
    card_h = Inches(5.1)

    create_card(s8, Inches(0.8), Inches(1.8), card_w, card_h, "計量經濟學特徵從零實作 (vsd.py)", border_color=COLOR_EMERALD)

    # Item 1 & Item 2 header
    tb_g1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.35), card_w - Inches(0.4), Inches(1.18))
    tf_g1 = tb_g1.text_frame
    tf_g1.word_wrap = True
    tf_g1.margin_top = 0
    p0 = tf_g1.paragraphs[0]
    add_runs(
        p0,
        [
            ("• 文獻複現 (JRFM 2026)：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("針對 S&P 500 歷史長時序，從零自主實作 QMLE（準最大概似法）參數估計。", {"size": 11}),
        ],
    )
    p0.space_after = Pt(6)

    p1 = tf_g1.add_paragraph()
    add_runs(
        p1,
        [
            ("• GARCH(1,1) 方差方程：", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("捕捉金融時序著名的『波動度聚集』：", {"size": 11}),
        ],
    )

    # Formula Pill Box 1: GARCH(1,1)
    add_formula_box(
        s8,
        eq_files["s8_garch"],
        Inches(1.1),
        Inches(3.48),
        Inches(5.06),
        Inches(0.55),
        Inches(3.9),
        bg_color=COLOR_EMERALD_BG,
        border_color=RGBColor(167, 243, 208),
    )

    # Item 3 header: GJR-GARCH(1,1)
    tb_g2 = s8.shapes.add_textbox(Inches(1.0), Inches(4.12), card_w - Inches(0.4), Inches(0.75))
    tf_g2 = tb_g2.text_frame
    tf_g2.word_wrap = True
    tf_g2.margin_top = 0
    p2 = tf_g2.paragraphs[0]
    add_runs(
        p2,
        [
            ("• GJR-GARCH(1,1) 非對稱槓桿：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("加入負向衝擊指示函數，捕捉『利空下跌時波動度劇烈大於利多上漲』之金融槓桿效應：", {"size": 11}),
        ],
    )

    # Formula Pill Box 2: GJR-GARCH(1,1)
    add_formula_box(
        s8,
        eq_files["s8_gjr"],
        Inches(1.1),
        Inches(4.92),
        Inches(5.06),
        Inches(0.58),
        Inches(4.75),
        bg_color=COLOR_EMERALD_BG,
        border_color=RGBColor(167, 243, 208),
    )

    # Item 4: Garman-Klass h_t, l_t
    tb_g3 = s8.shapes.add_textbox(Inches(1.0), Inches(5.60), card_w - Inches(0.4), Inches(1.1))
    tf_g3 = tb_g3.text_frame
    tf_g3.word_wrap = True
    tf_g3.margin_top = 0
    p3 = tf_g3.paragraphs[0]
    add_runs(
        p3,
        [
            ("• 日內波動極值特徵 (Garman-Klass 簡化)：\n", {"bold": True, "size": 11, "color": COLOR_NAVY_TITLE}),
            ("引入日內偏離 ", {"size": 11}),
            ("h", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" = (High − Close)/Close 與 ", {"size": 11}),
            ("l", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (" = (Low − Close)/Close 作為額外特徵矩陣。", {"size": 11}),
        ],
    )

    # Right Card: Transformer Integration Findings (with proper σ_t)
    create_card(s8, Inches(6.86), Inches(1.8), card_w, card_h, "與 Transformer 結合之初步發現")
    tb = s8.shapes.add_textbox(Inches(7.06), Inches(2.4), card_w - Inches(0.4), card_h - Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p_f1 = tf.paragraphs[0]
    add_runs(
        p_f1,
        [
            ("✔ 單純 Return 輸入的盲點：\n", {"bold": True, "size": 11, "color": COLOR_EMERALD}),
            ("僅輸入日報酬率時，模型容易將高波動期內的正常漲跌震盪誤判為結構性異常，雜訊較多。", {"size": 11}),
        ],
    )
    p_f1.space_after = Pt(12)

    p_f2 = tf.add_paragraph()
    add_runs(
        p_f2,
        [
            ("✔ 條件波動率 ", {"bold": True, "size": 11, "color": COLOR_EMERALD}),
            ("σ", {"math": True, "bold": True, "size": 11, "color": COLOR_EMERALD}),
            ("t", {"math": True, "bold": True, "size": 11, "script": "sub", "color": COLOR_EMERALD}),
            (" 的調節效果：\n", {"bold": True, "size": 11, "color": COLOR_EMERALD}),
            ("當模型輸入 [Return", {"size": 11}),
            ("t", {"math": True, "script": "sub"}),
            (", ", {"size": 11}),
            ("σ", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (", ", {"size": 11}),
            ("h", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            (", ", {"size": 11}),
            ("l", {"math": True}),
            ("t", {"math": True, "script": "sub"}),
            ("] 時，Transformer 能夠『理解當前環境處於高波動週期』，進而減少一般性震盪的虛警，提升真正厚尾黑天鵝的捕捉精度。", {"size": 11}),
        ],
    )
    p_f2.space_after = Pt(12)

    p_f3 = tf.add_paragraph()
    add_runs(
        p_f3,
        [
            ("✔ 消融實驗（Ablation）規劃：\n", {"bold": True, "size": 11, "color": COLOR_EMERALD}),
            ("正在對比不同波動度規格（GARCH vs EGARCH vs GJR-GARCH）對多 ETF 異常召回率（Recall）與誤報率（FPR）的影響。", {"size": 11}),
        ],
    )

    s8.notes_slide.notes_text_frame.text = (
        "【第 8 頁講稿 - GARCH 模組】\n"
        "除了 TranAD 本身，我們也複現並延伸了最新文獻中的計量金融模組，實作在 `vsd.py` 中。\n"
        "金融市場有一個關鍵特性叫做『波動度聚集』與『槓桿效應』，也就是壞消息引發的震盪遠高於好消息。\n"
        "我們從零用 Scipy QMLE 實作了 GARCH 與 GJR-GARCH 估計器。\n"
        "初步實驗發現：當把條件波動率 sigma 輸入給 Transformer 後，模型能夠識別出整體市場背景的高波動狀態，大幅降低了在震盪市中的誤判率。這是我們後續要深入探討的重要特徵消融方向。"
    )

    # =========================================================================
    # SLIDE 9: 已克服之工程挑戰與應對策略 (Challenges & Solutions)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, COLOR_BG_LIGHT)
    add_header(s9, "ENGINEERING CHALLENGES & METHODOLOGY FIXES", "遭遇挑戰、工程除錯與關鍵解決方案", "08")

    col_w = Inches(3.64)
    col_h = Inches(5.1)

    challenges = [
        ("挑戰 1：極端事件扭曲門檻", 
         COLOR_CORAL,
         "【遭遇問題】：\n2020 疫情崩盤的數值是常態的 5 倍以上，若驗證集包含該時期，會把閥值推到極高，導致後續所有回檔都被判定為『正常』。\n\n"
         "【解決方案】：\n建立『健康基準常態錨定法』，採用 2010–2018 年長期無重大惡性通膨的訓練期分佈計算 98% 門檻，確保邊界符合市場日常慣性。"),
        ("挑戰 2：多變量維度對齊與防前視", 
         COLOR_BLUE_ACCENT,
         "【遭遇問題】：\n多檔 ETF 開市交易日偶有細微缺漏，且滑動視窗構造時若未對齊容易產生前視偏差。\n\n"
         "【解決方案】：\n實作以日期為索引的嚴格交集對齊（Inner Join），標準化參數完全隔離於訓練集，且在解碼端施加 Causal Mask 保證時間不可逆。"),
        ("挑戰 3：黑盒子模型缺乏可信度", 
         COLOR_EMERALD,
         "【遭遇問題】：\n單一 Anomaly Score 無法向金融從業者解釋異常的本質與具體板塊風險來源。\n\n"
         "【解決方案】：\n推導出特徵維度重構誤差分解矩陣，將高維 Loss 投影回 9 檔 ETF 的相對百分比，成功實現板塊層級的 XAI 可解釋性診斷。")
    ]
    for i, (c_title, c_color, c_text) in enumerate(challenges):
        left_pos = Inches(0.8) + i * (col_w + gap)
        create_card(s9, left_pos, Inches(1.8), col_w, col_h, c_title, title_color=c_color, border_color=c_color)
        tb = s9.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.4), col_w - Inches(0.4), col_h - Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_text
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN

    s9.notes_slide.notes_text_frame.text = (
        "【第 9 頁講稿 - 挑戰與克服】\n"
        "在專題推進過程中，我們遭遇並克服了三個核心難題：\n"
        "第一是『極端值拉偏門檻』，我們透過健康常態期錨定法成功解決；\n"
        "第二是『多變量對齊與防前視』，我們建立了標準化的無母數前處理管線；\n"
        "第三是『模型黑盒子問題』，我們成功打通了特徵誤差分解，讓輸出具備實際的金融意義。這些除錯經驗也讓我們的系統架構更加成熟穩定。"
    )

    # =========================================================================
    # SLIDE 10: 下一步工作規劃與里程碑 (Next Steps & Milestones)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, COLOR_BG_LIGHT)
    add_header(s10, "FUTURE WORK & PROJECT ROADMAP", "下一步規劃：深化方向與專題預期產出", "09")

    box_w = Inches(11.733)
    step_h = Inches(1.05)
    step_gap = Inches(0.2)

    milestones = [
        ("Phase 1（近期 1–2 週）：特徵工程消融實驗 (Ablation Study)",
         "將 `vsd.py` 的 GARCH 條件波動率與日內極值完整整合至 9 檔 ETF 的 TranAD 主程式中，系統化比較純報酬率 vs 加入波動率後的 F1-score 與預警提早度。",
         COLOR_BLUE_ACCENT),
        ("Phase 2（中期 3–4 週）：從點異常邁向『持續性異常事件』與狀態偵測",
         "建立 Consecutive Anomaly 連續警示機制，定義『異常事件持續天數』與『跨 ETF 同步異常指數』，區分單日震盪與長期體制轉換（Regime Shift）。",
         COLOR_CYAN_ACCENT),
        ("Phase 3（後期 5–6 週）：傳統 Baseline 比較與 2024 時序大模型對照",
         "實作 TCN 與 Isolation Forest 作為比較基準；並探索能否引入 ICML 2024 的 MOMENT 或 NeurIPS 2024 UniTS 時序基底模型進行少樣本遷移對比。",
         COLOR_EMERALD),
        ("Phase 4（期末收網）：視覺化戰情看板與書面專題報告",
         "利用 Matplotlib 與 Plotly 開發互動式 ETF 異常趨勢儀表板，並彙整所有實驗數據與 References 完成正式專題論文。",
         COLOR_NAVY_TITLE)
    ]

    for i, (m_title, m_desc, m_color) in enumerate(milestones):
        top_pos = Inches(1.8) + i * (step_h + step_gap)
        create_card(s10, Inches(0.8), top_pos, box_w, step_h, border_color=m_color)
        tb = s10.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.12), box_w - Inches(0.4), step_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = m_title
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = m_color
        p2 = tf.add_paragraph()
        p2.text = m_desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(3)

    s10.notes_slide.notes_text_frame.text = (
        "【第 10 頁講稿 - 下一步規劃】\n"
        "最後向教授報告我們接下來的研發排程：\n"
        "第一階段，我們將在兩週內完成 GARCH 波動率與主模型的完整整合，進行特徵消融實驗；\n"
        "第二階段，我們將正式把單日異常擴充為『持續性異常事件與跨板塊同步指數』；\n"
        "第三階段，完成與傳統 TCN 及孤立森林的基準比較，並引用 2024 年 ICML MOMENT 大模型作為展望；\n"
        "第四階段，完成互動式視覺化儀表板並撰寫完整的專題論文。"
    )

    # =========================================================================
    # SLIDE 11: 結尾致謝與 Q&A (Conclusion - Dark Mode)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, COLOR_NAVY_DARK)

    badge = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.66), Inches(1.8), Inches(4.0), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = COLOR_BLUE_ACCENT
    badge.line.fill.background()
    tf_b = badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "PROGRESS REPORT CONCLUSION"
    p_b.font.name = FONT_BODY
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_WHITE
    p_b.alignment = PP_ALIGN.CENTER

    tb = s11.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11.333), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "感謝教授的聆聽與指導"
    p.font.name = FONT_TITLE
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "懇請 教授指正與討論 (Q & A)"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(22)
    p2.font.color.rgb = COLOR_CYAN_ACCENT
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    create_card(s11, Inches(2.66), Inches(4.8), Inches(8.0), Inches(1.6), bg_color=RGBColor(30, 41, 59), border_color=COLOR_BLUE_ACCENT)
    tb_sum = s11.shapes.add_textbox(Inches(2.86), Inches(4.95), Inches(7.6), Inches(1.3))
    tf_s = tb_sum.text_frame
    tf_s.word_wrap = True
    p_s = tf_s.paragraphs[0]
    p_s.text = "目前核心進度總結："
    p_s.font.size = Pt(12)
    p_s.font.bold = True
    p_s.font.color.rgb = COLOR_WHITE
    p_s.space_after = Pt(4)
    summary_bullets = [
        "✔ 跨 ETF 資料管道與嚴格無前視切分建置完成",
        "✔ TranAD 雙階段對抗機制與門檻校正完成驗證",
        "✔ XAI 板塊崩潰歸因分析打通，實測成果高度吻合金融宏觀事件"
    ]
    for b in summary_bullets:
        p_sub = tf_s.add_paragraph()
        p_sub.text = b
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = RGBColor(203, 213, 225)
        p_sub.space_after = Pt(2)

    s11.notes_slide.notes_text_frame.text = (
        "【第 11 頁講稿 - 結尾】\n"
        "以上是我們團隊目前的專題進度報告。\n"
        "我們已經確立了模型骨幹、完成了資料驗證與歸因實作，接下來會全力推進特徵消融與趨勢事件量化。\n"
        "謝謝教授，請教授指導！"
    )

    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")


if __name__ == "__main__":
    out_dir = r"c:\Users\user\Desktop\dl\報告進度"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "專題進度報告_ETF異常趨勢偵測.pptx")
    build_presentation(out_file)
