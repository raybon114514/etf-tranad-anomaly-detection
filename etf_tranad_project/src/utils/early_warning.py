import numpy as np
import pandas as pd


class EarlyWarningSystem:
    """
    ETF 跨板塊異常早期預警系統 (Early Warning System, EWS)
    
    解決傳統二元門檻 (Binary Threshold) 後知後覺的問題，
    透過「雙重動態門檻 + 分數速度/加速度 (Velocity & Acceleration) + 板塊集中度」
    在市場結構剛出現不對勁的「起頭潛伏期」提早發出警示。
    """

    LEVEL_NORMAL = "NORMAL"          # 🟢 綠燈: 常態穩定
    LEVEL_WATCH = "ONSET_WATCH"      # 🟡 黃燈: 不對勁起頭 (潛伏期異動)
    LEVEL_ELEVATED = "ELEVATED_RISK" # 🟠 橙燈: 結構裂解加速
    LEVEL_CRITICAL = "CRITICAL"      # 🔴 紅燈: 嚴重失真 / 全面爆發

    def __init__(self, train_scores, watch_pct=90.0, critical_pct=98.0):
        """
        以純淨訓練集 (2010-2018) 建立無前視偏差的基準門檻
        
        Args:
            train_scores: 訓練集樣本之異常分數陣列 (純量 MSE)
            watch_pct: 起頭警戒百分位 (預設 90th)
            critical_pct: 臨界爆發百分位 (預設 98th)
        """
        self.t_watch = float(np.percentile(train_scores, watch_pct))
        self.t_crit = float(np.percentile(train_scores, critical_pct))
        self.score_diff_std = float(np.std(np.diff(train_scores)) + 1e-8)

    def evaluate_sequence(self, dates, scores, attributions=None, tickers=None):
        """
        對測試集時序進行逐日早期預警評估
        
        Args:
            dates: DatetimeIndex 或 日期陣列
            scores: 每日異常分數 (一維 ndarray)
            attributions: 每日各資產重構誤差 (二維 ndarray, shape: [N, feats])
            tickers: ETF 代碼清單
            
        Returns:
            pd.DataFrame: 包含每日預警等級、速度、加速度與診斷訊息的資料表
        """
        n = len(scores)
        levels = []
        velocities = []      # 一階差分 (當日變化率)
        accelerations = []   # 二階差分 (變化率之加速度)
        reasons = []
        lead_signals = []    # 是否為起頭提示 (黃燈或剛由綠轉黃)
        max_contributors = []
        max_contrib_pcts = []

        for i in range(n):
            s_t = scores[i]
            
            # 計算一階速度與二階加速度
            if i >= 1:
                v_t = scores[i] - scores[i - 1]
            else:
                v_t = 0.0

            if i >= 2:
                v_prev = scores[i - 1] - scores[i - 2]
                a_t = v_t - v_prev
            else:
                a_t = 0.0

            velocities.append(v_t)
            accelerations.append(a_t)

            # 計算板塊集中度 (若有提供 attribution)
            top_ticker = "N/A"
            top_pct = 0.0
            if attributions is not None and tickers is not None:
                attr_t = attributions[i]
                tot = np.sum(attr_t) + 1e-8
                pcts = (attr_t / tot) * 100
                top_idx = int(np.argmax(pcts))
                top_ticker = tickers[top_idx]
                top_pct = float(pcts[top_idx])
            
            max_contributors.append(top_ticker)
            max_contrib_pcts.append(top_pct)

            # ----------------------------------------------------
            # 狀態燈號決策邏輯 (Decision Tree)
            # ----------------------------------------------------
            # 1. 嚴重全面爆發 (紅燈)
            if s_t >= self.t_crit:
                levels.append(self.LEVEL_CRITICAL)
                reasons.append(f"異常分數 ({s_t:.2f}) 突破常態臨界門檻 ({self.t_crit:.2f})")
                lead_signals.append(False)

            # 2. 結構裂解加速 (橙燈)
            elif (s_t >= self.t_crit * 0.75) or (s_t >= self.t_watch and top_pct >= 45.0):
                levels.append(self.LEVEL_ELEVATED)
                reasons.append(f"結構裂解加速中 (主導衝擊: {top_ticker} 佔 {top_pct:.1f}%)")
                lead_signals.append(False)

            # 3. 不對勁起頭 / 早期潛伏異動 (黃燈)
            # 條件 A: 分數超越警戒門檻
            # 條件 B: 分數雖在警戒線下，但變化斜率異常陡峭 (加速度爆發，代表結構正在急速裂解)
            elif s_t >= self.t_watch or (v_t > self.score_diff_std * 2.0 and a_t > 0):
                levels.append(self.LEVEL_WATCH)
                reasons.append(f"【不對勁起頭】定價慣性開始鬆動 (Score={s_t:.2f}, ΔScore/日=+{v_t:.2f})")
                lead_signals.append(True)

            # 4. 常態穩定 (綠燈)
            else:
                levels.append(self.LEVEL_NORMAL)
                reasons.append("跨板塊協同穩定，符合常態定價慣性")
                lead_signals.append(False)

        df_out = pd.DataFrame({
            "Date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
            "Score": scores,
            "Velocity": velocities,
            "Acceleration": accelerations,
            "Level": levels,
            "IsEarlyOnset": lead_signals,
            "TopContributor": max_contributors,
            "TopContribPct": max_contrib_pcts,
            "Diagnosis": reasons,
        })
        return df_out

    def generate_current_alert_report(self, df_signals):
        """產生最新一日的文字警示報告 (終端機輸出與日誌)"""
        latest = df_signals.iloc[-1]
        date_str = latest["Date"]
        level = latest["Level"]
        score = latest["Score"]
        velo = latest["Velocity"]
        top_t = latest["TopContributor"]
        top_p = latest["TopContribPct"]
        diag = latest["Diagnosis"]

        level_icons = {
            self.LEVEL_NORMAL: "🟢 [安全常態 - NORMAL]",
            self.LEVEL_WATCH: "🟡 [起頭預警 - ONSET WATCH]",
            self.LEVEL_ELEVATED: "🟠 [結構裂解 - ELEVATED RISK]",
            self.LEVEL_CRITICAL: "🔴 [臨界爆發 - CRITICAL RUPTURE]",
        }

        icon_title = level_icons.get(level, level)

        report = []
        report.append("=" * 65)
        report.append(f" ETF 早期預警系統 (EWS) 最新即時狀態報表 | {date_str}")
        report.append("=" * 65)
        report.append(f"【預警燈號】: {icon_title}")
        report.append(f"【異常分數】: {score:.2f} (警戒線: {self.t_watch:.2f} | 臨界線: {self.t_crit:.2f})")
        report.append(f"【變動速度】: {velo:+.2f} / 日 (一階斜率指標)")
        report.append(f"【最大失真板塊】: {top_t} (佔比: {top_p:.1f}%)")
        report.append(f"【系統診斷】: {diag}")
        report.append("-" * 65)

        # 操作建議
        if level == self.LEVEL_NORMAL:
            report.append("【量化避險建議】: 市場結構穩定，維持常規多頭或投資組合配置。")
        elif level == self.LEVEL_WATCH:
            report.append("【量化避險建議】: ⚠️ 偵測到結構不對勁起頭！建議收緊移動止損，暫停大幅追價加倉。")
        elif level == self.LEVEL_ELEVATED:
            report.append("【量化避險建議】: ⚠️ 板塊衝擊加速外溢，建議調降脆弱板塊曝險，提高現金水位。")
        else:
            report.append("【量化避險建議】: 🚨 系統性結構崩壞，觸發主動式動態避險 (Protective Hedging)。")
        report.append("=" * 65)

        return "\n".join(report)
