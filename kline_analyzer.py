# =============================================================================
# kline_analyzer.py  核心分析模块
# 已移除 RSI；升级了均线、成交量、MACD、趋势动量、压力/支撑、影线等算法；
# 输出结构化结果 dict，供 GUI 以表格形式展示。
# =============================================================================
import cv2
import numpy as np
from config import (REGION_RATIO, COLOR_HSV, SCORE_WEIGHT, PROB_CONFIG,
                    DRAW_COLOR, ANALYZE_WIDTH_RATIO)
from utils import safe_moments, preprocess_mask, split_region_by_ratio, crop_width_ratio


# ---------------- 基础掩膜辅助 ----------------
def _mask_red(hsv):
    """合并红色HSV区间掩膜（阳线/MACD红柱通用）"""
    m1 = cv2.inRange(hsv, np.array(COLOR_HSV["k_red"]["low"]), np.array(COLOR_HSV["k_red"]["high"]))
    m2 = cv2.inRange(hsv, np.array(COLOR_HSV["k_red2"]["low"]), np.array(COLOR_HSV["k_red2"]["high"]))
    return cv2.bitwise_or(m1, m2)


def _crop(img, top, bottom):
    """按高度切分 + 左侧宽度裁剪，返回分析区域"""
    reg = split_region_by_ratio(img, top, bottom)
    return crop_width_ratio(reg, ANALYZE_WIDTH_RATIO)


def _col_extent(col_mask, min_px=3):
    """计算某列有效像素的 (min_y, max_y, height)；像素不足返回 None"""
    ys = np.where(col_mask > 0)[0]
    if len(ys) < min_px:
        return None
    return int(np.min(ys)), int(np.max(ys)), int(np.max(ys) - np.min(ys))


# =============================================================================
# 1. K线实体识别（结构优先 + 列密度过滤 + 间距去重）
# 说明：K线颜色可能与均线相同、且实体有实心/空心之分，纯颜色分割不可靠。
# 这里以“竖直窄高实体 + 每列彩色像素密度”识别蜡烛体，并按最小间距去重，
# 结果用于红蓝占比/影线/支撑等补充信号；决策主干以颜色干净的 MA/MACD/量为主。
# =============================================================================
def _bar_candidates(mask, is_red):
    """逐列找主纵向色块，按相邻列合并成候选蜡烛，返回 (cx,top,bot,h,w,density)"""
    colruns = []
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        diff = np.diff(ys)
        br = np.where(diff > 2)[0]
        for seg in np.split(ys, br + 1):
            if len(seg) >= 4:
                colruns.append((x, int(seg[0]), int(seg[-1])))
    colruns.sort(key=lambda c: (c[0], c[1]))
    bars, cur = [], []
    for r in colruns:
        if not cur:
            cur = [r]
            continue
        if r[0] - cur[-1][0] <= 3 and not (r[2] < cur[0][1] - 4 or r[1] > cur[0][2] + 4):
            cur.append(r)
            continue
        bars.append(cur)
        cur = [r]
    if cur:
        bars.append(cur)

    out = []
    for b in bars:
        xs = [r[0] for r in b]
        tops = [r[1] for r in b]
        bots = [r[2] for r in b]
        x0, x1 = min(xs), max(xs)
        top, bot = min(tops), max(bots)
        w = x1 - x0 + 1
        hh = bot - top + 1
        if not (3 <= w <= 45 and hh >= 16):
            continue
        # 每列彩色像素密度：均线(细/粗曲线)每列仅1~3px，蜡烛体>=5px
        density = np.count_nonzero(mask[top:bot + 1, x0:x1 + 1]) / w
        if density < 5:
            continue
        cx = int(sum(r[0] for r in b) / len(b))
        out.append({"cx": cx, "top": top, "bot": bot, "h": hh, "w": w,
                    "density": density, "is_red": is_red})
    return out


def detect_candles(main_area):
    h_main, w_main = main_area.shape[:2]
    k_region = main_area[:int(h_main * 0.92), :]
    hsv = cv2.cvtColor(k_region, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 红/蓝掩膜（饱和度+亮度，色相放宽以兼容不同行情软件的红蓝K线）
    mred = ((((H <= 12) | (H >= 168)) & (S >= 60) & (V >= 50)).astype(np.uint8)) * 255
    mblue = (((H >= 78) & (H <= 165) & (S >= 60) & (V >= 50)).astype(np.uint8)) * 255

    cand = _bar_candidates(mred, True) + _bar_candidates(mblue, False)
    cand.sort(key=lambda c: c["cx"])

    # 间距去重：相邻两候选间隔<=12px视为同一根（或粗均线残留），保留更“实/高”者
    dedup = []
    for c in cand:
        if dedup and c["cx"] - dedup[-1]["cx"] <= 12:
            p = dedup[-1]
            if c["h"] * c["density"] > p["h"] * p["density"]:
                dedup[-1] = c
            continue
        dedup.append(c)

    k_list = [{"x": c["cx"], "y": c["top"], "h": c["h"], "w": c["w"],
               "cx": c["cx"], "cy": c["top"] + c["h"] // 2, "is_red": c["is_red"]}
              for c in dedup]
    k_list.sort(key=lambda k: k["x"])   # 按时间顺序从左到右
    return k_list


# =============================================================================
# 2. 均线分析（MA5 vs MA20 排列 + 斜率）
# =============================================================================
def analyze_ma(main_area):
    h = main_area.shape[0]
    hsv = cv2.cvtColor(main_area, cv2.COLOR_BGR2HSV)
    mask_ma5 = preprocess_mask(cv2.inRange(hsv, np.array(COLOR_HSV["ma5_green"]["low"]),
                                           np.array(COLOR_HSV["ma5_green"]["high"])))
    mask_ma20 = preprocess_mask(cv2.inRange(hsv, np.array(COLOR_HSV["ma20_yellow"]["low"]),
                                            np.array(COLOR_HSV["ma20_yellow"]["high"])))
    w = mask_ma5.shape[1]
    right_start = int(w * 0.62)          # 右侧38% 为最新区域，屏蔽左侧历史线条
    m5 = mask_ma5[:, right_start:]
    m20 = mask_ma20[:, right_start:]
    wn = m5.shape[1]

    n_win = 4
    y5_list, y20_list = [], []
    bulls = 0
    valid = 0
    for i in range(n_win):
        x0 = int(wn * i / n_win)
        x1 = int(wn * (i + 1) / n_win)
        win5, win20 = m5[:, x0:x1], m20[:, x0:x1]
        if cv2.countNonZero(win5) < 14 or cv2.countNonZero(win20) < 14:
            continue
        r5 = safe_moments(win5)
        r20 = safe_moments(win20)
        if not r5 or not r20:
            continue
        valid += 1
        y5_list.append(r5["cy"])
        y20_list.append(r20["cy"])
        bulls += 1 if r5["cy"] < r20["cy"] else 0   # y小=图上靠上=价高 → MA5在MA20上方

    if valid == 0:
        return {"score": 20, "desc": "⚠️ 均线识别不完整，无法判断排列",
                "ma_bull": None, "data_ok": False}

    ma5_above = bulls >= 3                          # 多数窗口 MA5 在上
    # 斜率：cy 增大 = 图上向下 = 价格下跌；末端 cy < 开头 cy = 均线向上
    slope_th = max(6, int(h * 0.012))
    ma5_up = (y5_list[0] - y5_list[-1]) > slope_th if len(y5_list) >= 2 else None
    ma20_up = (y20_list[0] - y20_list[-1]) > slope_th if len(y20_list) >= 2 else None
    # 趋势倾向（供趋势动量复用，均线斜率比蜡烛位置可靠）
    if ma20_up and ma5_up:
        trend_hint = "up"
    elif ma20_up is False and ma5_up is False:
        trend_hint = "down"
    elif ma20_up:
        trend_hint = "weak_up"
    elif ma20_up is False:
        trend_hint = "weak_down"
    else:
        trend_hint = "flat"

    if ma5_above:
        if ma20_up and ma5_up:
            return {"score": SCORE_WEIGHT["ma"], "desc": "✅ MA5站上MA20且均线向上，多头排列",
                    "ma_bull": True, "data_ok": True, "trend_hint": trend_hint}
        elif ma20_up:
            return {"score": int(SCORE_WEIGHT["ma"] * 0.9), "desc": "✅ MA5上穿MA20，MA20向上，金叉",
                    "ma_bull": True, "data_ok": True, "trend_hint": trend_hint}
        else:
            return {"score": int(SCORE_WEIGHT["ma"] * 0.65), "desc": "🟡 MA5在MA20上方但MA20仍向下，反弹性质谨慎",
                    "ma_bull": True, "data_ok": True, "trend_hint": trend_hint}
    else:
        if ma20_up is False and ma5_up is False:
            return {"score": 8, "desc": "❌ MA5在MA20下方且均线向下，空头排列",
                    "ma_bull": False, "data_ok": True, "trend_hint": trend_hint}
        elif ma20_up is False:
            return {"score": 12, "desc": "❌ MA5在MA20下方，空头信号",
                    "ma_bull": False, "data_ok": True, "trend_hint": trend_hint}
        else:
            return {"score": 16, "desc": "🟡 MA5在MA20下方但MA20走平/向上，方向待确认",
                    "ma_bull": False, "data_ok": True, "trend_hint": trend_hint}


# =============================================================================
# 3. 成交量分析（最新一根 vs 历史平均，尺度不变）
# =============================================================================
def analyze_vol(vol_area):
    w = vol_area.shape[1]
    hist_end = int(w * 0.90)                 # 左侧90% 历史
    his = vol_area[:, :hist_end]
    now = vol_area[:, hist_end:]
    hsv_h = cv2.cvtColor(his, cv2.COLOR_BGR2HSV)
    mask_his = preprocess_mask(cv2.inRange(hsv_h, np.array(COLOR_HSV["vol_bar"]["low"]),
                                           np.array(COLOR_HSV["vol_bar"]["high"])))
    hsv_n = cv2.cvtColor(now, cv2.COLOR_BGR2HSV)
    mask_now = preprocess_mask(cv2.inRange(hsv_n, np.array(COLOR_HSV["vol_bar"]["low"]),
                                           np.array(COLOR_HSV["vol_bar"]["high"])))
    his_density = cv2.countNonZero(mask_his) / max(1, hist_end)      # 每列平均像素密度
    now_density = cv2.countNonZero(mask_now) / max(1, now.shape[1])
    ratio = now_density / his_density if his_density > 0 else 0.0

    if ratio >= 1.45:
        return {"score": SCORE_WEIGHT["vol"], "desc": "✅ 最新K线显著放量，资金确认",
                "vol_strong": True, "ratio": ratio}
    elif ratio >= 1.1:
        return {"score": int(SCORE_WEIGHT["vol"] * 0.88), "desc": "✅ 最新K线放量",
                "vol_strong": True, "ratio": ratio}
    elif ratio >= 0.72:
        return {"score": int(SCORE_WEIGHT["vol"] * 0.6), "desc": "🟡 成交量与近期基本持平",
                "vol_strong": False, "ratio": ratio}
    else:
        return {"score": 8, "desc": "❌ 最新K线成交量萎缩，反弹/下跌缺资金",
                "vol_strong": False, "ratio": ratio}


# =============================================================================
# 4. MACD 分析（红绿柱方向 + 0轴 + 柱体动量）
# =============================================================================
def analyze_macd(macd_area):
    w = macd_area.shape[1]
    right_start = int(w * 0.55)              # 右侧45% 最近柱区
    region = macd_area[:, right_start:]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mred = preprocess_mask(_mask_red(hsv))
    mgreen = preprocess_mask(cv2.inRange(hsv, np.array(COLOR_HSV["macd_green"]["low"]),
                                         np.array(COLOR_HSV["macd_green"]["high"])))
    # 形态学开运算消除细小噪点
    kernel = np.ones((2, 2), np.uint8)
    mred = cv2.morphologyEx(mred, cv2.MORPH_OPEN, kernel)
    mgreen = cv2.morphologyEx(mgreen, cv2.MORPH_OPEN, kernel)

    red_heights, green_heights = [], []
    for x in range(region.shape[1]):
        e = _col_extent(mred[:, x])
        if e:
            red_heights.append(e[2])
        e = _col_extent(mgreen[:, x])
        if e:
            green_heights.append(e[2])

    red_sum = sum(red_heights)
    green_sum = sum(green_heights)
    if red_sum == 0 and green_sum == 0:
        return {"score": 10, "desc": "⚠️ 未识别到MACD柱信号，无法判断",
                "macd_bull": False, "data_ok": False}

    red_dom = red_sum >= green_sum
    # 柱体动量：对比每侧后半段与前半段的平均高度（0轴由柱色隐含）
    def _momentum(heights):
        if len(heights) < 2:
            return None
        half = len(heights) // 2
        head = np.mean(heights[:half]) if half > 0 else 0
        tail = np.mean(heights[half:]) if half < len(heights) else np.mean(heights)
        return tail >= head

    mom_red = _momentum(red_heights)
    mom_green = _momentum(green_heights)

    # DIF/DEA 相对0轴：红柱在0轴上方、绿柱在0轴下方，直接用柱色判断环境
    if red_dom:
        if mom_red is False:                       # 红柱但动能在减弱
            return {"score": 28, "desc": "🟡 MACD红柱占优但动能在减弱，注意钝化",
                    "macd_bull": True, "data_ok": True}
        return {"score": SCORE_WEIGHT["macd"], "desc": "✅ MACD红柱占优且动能向上，多头动量",
                "macd_bull": True, "data_ok": True}
    else:
        if mom_green is False:                     # 绿柱但动能在减弱 → 杀跌衰竭
            return {"score": 12, "desc": "🟡 MACD绿柱占优但动能在衰减，杀跌临近衰竭",
                    "macd_bull": False, "data_ok": True}
        return {"score": 8, "desc": "❌ MACD绿柱占优，下跌动量延续",
                "macd_bull": False, "data_ok": True}


# =============================================================================
# 5. 趋势动量（基于均线斜率，颜色干净、可靠；不再依赖脆弱的蜡烛位置）
# =============================================================================
def analyze_trend(ma):
    hint = ma.get("trend_hint", "flat")
    m = SCORE_WEIGHT["trend"]
    if hint == "up":
        return {"score": m, "desc": "✅ 均线向上延伸，短期上涨动量强", "data_ok": True}
    if hint == "weak_up":
        return {"score": int(m * 0.6), "desc": "🟢 均线温和向上，偏多", "data_ok": True}
    if hint == "down":
        return {"score": -m, "desc": "❌ 均线向下延伸，短期下跌动量强", "data_ok": True}
    if hint == "weak_down":
        return {"score": -int(m * 0.6), "desc": "🔴 均线温和向下，偏空", "data_ok": True}
    return {"score": 0, "desc": "🟡 均线走平，趋势方向不明", "data_ok": True}


# =============================================================================
# 6. 压力/支撑位（基于近期K线高低点）
# =============================================================================
def analyze_levels(k_list, main_h):
    if len(k_list) < 5:
        return {"resist_score": 0, "support_score": 0, "desc": "K线不足无法评估压力/支撑"}
    ys = [k["y"] for k in k_list]
    recent = ys[-12:] if len(ys) >= 12 else ys
    recent_high_y = min(recent)          # 顶部=最高价
    recent_low_y = max(recent)           # 底部=最低价
    cur = ys[-1]
    rng = recent_low_y - recent_high_y
    resist_score, support_score = 0, 0
    desc = "✅ 价格位于近期区间中段，压力/支撑均衡"
    if rng > 0:
        dist_high = (cur - recent_high_y) / rng
        dist_low = (recent_low_y - cur) / rng
        if dist_high < 0.18:
            resist_score = SCORE_WEIGHT["resist"]
            desc = "🔴 价格贴近近期高点压力位，追多风险大"
        elif dist_high < 0.36:
            resist_score = SCORE_WEIGHT["resist"] // 2
            desc = "🟠 价格接近上方压力位，注意承压"
        if dist_low < 0.15:
            support_score = SCORE_WEIGHT["support"]
            desc = "🟢 价格贴近近期低点支撑位，存在反弹机会"
    return {"resist_score": resist_score, "support_score": support_score, "desc": desc}


# =============================================================================
# 7. 最后一根K线：影线 + 十字星形态
# =============================================================================
def analyze_last_shape(main_area, k_list):
    if not k_list:
        return {"shadow_score": 0, "pattern_score": 0, "desc": "无有效K线，形态未识别"}
    last = k_list[-1]
    h_main = main_area.shape[0]
    x0 = max(0, last["cx"] - 3)
    x1 = last["cx"] + 4
    strip = main_area[:, x0:x1]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    candle = cv2.bitwise_or(_mask_red(hsv),
                            cv2.inRange(hsv, np.array(COLOR_HSV["k_blue"]["low"]),
                                        np.array(COLOR_HSV["k_blue"]["high"])))
    candle = preprocess_mask(candle)
    kernel = np.ones((2, 2), np.uint8)
    candle = cv2.morphologyEx(candle, cv2.MORPH_OPEN, kernel)
    ys = np.where(candle > 0)[0]
    if len(ys) < 8:
        return {"shadow_score": 0, "pattern_score": 0, "desc": "K线实体像素不足，形态未识别"}

    ymin, ymax = int(np.min(ys)), int(np.max(ys))
    full = ymax - ymin
    p30 = float(np.percentile(ys, 30))
    p70 = float(np.percentile(ys, 70))
    body = max(p70 - p30, 1.0)
    upper = max(p30 - ymin, 0)
    lower = max(ymax - p70, 0)

    shadow_score, pattern_score = 0, 0
    tags = []
    # 长上影：上方影线明显长于实体 → 抛压
    if body > 8 and upper / body > 2.2:
        shadow_score = SCORE_WEIGHT["shadow"]
        tags.append("⚠️ 长上影抛压")
    else:
        tags.append("✅ 无长上影压力")
    # 十字星：实体极短 → 反转预警
    if body < max(6, h_main * 0.008) and full > body * 3:
        pattern_score = SCORE_WEIGHT["pattern"]
        tags.append("⚠️ 十字星，趋势可能反转")
    else:
        tags.append("实体正常")
    return {"shadow_score": shadow_score, "pattern_score": pattern_score,
            "desc": "，".join(tags)}


# =============================================================================
# 8. 连续同向K线
# =============================================================================
def analyze_continue(k_list):
    if not k_list:
        return {"score": 0, "desc": "🟡 未检测到有效K线实体"}
    last_red = k_list[-1]["is_red"]
    cnt = 0
    for k in reversed(k_list):
        if k["is_red"] == last_red:
            cnt += 1
        else:
            break
    if last_red:
        if cnt >= 3:
            return {"score": SCORE_WEIGHT["continue_k"], "desc": f"✅ 连续{cnt}根阳线，短线多头延续"}
        return {"score": 0, "desc": f"🟡 末尾{cnt}根阳线，不足3根无连续多头"}
    else:
        if cnt >= 3:
            return {"score": -SCORE_WEIGHT["continue_k"], "desc": f"❌ 连续{cnt}根阴线，短线空头延续"}
        return {"score": 0, "desc": f"🟡 末尾{cnt}根阴线，不足3根无连续空头"}


# =============================================================================
# 9. 综合得分 → 概率（含动态震荡概率 + 强制决策）
# =============================================================================
def compute_prob(total_score):
    k = PROB_CONFIG["sigmoid_k"]
    center = PROB_CONFIG["sigmoid_center"]
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-k * (x - center)))
    bull_s = _sig(total_score)
    bear_s = _sig(100 - total_score)
    # 由于 sigmoid 对称性，bull_s + bear_s ≈ 1；用置信度推导震荡概率
    conf = abs(bull_s - bear_s) * 100                      # 0~100，越极端越大
    side_pct = (100 - conf) * PROB_CONFIG["side_max_ratio"]  # 信号越弱震荡越高
    remain = 100 - side_pct
    bull_pct = bull_s / (bull_s + bear_s) * remain
    bear_pct = bear_s / (bull_s + bear_s) * remain
    return round(bull_pct, 1), round(bear_pct, 1), round(side_pct, 1)


def compute_forced(bull_pct, bear_pct):
    """若必须开仓：二选一的方向 + 置信度"""
    total = bull_pct + bear_pct
    if total <= 0:
        return {"decision": "买跌", "confidence": 50.0, "note": "信号缺失，按中性处理"}
    conf = round(max(bull_pct, bear_pct) / total * 100, 1)
    decision = "买涨" if bull_pct >= bear_pct else "买跌"
    note = ""
    if conf < PROB_CONFIG["forced_conf_min"]:
        note = "⚠️ 两方向概率接近，信号偏弱，若必须开仓请轻仓并设严格止损"
    elif conf >= 70:
        note = "🔥 方向信号较明确，可正常仓位配合止损"
    else:
        note = "🟡 方向有一定优势，建议半仓试探并严格止损"
    return {"decision": decision, "confidence": conf, "note": note}


def build_action(bull_pct, bear_pct):
    """综合操作建议：买涨 / 买跌 / 观望"""
    up_th = PROB_CONFIG["action_up_th"]
    gap_th = PROB_CONFIG["action_gap_th"]
    if bull_pct >= up_th and (bull_pct - bear_pct) >= gap_th:
        return "买涨"
    if bear_pct >= up_th and (bear_pct - bull_pct) >= gap_th:
        return "买跌"
    return "观望"


# =============================================================================
# 10. 主分析入口：结构化结果
# =============================================================================
def calc_kline_feature(img_cv):
    h_full, w_full = img_cv.shape[:2]
    if h_full <= 10 or w_full <= 10:
        return {"error": "图片尺寸过小，无法识别K线"}

    main_area = _crop(img_cv, REGION_RATIO["main_top"], REGION_RATIO["main_bottom"])
    vol_area = _crop(img_cv, REGION_RATIO["vol_top"], REGION_RATIO["vol_bottom"])
    macd_area = _crop(img_cv, REGION_RATIO["macd_top"], REGION_RATIO["macd_bottom"])

    # ---- K线实体（一次识别，供多处复用） ----
    k_list = detect_candles(main_area)
    red_cnt = sum(1 for k in k_list if k["is_red"])
    blue_cnt = len(k_list) - red_cnt
    k_red_ratio = round(red_cnt / len(k_list) * 100, 2) if k_list else 0.0
    k_blue_ratio = round(blue_cnt / len(k_list) * 100, 2) if k_list else 0.0

    # ---- 各指标 ----
    ma = analyze_ma(main_area)
    vol = analyze_vol(vol_area)
    macd = analyze_macd(macd_area)
    trend = analyze_trend(ma)
    levels = analyze_levels(k_list, main_area.shape[0])
    shape = analyze_last_shape(main_area, k_list)
    cont = analyze_continue(k_list)

    # ---- 止损/入场参考（图像像素，仅供截图定位） ----
    entry_y = None
    stop_long_y = None
    stop_short_y = None
    if len(k_list) > 0:
        ys = [k["y"] for k in k_list]
        recent = ys[-12:] if len(ys) >= 12 else ys
        recent_high_y = min(recent)
        recent_low_y = max(recent)
        price_range = recent_low_y - recent_high_y
        cur_body_y = k_list[-1]["cy"]
        entry_y = int(cur_body_y)
        if price_range > 0:
            stop_long_y = int(recent_low_y + price_range * 0.05)
            stop_short_y = int(recent_high_y - price_range * 0.05)

    # ---- 综合得分 ----
    total_score = (ma["score"] + vol["score"] + macd["score"] + trend["score"]
                   + levels["resist_score"] + levels["support_score"]
                   + shape["shadow_score"] + shape["pattern_score"] + cont["score"])
    total_score = int(np.clip(total_score, PROB_CONFIG["score_min"], PROB_CONFIG["score_max"]))

    bull_pct, bear_pct, side_pct = compute_prob(total_score)
    forced = compute_forced(bull_pct, bear_pct)
    action = build_action(bull_pct, bear_pct)

    # ---- 结构化输出 ----
    rows = [
        {"name": "MA均线",   "desc": ma["desc"],      "score": ma["score"],   "max": SCORE_WEIGHT["ma"], "tag": ""},
        {"name": "成交量",   "desc": vol["desc"],     "score": vol["score"],  "max": SCORE_WEIGHT["vol"], "tag": ""},
        {"name": "MACD",     "desc": macd["desc"],    "score": macd["score"], "max": SCORE_WEIGHT["macd"], "tag": ""},
        {"name": "趋势动量", "desc": trend["desc"],   "score": trend["score"],"max": SCORE_WEIGHT["trend"], "tag": ""},
        {"name": "压力支撑", "desc": levels["desc"],  "score": levels["resist_score"] + levels["support_score"], "max": SCORE_WEIGHT["support"], "tag": "估算"},
        {"name": "影线形态", "desc": shape["desc"],   "score": shape["shadow_score"] + shape["pattern_score"], "max": 0, "tag": "估算"},
        {"name": "连续K线",  "desc": cont["desc"],    "score": cont["score"],"max": SCORE_WEIGHT["continue_k"], "tag": "估算"},
        {"name": "K线多空",  "desc": f"阳线占{k_red_ratio}% / 阴线占{k_blue_ratio}% (识别{len(k_list)}根)", "score": 0, "max": 0, "tag": "估算"},
    ]

    # 数据可信度：主干信号(MA/MACD/量)是否有效
    reliable_ok = sum(1 for s in [ma.get("data_ok"), macd.get("data_ok"), vol.get("data_ok")] if s)
    if reliable_ok >= 2:
        data_conf = "较高"
    elif reliable_ok == 1:
        data_conf = "一般"
    else:
        data_conf = "较低"
    if not k_list:
        k_note = "（未能识别到K线实体，红蓝/影线/支撑为估算）"
    elif len(k_list) < 8:
        k_note = f"（K线识别数量偏少:{len(k_list)}根，红蓝/影线/支撑仅供参考）"
    else:
        k_note = ""

    reasons = []
    reasons.append(f"综合得分 {total_score}/100，上涨概率 {bull_pct}%，下跌概率 {bear_pct}%，震荡 {side_pct}%。数据可信度：{data_conf}。")
    # 信号一致性（如实反映，不虚构共振）
    ma_b = ma.get("ma_bull")
    macd_b = macd.get("macd_bull")
    if ma_b is True and macd_b is True:
        reasons.append("信号一致：均线多头 + MACD多头 共振，做多信号较可靠。")
    elif ma_b is False and macd_b is False:
        reasons.append("信号一致：均线空头 + MACD空头 共振，做空信号较可靠。")
    else:
        reasons.append("信号分歧：均线与MACD方向不一致，以综合概率为准，胜率打折。")
    if action == "买涨":
        reasons.append("综合概率偏多，若选择进场请轻仓试探，并在跌破近期低点或MA5下穿MA20时离场。")
    elif action == "买跌":
        reasons.append("综合概率偏空，若选择进场请轻仓试探，并在突破近期高点或MA5上穿MA20时离场。")
    else:
        reasons.append("当前信号矛盾、概率接近，属于低胜率区，建议观望等待。")
    reasons.append("1分钟周期信号变化极快，任何一笔都必须带止损，禁止扛单。")

    # 强制决策的入场/止损/目标（图像像素参考）
    forced["entry_ref"] = entry_y
    forced["stop_long_ref"] = stop_long_y
    forced["stop_short_ref"] = stop_short_y

    result = {
        "rows": rows,
        "total_score": total_score,
        "prob_up": bull_pct,
        "prob_down": bear_pct,
        "prob_side": side_pct,
        "action": action,
        "reasons": reasons,
        "data_conf": data_conf,
        "k_note": k_note,
        "forced": forced,
        "entry_y": entry_y,
        "stop_long_y": stop_long_y,
        "stop_short_y": stop_short_y,
        "k_red_ratio": k_red_ratio,
        "k_blue_ratio": k_blue_ratio,
        "k_count": len(k_list),
    }
    return result


# =============================================================================
# 11. 绘图（止损线标注，纯可视化）
# =============================================================================
def draw_mark_image(src_img, feat):
    draw_img = src_img.copy()
    if "error" in feat:
        return draw_img
    h, w = draw_img.shape[:2]
    # 止损参考Y换算回整图坐标：主图区域相对整图从 REGION_RATIO['main_top'] 开始
    main_top = REGION_RATIO["main_top"]
    main_bottom = REGION_RATIO["main_bottom"]
    y_src = lambda yy: int(main_top * h + yy / (main_bottom - main_top) * (main_bottom * h - main_top * h)) \
        if yy is not None else None
    slong = y_src(feat.get("stop_long_y"))
    sshort = y_src(feat.get("stop_short_y"))
    if slong is not None and 0 <= slong < h:
        cv2.line(draw_img, (0, slong), (w, slong), DRAW_COLOR["stop_long"], 2)
        cv2.putText(draw_img, "做多止损", (20, slong - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    DRAW_COLOR["stop_long"], 2)
    if sshort is not None and 0 <= sshort < h:
        cv2.line(draw_img, (0, sshort), (w, sshort), DRAW_COLOR["stop_short"], 2)
        cv2.putText(draw_img, "做空止损", (20, sshort + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    DRAW_COLOR["stop_short"], 2)
    return draw_img


# =============================================================================
# 12. 对外统一入口
# =============================================================================
def analyze_kline_image(img_cv):
    feat = calc_kline_feature(img_cv)
    draw_img = draw_mark_image(img_cv, feat)
    return feat, draw_img
