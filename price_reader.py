# =============================================================================
# price_reader.py  真实价格与指标读取模块（v3，数量级聚类）
# 调用 Windows 自带 OCR（离线），整图 2 倍放大后读取头部/指标行的真实数值，
# 并建立“像素Y → 真实价格”映射，把入场/止损定位到具体价格。
# 解析核心：先按“同一数量级 + 至少4个数字”锁定 OHLC 行，用其中位数确定价格
# 量级，再按该量级过滤 MA/压力/支撑；成交量和 MACD 按行特征单独解析。
# 失败自动回退颜色识别，不影响主流程。
# =============================================================================
import os
import re
import subprocess
import tempfile

_OCR_PS1 = r"""
param([string]$Path)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType=WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
try {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($Path)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if(-not $engine){ $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("zh-CN")) }
    if(-not $engine){ Write-Output "__NO_ENGINE__"; exit }
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    foreach($line in $result.Lines){
        foreach($word in $line.Words){
            $r = $word.BoundingRect
            Write-Output ("{0}`t{1}`t{2}`t{3}`t{4}" -f $word.Text, [int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height)
        }
    }
} catch {
    Write-Output ("__ERR__ " + $_.Exception.Message)
}
"""


def _run_powershell(ps1_path, img_path, timeout=40):
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
           "-File", ps1_path, "-Path", img_path]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.stdout or ""


def ocr_image(img_bgr, scale=2):
    tmpdir = tempfile.gettempdir()
    img_path = os.path.join(tmpdir, "kline_ocr_in.png")
    ps1_path = os.path.join(tmpdir, "kline_ocr_helper.ps1")
    try:
        import cv2
        if scale and scale > 1:
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img_bgr)
        with open(ps1_path, "w", encoding="utf-8-sig") as f:
            f.write(_OCR_PS1)
        out = _run_powershell(ps1_path, img_path)
    except Exception:
        return []
    words = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            t = parts[0].strip()
            try:
                words.append((t, int(parts[1]), int(parts[2]),
                              int(parts[3]), int(parts[4])))
            except ValueError:
                pass
    if scale and scale > 1:
        words = [(t, int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                 for (t, x, y, w, h) in words]
    return words


def _group_lines(words, y_tol=6):
    lines = []
    for w in sorted(words, key=lambda w: (w[2], w[1])):
        placed = False
        for ln in lines:
            if abs(ln["y"] - w[2]) <= y_tol:
                ln["words"].append(w)
                placed = True
                break
        if not placed:
            lines.append({"y": w[2], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w[1])
    return lines


def _int_runs(wds, gap=0):
    """返回 [(x, int)]：每个含数字的词取第一个数字串转 int（不跨词合并）"""
    out = []
    for w in wds:
        t, x = w[0], w[1]
        m = re.search(r'\d+', t)
        if m and m.group().isdigit():
            out.append((x, int(m.group())))
    return out


def _cluster(vals, min_count=4, ratio=1.6):
    """在排序值里找最大的、跨度在 ratio 内的连续簇"""
    vs = sorted(vals)
    best = []
    for i in range(len(vs) - min_count + 1):
        window = vs[i:i + min_count]
        if window[-1] / max(window[0], 1e-9) < ratio:
            if len(window) > len(best):
                best = window
    return best


def parse_ocr(words, img_h, img_w=None):
    """返回 dict：ohc、pressure、support、ma1、ma2、vol、vol_ma、diff、dea、macd、lines"""
    res = {"ohc": {}, "pressure": None, "support": None,
           "ma1": None, "ma2": None, "vol": None, "vol_ma": None,
           "diff": None, "dea": None, "macd": None, "lines": []}
    scale = None
    # 只取左 70% 宽度，排除右侧盘口/逐笔干扰
    x_limit = int((img_w or 9999) * 0.7)
    words = [w for w in words if w[1] < x_limit]
    lines = sorted(_group_lines(words), key=lambda l: l["y"])

    # ---- 锁定 OHLC 行：顶部、存在 >=4 个接近的数量级簇 ----
    ohc_line = None
    for ln in lines:
        if ln["y"] > img_h * 0.12:
            break
        ints = [v for _, v in _int_runs(ln["words"])]
        cl = _cluster(ints)
        if len(cl) >= 4:
            ohc_line = ln
            scale = cl[len(cl) // 2]
            break

    if ohc_line is not None:
        res["price_scale"] = scale
        lo, hi = scale * 0.9, scale * 1.1
        r = [(x, v) for x, v in _int_runs(ohc_line["words"]) if lo <= v <= hi]
        r.sort(key=lambda p: p[0])
        ohc4 = [v for _, v in r[:4]]
        if len(ohc4) >= 4:
            res["ohc"]["open"], res["ohc"]["high"], \
                res["ohc"]["low"], res["ohc"]["close"] = ohc4[0], ohc4[1], ohc4[2], ohc4[3]

        # ---- 顶部 MA / 压力支撑行：按 X 拆为“压力/支撑区”和“MA区” ----
        band = (scale * 0.8, scale * 1.2)
        for ln in lines:
            if ln is ohc_line or ln["y"] > img_h * 0.15:
                continue
            ints = sorted(_int_runs(ln["words"]), key=lambda p: p[0])
            pv = [(x, v) for x, v in ints if band[0] <= v <= band[1]]
            txt = "".join(w[0] for w in ln["words"])
            if len(pv) >= 2 and ("压力" in txt or "支撑" in txt or "MA" in txt or "移" in txt or "平" in txt):
                res["ma_line_y"] = ln["y"]
                # 左区（压力/支撑）与右区（MA），按中位 X 切分
                if pv:
                    mid = (pv[0][0] + pv[-1][0]) / 2.0
                    left = [v for x, v in pv if x <= mid]
                    right = [v for x, v in pv if x > mid]
                    if left:
                        res["support"] = left[-1]           # 支撑在压力之后
                        if len(left) >= 2:
                            res["pressure"] = left[-2]
                    if right:
                        res["ma1"] = right[0]
                    if len(right) >= 2:
                        res["ma2"] = right[1]
                break

    # ---- 成交量行：下半部，含大数(>3*scale) ----
    for ln in lines:
        if not (img_h * 0.48 <= ln["y"] <= img_h * 0.64):
            continue
        ints = sorted(_int_runs(ln["words"]), key=lambda p: p[0])
        thr = scale * 3 if scale else 10000
        bigs = [v for _, v in ints if thr <= v <= thr * 60]
        if bigs:
            res["vol"] = float(bigs[0])
            if len(bigs) >= 2:
                res["vol_ma"] = float(bigs[1])
            break

    # ---- MACD 行：下半区，DIFF/DEA 形如 1.6145（OCR 拆成 “1”+“6145” 或直接 “6145”） ----
    for ln in lines:
        if ln["y"] < img_h * 0.66:
            continue
        ints = sorted(_int_runs(ln["words"]), key=lambda p: p[0])
        txt = "".join(w[0] for w in ln["words"])
        # 配对：单数字(1) + 4位小数段(6145) → 16145 → 1.6145；若没有单数字，则按 0.xxxx
        fracs = []
        i = 0
        while i < len(ints):
            x, v = ints[i]
            if 1 <= v <= 9 and i + 1 < len(ints) and 1000 <= ints[i + 1][1] <= 9999 \
                    and (ints[i + 1][0] - (x + 3)) < 16:
                combined = int(str(v) + str(ints[i + 1][1]))
                fracs.append(combined / 10000.0)
                i += 2
                continue
            if 1000 <= v <= 9999:
                fracs.append(v / 10000.0)
            i += 1
        if len(fracs) >= 2 and ("D" in txt or "F" in txt or "DEA" in txt or "平" in txt):
            res["diff"] = fracs[0]
            res["dea"] = fracs[1]
            break
    return res


# ---------------------------------------------------------------------------
# MA 行增强读取：对 MA 行区域做 5 倍放大 OCR，恢复带小数的 MA1/MA2 真实值
# （整图 2 倍对 "1043.20" 这种带小数的小字会读坏，5 倍可读准）
# ---------------------------------------------------------------------------
def _parse_ma_values(wds):
    wds = sorted(wds, key=lambda w: w[1])
    key_pos, num_items = [], []
    for w in wds:
        t, x = w[0], w[1]
        km = re.search(r'MA\s*([12I])', t, re.IGNORECASE)
        dm = re.search(r'(\d+)', t)
        if km:
            key_pos.append((x, km.group(1).upper()))
        if dm and dm.group().isdigit():
            num_items.append((x, int(dm.group())))
    num_items.sort()
    res = {}
    for kx, kn in key_pos:
        cand = [n for n in num_items if n[0] > kx]
        if not cand:
            continue
        first = cand[0]
        val = float(first[1])
        # 小数部分必须是 2~3 位（.20/.40），避免把 MA2 标签里的单个数字“2”误当小数
        if len(cand) >= 2 and (cand[1][0] - first[0]) < 42 \
                and 2 <= len(str(cand[1][1])) <= 3:
            frac = cand[1][1]
            val += frac / float(10 ** len(str(frac)))
        key = "ma1" if kn in ("1", "I") else "ma2"
        res[key] = round(val, 2)
    return res


def enhance_ma(img_bgr, ocr, scale=10):
    """对 MA 值区域 10 倍放大 OCR，读准带小数的 MA1/MA2。
    仅在读到合理价位时才覆盖原解析，否则保留原值（避免把读对的改坏）。"""
    if not ocr or not ocr.get("ma_line_y") or not img_bgr.size:
        return ocr
    try:
        import cv2
        h = img_bgr.shape[0]
        w = img_bgr.shape[1]
        y0 = max(0, int(ocr["ma_line_y"]) - 11)
        y1 = min(h, int(ocr["ma_line_y"]) + 13)
        # 窄裁剪 MA1/MA2 值区域（宽裁剪会把带小数的小字读坏）
        x0 = int(w * 0.19)
        x1 = int(w * 0.35)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return ocr
        words = ocr_image(crop, scale=scale)
        words = [(t, int(x / scale), int(y / scale), int(ww / scale), int(hh / scale))
                 for (t, x, y, ww, hh) in words]
        lines = _group_lines(words)
        if not lines:
            return ocr
        main = sorted(lines, key=lambda l: sum(w[3] for w in l["words"]), reverse=True)[0]
        vals = _parse_ma_values(main["words"])
        # 合理性校验：读到的 MA 应落在价格量级附近（0.8~1.2 倍），否则视为误读
        scale_ref = ocr.get("price_scale") or ocr.get("ma1") or 1000.0
        lo, hi = scale_ref * 0.8, scale_ref * 1.2
        for key in ("ma1", "ma2"):
            if vals.get(key) is not None and lo <= vals[key] <= hi:
                ocr[key] = vals[key]
    except Exception:
        pass
    return ocr


# ---------------------------------------------------------------------------
# 基于 OCR 数字计算指标信号
# ---------------------------------------------------------------------------
def ocr_ma_signal(ma1, ma2):
    if ma1 is None or ma2 is None or ma1 == 0:
        return None
    if ma1 > ma2:
        return {"score": 40, "desc": f"✅ 均线多头 MA1={ma1:.2f} > MA2={ma2:.2f}（真实数值）",
                "ma_bull": True, "data_ok": True, "trend_hint": "up", "src": "ocr"}
    return {"score": 8, "desc": f"❌ 均线空头 MA1={ma1:.2f} < MA2={ma2:.2f}（真实数值）",
            "ma_bull": False, "data_ok": True, "trend_hint": "down", "src": "ocr"}


def ocr_vol_signal(vol, vol_ma):
    if vol is None or vol_ma is None or vol_ma <= 0:
        return None
    ratio = vol / vol_ma
    if ratio >= 1.45:
        return {"score": 25, "desc": f"✅ 明显放量 {ratio:.1f}倍（真实 {int(vol)} / MA{int(vol_ma)}）",
                "vol_strong": True, "src": "ocr", "ratio": round(ratio, 2)}
    if ratio >= 1.1:
        return {"score": 22, "desc": f"🟢 温和放量 {ratio:.1f}倍（真实数值）", "vol_strong": True, "src": "ocr", "ratio": round(ratio, 2)}
    if ratio >= 0.72:
        return {"score": 15, "desc": f"🟡 成交量与均量持平 {ratio:.1f}倍", "vol_strong": False, "src": "ocr", "ratio": round(ratio, 2)}
    return {"score": 8, "desc": f"❌ 缩量 {ratio:.1f}倍（真实数值）", "vol_strong": False, "src": "ocr", "ratio": round(ratio, 2)}


def ocr_macd_signal(diff, dea, macd):
    if diff is None or dea is None:
        return None
    bull = diff >= dea
    hist_pos = macd is None or macd >= 0
    tag = f"DIFF={diff:.4f} DEA={dea:.4f}"
    if bull and hist_pos:
        return {"score": 35, "desc": f"✅ MACD多头 {tag}（真实数值）", "macd_bull": True, "data_ok": True, "src": "ocr"}
    if bull:
        return {"score": 28, "desc": f"🟢 DIFF上穿DEA，柱转正初期 {tag}", "macd_bull": True, "data_ok": True, "src": "ocr"}
    if not hist_pos:
        return {"score": 8, "desc": f"❌ MACD空头 {tag}（真实数值）", "macd_bull": False, "data_ok": True, "src": "ocr"}
    return {"score": 12, "desc": f"🟠 DIFF在DEA下方但柱体仍正，动能转弱 {tag}", "macd_bull": False, "data_ok": True, "src": "ocr"}


# ---------------------------------------------------------------------------
# 像素Y → 价格映射（用当前K线高/低价 + 其像素位置）
# ---------------------------------------------------------------------------
def build_price_mapper(ohc, k_list):
    if not k_list or not ohc or ohc.get("high") is None or ohc.get("low") is None:
        return None, None
    last = k_list[-1]
    high_y = last["y"]
    low_y = last["y"] + last["h"]
    span = low_y - high_y
    hp, lp = float(ohc["high"]), float(ohc["low"])
    if span < 8 or hp == lp:
        return None, None
    slope = (lp - hp) / span

    def y2price(y):
        if y is None:
            return None
        return round(hp + (y - high_y) * slope, 1)

    return y2price, {"high": hp, "low": lp, "slope": round(slope, 4)}
