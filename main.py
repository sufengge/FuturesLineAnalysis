# =============================================================================
# main.py   K线截图分析工具【表格化重构版】
# 单窗口：捕获桌面 → 分析 → 以表格+概率条+强制决策展示结果
# 已移除 RSI；结果用表格列出，方便直接查看指标、概率和建议。
# =============================================================================
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
from utils import capture_desktop, cv2_to_pil, resize_pil_image
from kline_analyzer import analyze_kline_image


class FuturesAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K线截图分析工具【表格优化版】")
        self.root.geometry("1060x820")
        self.root.minsize(1000, 760)
        self.img_cv = None
        self.tk_img = None
        self.last_feat = None

        self._build_toolbar()
        self._build_main_panel()
        self._build_prob_panel()
        self._build_forced_panel()
        self._build_advice_panel()
        self.status_var.set("就绪：点击【捕获桌面K线图】后分析。")

    # ---------------- 顶部按钮 ----------------
    def _build_toolbar(self):
        bar = tk.Frame(self.root)
        bar.pack(pady=6, fill=tk.X)
        self.btn_capture = tk.Button(bar, text="① 捕获桌面K线图", command=self.on_capture,
                                     width=18, height=2, bg="#4a90d9", fg="white")
        self.btn_capture.pack(side=tk.LEFT, padx=8)
        self.btn_analyze = tk.Button(bar, text="② 开始分析", command=self.on_analyze,
                                     width=16, height=2, bg="#5cb85c", fg="white")
        self.btn_analyze.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar()
        tk.Label(bar, textvariable=self.status_var, fg="#666", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=12)

    # ---------------- 主面板：左图 + 右表 ----------------
    def _build_main_panel(self):
        main = tk.Frame(self.root)
        main.pack(pady=4, fill=tk.BOTH, expand=True)

        # 左：图片预览
        img_frame = tk.LabelFrame(main, text=" 截图预览（红=做多止损 / 蓝=做空止损） ", font=("Microsoft YaHei", 9))
        img_frame.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)
        self.img_label = tk.Label(img_frame, text="点击「捕获桌面K线图」后此处显示截图",
                                  bg="#222", fg="#aaa", width=52, height=24)
        self.img_label.pack(padx=6, pady=6, fill=tk.BOTH, expand=True)

        # 右：指标结果表
        table_frame = tk.LabelFrame(main, text=" 指标结果表 ", font=("Microsoft YaHei", 9))
        table_frame.pack(side=tk.RIGHT, padx=8, fill=tk.BOTH, expand=True)
        cols = ("score", "desc")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="tree headings",
                                 height=9)
        self.tree.heading("#0", text="指标")
        self.tree.column("#0", width=110, anchor="w")
        self.tree.heading("score", text="评分")
        self.tree.column("score", width=70, anchor="center")
        self.tree.heading("desc", text="分析结论 / 说明")
        self.tree.column("desc", width=230, anchor="w")
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.tag_configure("bull", foreground="#0a7d2c")
        self.tree.tag_configure("bear", foreground="#c00000")
        self.tree.tag_configure("mid", foreground="#b8860b")
        self.tree.tag_configure("est", foreground="#888")

    # ---------------- 概率条 ----------------
    def _build_prob_panel(self):
        p = tk.LabelFrame(self.root, text=" 概率预判 ", font=("Microsoft YaHei", 9))
        p.pack(pady=4, fill=tk.X, padx=8)
        row = tk.Frame(p)
        row.pack(pady=6)
        self.lbl_up = tk.Label(row, text="上涨  --", font=("Microsoft YaHei", 16, "bold"),
                               fg="#0a7d2c", bg="#eaf7ec", width=14, pady=6)
        self.lbl_up.pack(side=tk.LEFT, padx=8)
        self.lbl_side = tk.Label(row, text="震荡  --", font=("Microsoft YaHei", 16, "bold"),
                                 fg="#b8860b", bg="#fdf7e3", width=14, pady=6)
        self.lbl_side.pack(side=tk.LEFT, padx=8)
        self.lbl_down = tk.Label(row, text="下跌  --", font=("Microsoft YaHei", 16, "bold"),
                                 fg="#c00000", bg="#fdeaea", width=14, pady=6)
        self.lbl_down.pack(side=tk.LEFT, padx=8)
        self.lbl_score = tk.Label(p, text="综合得分：-- / 100   数据可信度：--",
                                  font=("Microsoft YaHei", 10))
        self.lbl_score.pack(pady=(0, 4))

    # ---------------- 强制决策区 ----------------
    def _build_forced_panel(self):
        f = tk.LabelFrame(self.root, text=" 若当前必须开仓，应买涨还是买跌？ ", font=("Microsoft YaHei", 9))
        f.pack(pady=4, fill=tk.X, padx=8)
        self.lbl_forced = tk.Label(f, text="尚未分析", font=("Microsoft YaHei", 15, "bold"),
                                   fg="#333", bg="#f0f0f0", pady=6)
        self.lbl_forced.pack(fill=tk.X, padx=8, pady=4)
        self.lbl_forced_note = tk.Label(f, text="", font=("Microsoft YaHei", 9), fg="#555")
        self.lbl_forced_note.pack(fill=tk.X, padx=8, pady=(0, 4))

    # ---------------- 建议 + 风控 ----------------
    def _build_advice_panel(self):
        a = tk.LabelFrame(self.root, text=" 操作建议 & 风控 ", font=("Microsoft YaHei", 9))
        a.pack(pady=4, fill=tk.BOTH, expand=True, padx=8)
        self.txt_advice = tk.Text(a, height=6, font=("Microsoft YaHei", 10), wrap="word",
                                  state="disabled", bg="#fbfbfb")
        self.txt_advice.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ---------------- 捕获 ----------------
    def on_capture(self):
        self.status_var.set("正在捕获桌面…（窗口会暂时最小化）")
        self.root.update_idletasks()
        self.root.withdraw()
        self.root.after(500, self._do_capture)

    def _do_capture(self):
        try:
            self.img_cv = capture_desktop()
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("错误", f"截图失败：{e}")
            self.status_var.set("截图失败，请重试")
            return
        self.root.deiconify()
        self.update_screenshot(self.img_cv)
        self.last_feat = None
        self._reset_results("✅ 桌面截图成功！点击【② 开始分析】进行识别")
        self.status_var.set("截图完成，请点击【开始分析】。")

    def update_screenshot(self, img):
        img_pil = cv2_to_pil(img)
        img_pil = resize_pil_image(img_pil, 500, 380)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.img_label.config(image=self.tk_img, text="")
        self.img_label.image = self.tk_img

    # ---------------- 分析 ----------------
    def on_analyze(self):
        if self.img_cv is None:
            messagebox.showwarning(title="警告", message="请先捕获桌面K线图!")
            return
        self.status_var.set("正在分析：均线 / 成交量 / MACD / 趋势…（已移除RSI）")
        self._reset_results("正在分析K线形态，请稍候…")
        threading.Thread(target=self._analysis_task, daemon=True).start()

    def _analysis_task(self):
        try:
            feat, img_draw = analyze_kline_image(self.img_cv)
            self.root.after(0, self.update_screenshot, img_draw)
            self.root.after(0, self._render_result, feat)
            self.root.after(0, lambda: self.status_var.set("分析完成。"))
        except Exception as e:
            err = f"分析出错：{e}"
            self.root.after(0, lambda: self._reset_results(err))
            self.root.after(0, lambda: self.status_var.set("分析异常，请查看下方信息"))

    # ---------------- 渲染结果到表格 ----------------
    def _render_result(self, feat):
        self.last_feat = feat
        if "error" in feat:
            self._reset_results(f"⚠️ 识别失败：{feat['error']}")
            return
        # 清空表格
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for row in feat["rows"]:
            name, desc = row["name"], row["desc"]
            sc = row["score"]
            disp = f"{sc:+d}" if sc != 0 else "0"
            tag = "mid"
            if sc > 0:
                tag = "bull"
            elif sc < 0:
                tag = "bear"
            if row.get("tag") == "估算":
                tag = "est"
                desc = desc + " (估算)"
            self.tree.insert("", "end", text=name, values=(disp, desc), tags=(tag,))

        # 概率
        self.lbl_up.config(text=f"上涨 {feat['prob_up']}%")
        self.lbl_side.config(text=f"震荡 {feat['prob_side']}%")
        self.lbl_down.config(text=f"下跌 {feat['prob_down']}%")
        self.lbl_score.config(text=f"综合得分：{feat['total_score']} / 100   数据可信度：{feat['data_conf']}")

        # 强制决策
        f = feat["forced"]
        conf = f["confidence"]
        if f["decision"] == "买涨":
            fg, bg = "#0a7d2c", "#eaf7ec"
        else:
            fg, bg = "#c00000", "#fdeaea"
        self.lbl_forced.config(text=f"若必须开仓 → 建议【{f['decision']}】  方向置信度 {conf}%",
                               fg=fg, bg=bg)
        self.lbl_forced_note.config(text=f["note"] + feat.get("k_note", ""))

        # 建议文本
        lines = []
        lines.append(f"【操作建议】{self._action_text(feat['action'])}")
        lines.append("")
        for r in feat["reasons"]:
            lines.append("  • " + r)
        lines.append("")
        lines.append("【强制决策入场参考】若必须下单：")
        fref = feat["forced"]
        if fref.get("entry_ref") is not None:
            lines.append(f"    入场参考：当前K线收盘像素 Y={fref['entry_ref']}（图像像素，非真实价格）")
        if fref.get("stop_long_ref") is not None and fref["decision"] == "买涨":
            lines.append(f"    止损参考：Y={fref['stop_long_ref']}（近期低点下方）")
        if fref.get("stop_short_ref") is not None and fref["decision"] == "买跌":
            lines.append(f"    止损参考：Y={fref['stop_short_ref']}（近期高点上方）")
        lines.append("")
        lines.append("⚠️ 免责声明：本程序为图像识别技术分析，仅供学习研究，不构成任何交易建议。"
                     "期货杠杆高、风险巨大，盈亏由交易者自行承担。每一笔务必带止损。")
        self.txt_advice.config(state="normal")
        self.txt_advice.delete(1.0, tk.END)
        self.txt_advice.insert(tk.END, "\n".join(lines))
        self.txt_advice.config(state="disabled")

    def _action_text(self, action):
        return {"买涨": "📈 综合偏多，可轻仓买涨（做多）",
                "买跌": "📉 综合偏空，可轻仓买跌（做空）",
                "观望": "⏸ 信号矛盾，建议观望不进场"}.get(action, "观望")

    def _reset_results(self, msg):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.lbl_up.config(text="上涨  --")
        self.lbl_side.config(text="震荡  --")
        self.lbl_down.config(text="下跌  --")
        self.lbl_score.config(text="综合得分：-- / 100   数据可信度：--")
        self.lbl_forced.config(text="尚未分析", fg="#333", bg="#f0f0f0")
        self.lbl_forced_note.config(text="")
        self.txt_advice.config(state="normal")
        self.txt_advice.delete(1.0, tk.END)
        self.txt_advice.insert(tk.END, msg)
        self.txt_advice.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = FuturesAnalysisApp(root)
    root.mainloop()
