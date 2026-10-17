import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

# 从utils导入工具函数
from utils import capture_desktop, cv2_to_pil, resize_cv_image
# 导入K线分析函数
from kline_analyzer import analyze_kline_image


class FuturesAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K线截图分析工具")
        self.root.geometry("900x700")

        # 保存cv图像变量
        self.img_cv = None
        self.tk_img = None

        # ============ 界面布局 ============
        # 顶部按钮区域
        frame_top = tk.Frame(root)
        frame_top.pack(pady=5, fill=tk.X)

        self.btn_capture = tk.Button(frame_top, text="捕获桌面K线图", command=self.on_capture, width=18, height=2)
        self.btn_capture.pack(side=tk.LEFT, padx=10)

        self.btn_analyze = tk.Button(frame_top, text="开始分析K线", command=self.on_analyze, width=18, height=2)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        # 图片预览区域
        frame_img = tk.Frame(root, bg="#333333")
        frame_img.pack(pady=5)
        self.img_label = tk.Label(frame_img, text="图片预览区域", bg="black", fg="white")
        self.img_label.pack()

        # 文本输出框
        frame_text = tk.Frame(root)
        frame_text.pack(pady=5, fill=tk.BOTH, expand=True)

        scroll_bar = tk.Scrollbar(frame_text)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(frame_text, yscrollcommand=scroll_bar.set, font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scroll_bar.config(command=self.result_text.yview)

    def on_capture(self):
        """捕获桌面截图"""
        self.root.withdraw()
        self.root.after(400)
        self.img_cv = capture_desktop()
        self.root.deiconify()
        # 更新预览图
        self.update_screenshot(self.img_cv)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "✅ 桌面截图成功! 点击【开始分析K线】进行识别\n")

    def update_screenshot(self, img):
        """更新界面图片预览：调用utils工具函数完成图像转换+缩放"""
        # 调用utils：cv2 BGR -> PIL
        img_pil = cv2_to_pil(img)
        # 调用utils：缩放图片到预览尺寸 580 × 420
        img_pil = resize_cv_image(img_pil, 580, 420)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.img_label.config(image=self.tk_img, text="")
        self.img_label.image = self.tk_img

    def on_analyze(self):
        """点击分析按钮"""
        if self.img_cv is None:
            messagebox.showwarning(title="警告", message="请先捕获桌面K线图!")
            return
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在解析K线形态、均线、MACD、成交量...\n\n")
        # 后台线程执行分析，防止UI卡死
        def analysis_task():
            try:
                # 接收两个返回值：报告文本、带止损线图片
                res_text, img_draw = analyze_kline_image(self.img_cv)
                # UI更新必须放在after
                self.root.after(0, self.update_screenshot, img_draw)
                self.root.after(0, self.show_result, res_text)
            except Exception as e:
                err_msg = f"分析出错：{str(e)}"
                self.root.after(0, self.show_result, err_msg)
        threading.Thread(target=analysis_task, daemon=True).start()

    def show_result(self, text):
        """展示分析报告文本"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)


if __name__ == "__main__":
    root = tk.Tk()
    app = FuturesAnalysisApp(root)
    root.mainloop()
