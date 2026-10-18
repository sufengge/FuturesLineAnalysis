# =============================================================================
# build.py  自动打包脚本（生成独立 exe）
# 用法：在项目目录下运行  python build.py
# 产物：dist/KLineAnalysisTool.exe（单文件、无黑窗口）
# =============================================================================
import os
import shutil
import subprocess


def clean_old_build():
    """清理上次打包生成的 build、dist、spec 文件"""
    for d in ["build", "dist"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✅ 已删除目录: {d}")
    for file in os.listdir("."):
        if file.endswith(".spec"):
            os.remove(file)
            print(f"✅ 已删除旧spec文件: {file}")


def run_pyinstaller():
    """执行PyInstaller打包命令：-F单文件，-w隐藏黑窗口"""
    cmd = ["pyinstaller", "-F", "-w", "-n", "KLineAnalysisTool", "main.py"]
    # 若存在图标则加上，避免缺失导致打包失败
    if os.path.exists("app.ico"):
        cmd.insert(cmd.index("-n"), "-i")
        cmd.insert(cmd.index("-n"), "app.ico")
    print("\n🚀 开始执行打包命令：")
    print(" ".join(cmd))
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print("\n🎉 打包成功！exe文件在 dist/ 文件夹内: dist/KLineAnalysisTool.exe")
    else:
        print("\n❌ 打包失败，请查看上面日志！")


if __name__ == "__main__":
    print("======= 自动打包脚本开始 =======")
    clean_old_build()
    run_pyinstaller()
