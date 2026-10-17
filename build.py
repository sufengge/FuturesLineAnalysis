import os
import shutil
import subprocess

def clean_old_build():
    """清理上次打包生成的 build、dist、spec 文件"""
    # 要删除的文件夹
    dirs_to_remove = ["build", "dist"]
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✅ 已删除目录: {d}")

    # 删除 .spec 打包配置文件
    for file in os.listdir("."):
        if file.endswith(".spec"):
            os.remove(file)
            print(f"✅ 已删除旧spec文件: {file}")


def run_pyinstaller():
    """执行PyInstaller打包命令"""
    # 打包命令：-F单文件，-w隐藏黑窗口，程序名KLineAnalysisTool，入口main.py
    cmd = [
        "pyinstaller",
        "-F",
        "-w",
        "-i", "app.ico",
        "-n", "KLineAnalysisTool",
        "main.py"
    ]
    print("\n🚀 开始执行打包命令：")
    print(" ".join(cmd))
    # 调用终端执行打包
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print("\n🎉 打包成功！exe文件在 dist/ 文件夹内")
    else:
        print("\n❌ 打包失败，请查看上面日志！")


if __name__ == "__main__":
    print("======= 自动打包脚本开始 =======")
    clean_old_build()
    run_pyinstaller()
