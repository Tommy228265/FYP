#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机一键入口：运行本文件即可启动 face_app（RealSense 人脸）并常驻 launcher（供树莓派 POST /launch）。

用法（在 conda realsense 环境中）:
  cd FYP 目录
  python fyp_launcher.py

环境变量（可选）:
  RADAR_PI_BASE             树莓派 HTTP 根地址（雷达 + 可选摄像头流）
  USE_PI_CAMERA=1           视频从树莓派 /camera/rgb 拉流，本机不占用 RealSense
  FYP_DEFAULT_RADAR_PI_BASE 覆盖内置默认树莓派地址（默认 http://10.245.232.43:5000）
  FYP_LAUNCHER_PORT         默认 8787
  FYP_LAUNCHER_TOKEN        树莓派请求 /launch 时的 X-FYP-Token
  FYP_FACEAPP_CHECK         检测 face_app 是否就绪的 URL
  FYP_SKIP_FACE_APP_BOOT=1  仅启动 launcher，不自动拉起 face_app（调试用）
  FYP_EXIT_KILL_FACE=0     退出 launcher 时不结束由本程序拉起的 face_app（默认 1=会一起结束）
"""

import atexit
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
LAUNCHER_PORT = int(os.environ.get("FYP_LAUNCHER_PORT", "8787"))
TOKEN = os.environ.get("FYP_LAUNCHER_TOKEN", "").strip()
FACE_APP_URL = os.environ.get("FYP_FACEAPP_CHECK", "http://127.0.0.1:5000/api/status")

# 与树莓派 shumeipai 监听端口一致；换网络时可改环境变量 FYP_DEFAULT_RADAR_PI_BASE
_BUILTIN_DEFAULT_RADAR = "http://10.245.232.43:5000"
DEFAULT_RADAR_PI_BASE = (
    os.environ.get("FYP_DEFAULT_RADAR_PI_BASE") or _BUILTIN_DEFAULT_RADAR
).strip()

launcher_app = Flask(__name__)

# 仅当本 launcher 通过 _popen_face_app 启动 face_app 时记录 PID，退出时一并结束
_spawned_face_pid = None


def _face_app_running():
    try:
        urllib.request.urlopen(FACE_APP_URL, timeout=1.5)
        return True
    except Exception:
        return False


def _merge_env_for_face_app():
    env = os.environ.copy()
    if not (env.get("RADAR_PI_BASE") or "").strip():
        env["RADAR_PI_BASE"] = DEFAULT_RADAR_PI_BASE
        print("[INFO] 未检测到 RADAR_PI_BASE，已为 face_app 注入默认: %s" % env["RADAR_PI_BASE"])
        print("       （可在本终端先执行 $env:RADAR_PI_BASE=\"...\" 覆盖）")
    return env


def _popen_face_app():
    script = ROOT / "face_app.py"
    if not script.is_file():
        raise FileNotFoundError("找不到 face_app.py: %s" % script)
    env = _merge_env_for_face_app()
    kwargs = {"cwd": str(ROOT), "env": env}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True
    kwargs["stdout"] = subprocess.DEVNULL
    kwargs["stderr"] = subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, str(script)], **kwargs)
    global _spawned_face_pid
    _spawned_face_pid = proc.pid
    return proc


def _terminate_spawned_face_app():
    """退出 launcher 时结束由本进程启动的 face_app（含子进程）。"""
    global _spawned_face_pid
    if not _spawned_face_pid:
        return
    if os.environ.get("FYP_EXIT_KILL_FACE", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print("[INFO] 已设置 FYP_EXIT_KILL_FACE=0，保留 face_app 进程")
        return
    pid = _spawned_face_pid
    _spawned_face_pid = None
    print("[INFO] 正在结束本程序启动的 face_app（PID %s）..." % pid)
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        else:
            import signal as _sig

            os.kill(pid, _sig.SIGTERM)
            time.sleep(1.5)
            try:
                os.kill(pid, 0)
                os.kill(pid, _sig.SIGKILL)
            except OSError:
                pass
    except Exception as e:
        print("[WARN] 结束 face_app 时出现问题: %s" % e)


def _wait_face_app(timeout_sec=120, step_sec=1.0):
    deadline = time.time() + timeout_sec
    last_report = 0.0
    while time.time() < deadline:
        if _face_app_running():
            return True
        now = time.time()
        if now - last_report >= 12.0:
            waited = int(now - (deadline - timeout_sec))
            print(
                "[INFO] 仍在等待 face_app 监听 %s …（已等待约 %ss，首次启动会在后台加载深度学习模型）"
                % (FACE_APP_URL, waited)
            )
            last_report = now
        time.sleep(step_sec)
    return False


def ensure_face_app_at_boot():
    """本进程启动时：若未禁止则后台拉起 face_app 并等待就绪。"""
    if os.environ.get("FYP_SKIP_FACE_APP_BOOT", "").strip() in ("1", "true", "yes"):
        print("[INFO] 已设置 FYP_SKIP_FACE_APP_BOOT，跳过自动启动 face_app")
        return
    if _face_app_running():
        print("[OK] face_app 已在运行  (%s)" % FACE_APP_URL)
        return
    print("[INFO] 正在启动 face_app.py（与本 launcher 使用同一 Python: %s）" % sys.executable)
    try:
        _popen_face_app()
    except Exception as e:
        print("[ERROR] 无法启动 face_app: %s" % e)
        return
    if _wait_face_app():
        print("[OK] face_app 已就绪  浏览器: http://127.0.0.1:5000")
    else:
        print(
            "[WARN] %s 秒内未检测到 face_app。" % 120
            + " 请在 FYP 目录手动运行: python face_app.py 查看报错。"
        )


@launcher_app.route("/health")
def health():
    return jsonify({"ok": True, "face_app": _face_app_running()})


@launcher_app.route("/launch", methods=["POST"])
def launch():
    if TOKEN:
        if request.headers.get("X-FYP-Token") != TOKEN:
            return jsonify({"ok": False, "error": "未授权"}), 403
    if _face_app_running():
        return jsonify({"ok": True, "message": "face_app 已在运行", "started": False})

    script = ROOT / "face_app.py"
    if not script.is_file():
        return jsonify({"ok": False, "error": "找不到 face_app.py"}), 500

    try:
        _popen_face_app()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    if _wait_face_app(timeout_sec=90):
        return jsonify({"ok": True, "message": "face_app 已启动", "started": True})
    return jsonify(
        {
            "ok": False,
            "error": "已尝试启动但 90 秒内未检测到 face_app，请在本机手动运行 python face_app.py 查看报错",
            "started": True,
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("FYP 上位机一键启动")
    print("  Python: %s" % sys.executable)
    print("  工程目录: %s" % ROOT)
    print("=" * 60)
    ensure_face_app_at_boot()
    print("-" * 60)
    print("launcher 监听 0.0.0.0:%s   POST /launch  （树莓派可唤醒备用）" % LAUNCHER_PORT)
    print("检查 face_app: %s" % FACE_APP_URL)
    if TOKEN:
        print("已启用令牌: 树莓派设置相同 FYP_LAUNCHER_TOKEN")
    print("按 Ctrl+C 退出 launcher（将同时结束由本程序启动的 face_app）")
    print("=" * 60)
    atexit.register(_terminate_spawned_face_app)

    try:
        launcher_app.run(host="0.0.0.0", port=LAUNCHER_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print()
    finally:
        _terminate_spawned_face_app()
