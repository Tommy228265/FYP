"""
仅用于本地预览 templates/index.html + static 样式，不加载 RealSense / 模型。

用法（在项目根目录）：
  pip install flask
  python preview_ui.py

浏览器打开提示的地址（默认 http://127.0.0.1:5500）。

说明：
  - /api/status、/api/radar/poll 返回简易模拟 JSON，页面可正常渲染布局；按钮调用 POST 接口也会返回成功占位。
  - 不替代 face_app.py；答辩或联调仍以 face_app 为准。
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")


def _mock_profiles():
    rows = []
    for i in range(1, 11):
        ready = i == 1
        samples = 20 if ready else 0
        rows.append(
            {
                "id": f"person{i}",
                "label": f"Person {i}",
                "ready": ready,
                "samples": samples,
            }
        )
    return rows


def _mock_status():
    profiles = _mock_profiles()
    ready_count = sum(1 for p in profiles if p["ready"])
    return {
        "sensor": {"camera_backend": "realsense"},
        "live_preview": {
            "faces": [
                {
                    "slot": 1,
                    "quality_ok": True,
                    "age": {
                        "label": "Young adult",
                        "range": "18–35",
                        "confidence": 0.82,
                    },
                },
                {
                    "slot": 2,
                    "quality_ok": False,
                    "age": None,
                },
            ],
            "updated_at": 0.0,
        },
        "profiles": profiles,
        "max_profiles": 10,
        "target_samples": 20,
        "ready_count": ready_count,
        "last_recognition": {
            "label": "Idle",
            "summary": "",
            "labels": [],
            "scores": [],
            "ages": [],
            "score": 0.0,
            "is_known": False,
            "updated_at": 0.0,
            "recognition_locked": False,
            "locked_since": 0.0,
        },
        "vitals": {
            "enabled": True,
            "error": None,
            "weights": "",
            "faces": [],
        },
        "vitals_fusion": {
            "people": [],
            "radar_channel_count": 0,
            "assignment": "depth_rank_to_bin_rank",
            "fusion_model": "",
            "depth_scene_m": None,
            "use_depth_bin_match": True,
        },
        "mode": "idle",
        "status_text": "Preview (preview_ui.py)",
        "enroll_person": None,
        "enrolling_display_name": None,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(_mock_status())


@app.route("/api/radar/poll")
def api_radar_poll():
    return jsonify({"ok": False, "error": "Preview: Pi radar not connected"})


@app.route("/api/enroll/start", methods=["POST"])
@app.route("/api/recognize/start", methods=["POST"])
@app.route("/api/profile/delete", methods=["POST"])
@app.route("/api/stop", methods=["POST"])
@app.route("/api/radar/start", methods=["POST"])
@app.route("/api/radar/stop", methods=["POST"])
def api_ok_placeholder():
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Preview: http://127.0.0.1:5500  (Ctrl+C to quit)")
    app.run(host="127.0.0.1", port=5500, debug=False)
