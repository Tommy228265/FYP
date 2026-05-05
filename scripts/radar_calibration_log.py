#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮询树莓派 shumeipai 的 /api/radar，按行打印时间戳与雷达 HR/RR（及质量），便于与参考设备手记对照。

用法（在树莓派本机，雷达服务已启动时）：
  python3 scripts/radar_calibration_log.py
  python3 scripts/radar_calibration_log.py --url http://127.0.0.1:5000 --interval 5

把终端输出复制到表格软件，另建两列手写「参考心率」「参考呼吸」即可算平均偏差。
参考设备：指夹/腕式血氧仪（心率）、人工计 30s 呼吸次数×2（呼吸）、或带连续 RR 的可穿戴。
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def fetch_radar(base: str) -> Optional[Dict[str, Any]]:
    url = base.rstrip("/") + "/api/radar"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fyp-radar-calibration-log"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"# error: {e}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Log shumeipai /api/radar HR/RR for calibration notes.")
    p.add_argument(
        "--url",
        default="http://127.0.0.1:5000",
        help="shumeipai 根地址（树莓派本机默认 127.0.0.1:5000）",
    )
    p.add_argument("--interval", type=float, default=5.0, help="采样间隔（秒）")
    p.add_argument(
        "--csv",
        action="store_true",
        help="输出逗号分隔，首行为表头，便于导入表格",
    )
    args = p.parse_args()

    print(f"# polling {args.url}/api/radar every {args.interval}s (Ctrl+C stop)", file=sys.stderr)
    if args.csv:
        print("unix_ts,iso_time,radar_hr_bpm,radar_br_bpm,radar_hq,radar_bq,is_running")
    else:
        print("# columns: time | radar_HR | radar_BR | heart_Q | breath_Q | running")

    try:
        while True:
            d = fetch_radar(args.url)
            ts = time.time()
            iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            if not isinstance(d, dict):
                hr = br = hq = bq = ""
                run = ""
            else:
                hr = d.get("heart_rate", "")
                br = d.get("breathing_rate", "")
                hq = d.get("heart_quality", "")
                bq = d.get("breathing_quality", "")
                run = d.get("is_running", "")
            if args.csv:
                print(
                    f"{ts:.0f},{iso},{hr},{br},{hq},{bq},{run}",
                    flush=True,
                )
            else:
                print(
                    f"{iso}\t{hr}\t{br}\t{hq}\t{bq}\t{run}",
                    flush=True,
                )
            time.sleep(max(0.5, float(args.interval)))
    except KeyboardInterrupt:
        print("\n# stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
