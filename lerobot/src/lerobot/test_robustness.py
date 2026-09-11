#!/usr/bin/env python3
"""
主从臂数据传输稳定性测试。
使用与 teleoperate.py 完全相同的类和函数，基于时间统计正常运行率和停机时间。

用法：
    python -m lerobot.test_robustness --test leader --duration 120 --fps 30
    python -m lerobot.test_robustness --test follower --duration 120 --fps 30
    python -m lerobot.test_robustness --test both --duration 120 --fps 30 --log-file results.csv
"""

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field

from lerobot.robots.enpei_follower.config_enpei_follower import EnpeiFollowerConfig
from lerobot.robots.enpei_follower.enpei_follower import EnpeiFollower
from lerobot.teleoperators.enpei_leader.config_enpei_leader import EnpeiLeaderConfig
from lerobot.teleoperators.enpei_leader.enpei_leader import EnpeiLeader
from lerobot.utils.robot_utils import busy_wait

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

STATS_INTERVAL = 5.0

# 从臂默认角度（与 enpei_leader SERVO_REFERENCE_POSITIONS 对应）
DEFAULT_ACTION = {
    "joint1.pos": 180.0, "joint2.pos": 90.0, "joint3.pos": 83.0,
    "joint4.pos": 210.0, "joint5.pos": 110.0, "joint6.pos": 210.0,
    "gripper.pos": 10.0,
}


@dataclass
class TestStats:
    total_cycles: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    wall_time_s: float = 0.0
    success_time_ms: float = 0.0
    time_to_first_failure: float = -1.0
    failure_times: list = field(default_factory=list)
    latencies: list = field(default_factory=list)

    @property
    def uptime_s(self) -> float:
        up = self.success_time_ms / 1000
        if self.failures == 0:
            return self.wall_time_s
        return min(up, self.wall_time_s)

    @property
    def downtime_s(self) -> float:
        return max(self.wall_time_s - self.uptime_s, 0.0)

    @property
    def uptime_pct(self) -> float:
        return self.uptime_s / self.wall_time_s * 100 if self.wall_time_s > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    @property
    def mtbf(self) -> float:
        if len(self.failure_times) < 2:
            return float("inf")
        gaps = [self.failure_times[i] - self.failure_times[i - 1]
                for i in range(1, len(self.failure_times))]
        return sum(gaps) / len(gaps)

    def record_success(self, cycle_ms: float, frame_ms: float):
        """cycle_ms: 调用耗时, frame_ms: 整个帧耗时(含busy_wait)"""
        self.total_cycles += 1
        self.latencies.append(cycle_ms)
        self.success_time_ms += frame_ms
        self.consecutive_failures = 0

    def record_failure(self, cycle_ms: float, elapsed_s: float):
        self.total_cycles += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.max_consecutive_failures = max(
            self.max_consecutive_failures, self.consecutive_failures
        )
        self.failure_times.append(elapsed_s)
        if self.time_to_first_failure < 0:
            self.time_to_first_failure = elapsed_s

    def latency_histogram(self) -> dict:
        buckets = {"<5ms": 0, "5-10ms": 0, "10-20ms": 0,
                   "20-50ms": 0, "50-100ms": 0, ">100ms": 0}
        for lat in self.latencies:
            if lat < 5:
                buckets["<5ms"] += 1
            elif lat < 10:
                buckets["5-10ms"] += 1
            elif lat < 20:
                buckets["10-20ms"] += 1
            elif lat < 50:
                buckets["20-50ms"] += 1
            elif lat < 100:
                buckets["50-100ms"] += 1
            else:
                buckets[">100ms"] += 1
        return buckets


# ---------------------------------------------------------------------------
# 输出函数
# ---------------------------------------------------------------------------

def _color(pct: float) -> str:
    if pct >= 99.9:
        return GREEN
    if pct >= 95:
        return YELLOW
    return RED


def print_periodic_stats(stats: TestStats, label: str, elapsed: float):
    c = _color(stats.uptime_pct)
    down = stats.downtime_s
    parts = [
        f"{c}[{elapsed:6.1f}s] {label}",
        f"正常运行: {stats.uptime_pct:.1f}%",
        f"循环: {stats.total_cycles}",
        f"失败: {stats.failures}",
    ]
    if down > 0:
        parts.append(f"停机: {down:.1f}s")
    parts.append(f"平均: {stats.avg_latency:.1f}ms")
    parts.append(f"最大: {stats.max_latency:.1f}ms{RESET}")
    print(" | ".join(parts))


def print_final_summary(stats: TestStats, label: str):
    c = _color(stats.uptime_pct)
    hist = stats.latency_histogram()
    up_s = stats.uptime_s
    down_s = stats.downtime_s
    wall_s = stats.wall_time_s

    print(f"\n{'=' * 60}")
    print(f"{BOLD}  {label} 测试报告{RESET}")
    print(f"{'=' * 60}")
    print(f"  总时长:                {wall_s:.1f}s")
    print(f"  正常运行时间:          {c}{up_s:.1f}s ({stats.uptime_pct:.1f}%){RESET}")
    print(f"  停机时间:              {down_s:.1f}s")
    if stats.time_to_first_failure >= 0:
        print(f"  首次故障时间:          {stats.time_to_first_failure:.1f}s")
    else:
        print(f"  首次故障时间:          无故障")
    mtbf = stats.mtbf
    print(f"  MTBF:                  {mtbf:.1f}s" if mtbf != float("inf") else f"  MTBF:                  N/A")
    print(f"  总循环数:              {stats.total_cycles}")
    print(f"  失败循环数:            {stats.failures}")
    print(f"  最大连续失败:          {stats.max_consecutive_failures}")
    print(f"  平均延迟:              {stats.avg_latency:.2f} ms（仅成功循环）")
    print(f"  最大延迟:              {stats.max_latency:.2f} ms")
    print(f"  P95延迟:               {stats.percentile(95):.2f} ms")
    print(f"  P99延迟:               {stats.percentile(99):.2f} ms")
    print(f"\n  延迟分布:")
    for bucket, count in hist.items():
        bar = "#" * min(count * 50 // max(stats.total_cycles, 1), 50)
        print(f"    {bucket:>8s}: {count:5d}  {bar}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 主臂测试
# ---------------------------------------------------------------------------

def test_leader(args) -> TestStats:
    stats = TestStats()
    print(f"\n{BOLD}开始主臂 (Leader) 串口测试{RESET}")
    print(f"  端口: {args.port}, FPS: {args.fps}, 时长: {args.duration}s\n")

    config = EnpeiLeaderConfig(port=args.port, id="test_leader")
    teleop = EnpeiLeader(config)
    teleop.current_mode = args.speed_mode
    teleop.filter_alpha = teleop.filter_config[args.speed_mode]

    try:
        teleop.connect()
    except Exception as e:
        print(f"{RED}主臂连接失败: {e}{RESET}")
        return stats

    csv_writer, csv_file = None, None
    if args.log_file:
        path = args.log_file.replace(".csv", "_leader.csv") if args.test == "both" else args.log_file
        csv_file = open(path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "cycle", "success", "latency_ms", "error"])

    start = time.perf_counter()
    last_stats_time = start

    try:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= args.duration:
                break

            loop_start = time.perf_counter()
            error_msg = ""
            success = False

            try:
                action = teleop.get_action()
                cycle_ms = (time.perf_counter() - loop_start) * 1000
                if action is None:
                    error_msg = "get_action() returned None"
                    stats.record_failure(cycle_ms, elapsed)
                    print(f"{RED}[{elapsed:6.1f}s] LEADER: {error_msg} ({cycle_ms:.0f}ms){RESET}")
                else:
                    success = True
            except Exception as e:
                cycle_ms = (time.perf_counter() - loop_start) * 1000
                error_msg = str(e)
                stats.record_failure(cycle_ms, elapsed)
                print(f"{RED}[{elapsed:6.1f}s] LEADER 异常 ({cycle_ms:.0f}ms): {error_msg}{RESET}")

            if time.perf_counter() - start >= args.duration:
                break

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / args.fps - dt_s)
            frame_ms = (time.perf_counter() - loop_start) * 1000

            if success:
                stats.record_success(cycle_ms, frame_ms)

            if csv_writer:
                ok = "1" if success else "0"
                csv_writer.writerow([f"{elapsed:.3f}", stats.total_cycles, ok, f"{cycle_ms:.2f}", error_msg])

            if time.perf_counter() - last_stats_time >= STATS_INTERVAL:
                stats.wall_time_s = time.perf_counter() - start
                print_periodic_stats(stats, "LEADER", time.perf_counter() - start)
                last_stats_time = time.perf_counter()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}主臂测试被用户中断{RESET}")
    finally:
        teleop.disconnect()
        if csv_file:
            csv_file.close()

    stats.wall_time_s = time.perf_counter() - start
    if stats.wall_time_s > args.duration:
        print(f"{YELLOW}  注意: 实际运行 {stats.wall_time_s:.1f}s 超过设定的 {args.duration}s，"
              f"因内部重试阻塞了 {stats.wall_time_s - args.duration:.1f}s{RESET}")
    print_final_summary(stats, "主臂 (LEADER)")
    return stats


# ---------------------------------------------------------------------------
# 从臂测试
# ---------------------------------------------------------------------------

def test_follower(args) -> TestStats:
    stats = TestStats()
    print(f"\n{BOLD}开始从臂 (Follower) TCP 测试{RESET}")
    print(f"  服务器: {args.ip}:{args.tcp_port}, FPS: {args.fps}, 时长: {args.duration}s\n")

    config = EnpeiFollowerConfig(
        ip_address=args.ip, port=args.tcp_port,
        id="test_follower", cameras={},
    )
    robot = EnpeiFollower(config)
    robot.current_mode = args.speed_mode
    robot.motors_max_speed = robot.speed_config[args.speed_mode]
    robot.filter_alpha = robot.filter_config[args.speed_mode]

    try:
        robot.connect()
    except Exception as e:
        print(f"{RED}从臂连接失败: {e}{RESET}")
        return stats

    # 给底层socket设置超时，防止 _recv_exact() 在连接断开时永久阻塞
    # 生产代码没有设置超时，这正是遥操作卡死的原因之一
    SOCKET_TIMEOUT = 5.0
    _original_ensure = robot.controller._ensure_connected

    def _ensure_with_timeout():
        _original_ensure()
        if robot.controller._socket:
            robot.controller._socket.settimeout(SOCKET_TIMEOUT)

    robot.controller._ensure_connected = _ensure_with_timeout
    if robot.controller._socket:
        robot.controller._socket.settimeout(SOCKET_TIMEOUT)
    print(f"  已设置 socket 超时: {SOCKET_TIMEOUT}s")

    csv_writer, csv_file = None, None
    if args.log_file:
        path = args.log_file.replace(".csv", "_follower.csv") if args.test == "both" else args.log_file
        csv_file = open(path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "cycle", "success", "latency_ms", "error", "phase"])

    last_action = dict(DEFAULT_ACTION)
    start = time.perf_counter()
    last_stats_time = start

    try:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= args.duration:
                break

            loop_start = time.perf_counter()
            error_msg = ""
            phase = "get_observation"
            success = False

            try:
                obs = robot.get_observation()
                if obs is None:
                    cycle_ms = (time.perf_counter() - loop_start) * 1000
                    error_msg = "get_observation() returned None"
                    stats.record_failure(cycle_ms, elapsed)
                    print(f"{RED}[{elapsed:6.1f}s] FOLLOWER: {error_msg} ({cycle_ms:.0f}ms){RESET}")
                else:
                    phase = "send_action"
                    last_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
                    robot.send_action(dict(last_action))
                    cycle_ms = (time.perf_counter() - loop_start) * 1000
                    success = True
            except Exception as e:
                cycle_ms = (time.perf_counter() - loop_start) * 1000
                error_msg = str(e)
                stats.record_failure(cycle_ms, elapsed)
                print(f"{RED}[{elapsed:6.1f}s] FOLLOWER {phase} 异常 ({cycle_ms:.0f}ms): {error_msg}{RESET}")

            if time.perf_counter() - start >= args.duration:
                break

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / args.fps - dt_s)
            frame_ms = (time.perf_counter() - loop_start) * 1000

            if success:
                stats.record_success(cycle_ms, frame_ms)

            if csv_writer:
                ok = "1" if success else "0"
                csv_writer.writerow([f"{elapsed:.3f}", stats.total_cycles, ok, f"{cycle_ms:.2f}", error_msg, phase])

            if time.perf_counter() - last_stats_time >= STATS_INTERVAL:
                stats.wall_time_s = time.perf_counter() - start
                print_periodic_stats(stats, "FOLLOWER", time.perf_counter() - start)
                last_stats_time = time.perf_counter()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}从臂测试被用户中断{RESET}")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        if csv_file:
            csv_file.close()

    stats.wall_time_s = time.perf_counter() - start
    if stats.wall_time_s > args.duration:
        print(f"{YELLOW}  注意: 实际运行 {stats.wall_time_s:.1f}s 超过设定的 {args.duration}s，"
              f"因内部重试阻塞了 {stats.wall_time_s - args.duration:.1f}s{RESET}")
    print_final_summary(stats, "从臂 (FOLLOWER)")
    return stats


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="主从臂数据传输稳定性测试")
    parser.add_argument("--test", required=True, choices=["leader", "follower", "both"],
                        help="测试模式")
    parser.add_argument("--duration", type=float, default=120, help="测试时长（秒）")
    parser.add_argument("--fps", type=int, default=30, help="目标频率")
    parser.add_argument("--port", default="/dev/ttyACM0", help="主臂串口端口")
    parser.add_argument("--ip", default="localhost", help="从臂服务器IP")
    parser.add_argument("--tcp-port", type=int, default=12345, help="从臂服务器TCP端口")
    parser.add_argument("--log-file", default=None, help="CSV日志文件路径")
    parser.add_argument("--speed-mode", default="record",
                        choices=["record", "inference", "teleop"], help="速度模式")
    args = parser.parse_args()

    print(f"{BOLD}主从臂数据传输稳定性测试{RESET}")
    print(f"  模式: {args.test}, 时长: {args.duration}s, FPS: {args.fps}, 速度模式: {args.speed_mode}")

    if args.test in ("leader", "both"):
        test_leader(args)
    if args.test in ("follower", "both"):
        test_follower(args)

    print("测试完成。")


if __name__ == "__main__":
    main()
