# -*- coding: utf-8 -*-
"""控制网关（联调四：坐标指令接收与稳定驱动）的测试。"""

import threading
import time

import pytest

from src.qiming.arm_control.controller import ArmController
from src.qiming.arm_control.gateway import CoordinateGateway
from src.qiming.arm_control.status import status_hub

VALID_POSE = [120.0, -120.0, 60.0, 0.0, 180.0, 90.0]
VALID_PLACE = [80.0, -220.0, 190.0, 0.0, 180.0, 90.0]


class RecordingDriver:
    """记录下发指令的仿真驱动，可注入异常与并发检测。"""

    def __init__(self) -> None:
        self.moves: list[tuple] = []
        self.gripper_calls: list[str] = []
        self.fail_on_move = False
        self.move_delay = 0.0
        self.concurrent_calls = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def move_to(self, pose) -> None:
        with self._lock:
            self.concurrent_calls += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_calls)
        try:
            if self.move_delay:
                time.sleep(self.move_delay)
            if self.fail_on_move:
                raise RuntimeError("串口写入失败")
            with self._lock:
                self.moves.append(tuple(pose))
        finally:
            with self._lock:
                self.concurrent_calls -= 1

    def open_gripper(self) -> None:
        self.gripper_calls.append("open")

    def close_gripper(self) -> None:
        self.gripper_calls.append("close")


@pytest.fixture()
def driver():
    status_hub.reset()
    return RecordingDriver()


@pytest.fixture()
def gateway(driver):
    controller = ArmController(driver, {"grasp_preset": "top_down", "place_preset": "top_down_place"})
    return CoordinateGateway(controller, default_place_pose=VALID_PLACE)


class TestCommandValidation:
    def test_move_command_executes(self, gateway, driver):
        result = gateway.submit({"action": "move", "pick_pose": VALID_POSE})

        assert result["ok"] is True
        assert result["action"] == "move"
        assert driver.moves == [tuple(VALID_POSE)]
        assert status_hub.snapshot()["state"] == "idle"

    def test_grasp_command_runs_full_flow(self, gateway, driver):
        result = gateway.submit({
            "action": "grasp",
            "pick_pose": VALID_POSE,
            "place_pose": VALID_PLACE,
            "object_name": "水杯",
        })

        assert result["ok"] is True
        assert result["object"] == "水杯"
        assert len(driver.moves) == 6  # 接近/下降/抬起/递送接近/下降/撤离
        assert driver.gripper_calls == ["open", "close", "open"]

    def test_missing_pick_pose_rejected(self, gateway, driver):
        result = gateway.submit({"action": "move"})

        assert result["ok"] is False
        assert result["rejected"] is True
        assert "pick_pose" in result["error"]
        assert driver.moves == []

    @pytest.mark.parametrize("pose", [
        [1.0, 2.0, 3.0],                       # 维度不足
        [1, 2, 3, "x", 5, 6],                  # 非数值
        [999.0, 0.0, 50.0, 0.0, 180.0, 90.0],  # 越界
        [float("nan"), 0.0, 50.0, 0.0, 180.0, 90.0],  # 非有限值
    ])
    def test_invalid_pose_rejected(self, gateway, driver, pose):
        result = gateway.submit({"action": "move", "pick_pose": pose})

        assert result["ok"] is False and result["rejected"] is True
        assert driver.moves == []

    def test_unknown_action_rejected(self, gateway, driver):
        result = gateway.submit({"action": "fly", "pick_pose": VALID_POSE})

        assert result["ok"] is False and result["rejected"] is True
        assert driver.moves == []

    def test_grasp_uses_default_place_pose(self, gateway, driver):
        result = gateway.submit({"action": "grasp", "pick_pose": VALID_POSE})

        assert result["ok"] is True
        assert any(move[2] == VALID_PLACE[2] for move in driver.moves)


class TestStability:
    def test_idempotent_by_command_id(self, gateway, driver):
        first = gateway.submit({"command_id": "cmd-1", "action": "move", "pick_pose": VALID_POSE})
        second = gateway.submit({"command_id": "cmd-1", "action": "move", "pick_pose": VALID_POSE})

        assert first["ok"] is True
        assert second.get("duplicate") is True
        assert len(driver.moves) == 1  # 只驱动一次
        assert gateway.stats.duplicate == 1

    def test_driver_failure_is_reported_and_state_reset(self, gateway, driver):
        driver.fail_on_move = True
        result = gateway.submit({"action": "move", "pick_pose": VALID_POSE})

        assert result["ok"] is False
        assert "串口写入失败" in result["error"]
        assert status_hub.snapshot()["state"] == "idle"
        assert gateway.stats.failed == 1

    def test_estop_blocks_commands_until_reset(self, driver):
        controller = ArmController(driver, {})
        gateway = CoordinateGateway(controller, default_place_pose=VALID_PLACE)

        controller.stop()
        blocked = gateway.submit({"action": "move", "pick_pose": VALID_POSE})
        assert blocked["ok"] is False and blocked["rejected"] is True
        assert driver.moves == []

        status_hub.reset()
        resumed = gateway.submit({"action": "move", "pick_pose": VALID_POSE})
        assert resumed["ok"] is True

    def test_commands_are_serialized(self, gateway, driver):
        driver.move_delay = 0.02
        threads = [
            threading.Thread(target=gateway.submit,
                             args=({"action": "move", "pick_pose": VALID_POSE},))
            for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert driver.max_concurrent == 1, "驱动被并发调用"
        assert len(driver.moves) == 5

    def test_stats_accumulate(self, gateway, driver):
        gateway.submit({"action": "move", "pick_pose": VALID_POSE})
        gateway.submit({"action": "move", "pick_pose": [1.0, 2.0]})

        assert gateway.stats.total == 2
        assert gateway.stats.succeeded == 1
        assert gateway.stats.rejected == 1
        assert gateway.stats.last_error is not None

    def test_gateway_reset_clears_cache(self, gateway, driver):
        gateway.submit({"command_id": "cmd-1", "action": "move", "pick_pose": VALID_POSE})
        gateway.reset()

        assert gateway.stats.total == 0
        again = gateway.submit({"command_id": "cmd-1", "action": "move", "pick_pose": VALID_POSE})
        assert again.get("duplicate") is None
        assert len(driver.moves) == 2
