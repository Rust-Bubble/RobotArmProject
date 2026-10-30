# -*- coding: utf-8 -*-
"""预设运动脚本（无视觉）的测试。"""

import pytest

from src.qiming.arm_control.controller import ArmController
from src.qiming.arm_control.motion_scripts import get_script, list_scripts
from src.qiming.arm_control.status import status_hub
from src.qiming.arm_control.workspace import validate_pose
from src.qiming.tools.builtin import create_default_registry


class RecordingDriver:
    """记录下发指令的仿真驱动。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def move_to(self, pose) -> None:
        self.calls.append(("move", tuple(round(float(v), 3) for v in pose)))

    def open_gripper(self) -> None:
        self.calls.append(("open",))

    def close_gripper(self) -> None:
        self.calls.append(("close",))


@pytest.fixture()
def driver():
    status_hub.reset()
    return RecordingDriver()


def test_all_scripts_are_valid():
    """每个脚本至少一步，移动位姿必须在安全工作空间内。"""
    scripts = list_scripts()
    assert scripts, "应至少有一个预设脚本"

    for script in scripts:
        assert script.steps, f"{script.name} 没有步骤"
        for step in script.steps:
            assert step.kind in ("move", "open", "close")
            if step.kind == "move":
                assert step.pose is not None
                validate_pose(step.pose, step.label)


def test_script_names_unique():
    names = [s.name for s in list_scripts()]
    assert len(names) == len(set(names))


def test_get_script_unknown_raises():
    with pytest.raises(KeyError):
        get_script("not_exist")


def test_run_motion_script_executes_steps(driver):
    controller = ArmController(driver, {})
    script = get_script("gripper_test")

    result = controller.run_motion_script("gripper_test")

    assert result["script"] == "gripper_test"
    assert result["steps"] == len(script.steps)
    assert [c[0] for c in driver.calls] == ["open", "close", "open"]
    assert status_hub.snapshot()["state"] == "idle"


def test_run_motion_script_moves_in_order(driver):
    controller = ArmController(driver, {})
    script = get_script("wave_demo")

    controller.run_motion_script("wave_demo")

    moves = [c[1] for c in driver.calls if c[0] == "move"]
    assert len(moves) == len(script.steps)
    assert moves[0] == tuple(step.pose for step in script.steps if step.kind == "move")[0]


def test_tools_registered_and_list_works(driver):
    reg = create_default_registry()
    assert "list_motion_scripts" in reg.names()
    assert "run_motion_script" in reg.names()

    listed = reg.execute("list_motion_scripts", {})
    assert listed["ok"] is True
    names = [item["name"] for item in listed["result"]]
    assert "gripper_test" in names and "wave_demo" in names


def test_run_tool_unknown_script_returns_error(driver):
    reg = create_default_registry()
    result = reg.execute("run_motion_script", {"name": "not_exist"})

    assert result["ok"] is False
    assert "未知的运动脚本" in result["error"]
