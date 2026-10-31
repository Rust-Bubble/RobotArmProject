# -*- coding: utf-8 -*-
"""异步辅助工具。

机械臂动作与工具执行都是同步阻塞调用（驱动、串口、硬件），需要放进线程执行；
执行期间 ArmController 会把运行状态写入 status_hub（见 arm_control/status.py）。
这里把「线程里执行阻塞调用」和「把状态更新转成事件流」封装为一个可复用的生成器，
避免 Orchestrator 与 WebSocket 端点各写一份队列转发逻辑。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable


async def run_with_status_events(
    func: Callable[..., Any],
    holder: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[dict]:
    """在线程中执行阻塞调用，并把执行期间的机械臂状态逐个 yield 出去。

    产出的事件形如 ``{"type": "robot_status", "state": ..., "phase": ...}``。
    函数返回值写入 ``holder["result"]``，调用方在迭代结束后读取。
    """
    from src.qiming.arm_control.status import status_hub

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    token = status_hub.subscribe(
        lambda snapshot: loop.call_soon_threadsafe(queue.put_nowait, snapshot))

    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {getter, task}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield {"type": "robot_status", **getter.result()}
                continue
            # 调用已结束：取消这次等待，收尾后再补发残留状态
            getter.cancel()
            break

        holder["result"] = await task
        while not queue.empty():
            yield {"type": "robot_status", **queue.get_nowait()}
    finally:
        status_hub.unsubscribe(token)
