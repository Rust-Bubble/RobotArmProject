# -*- coding: utf-8 -*-
"""异步重试工具：带指数退避的重试封装。

后端里凡是「可能因外部依赖抖动而暂时失败」的调用——数据库写入被锁、
MLLM API 网络闪断等——都应通过这里重试，而不是各自手写 while 循环。

用法一（包装函数）：
    result = await retry_async(send_to_mllm, text, retries=3)

用法二（装饰器）：
    @async_retry(retries=3, exceptions=(OperationalError,))
    async def record_turn(...): ...
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 默认针对「暂时性错误」重试的异常类型；调用方可按需收窄或放宽
DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
    on_retry: Callable[[BaseException, int], None] | None = None,
    **kwargs: Any,
) -> T:
    """执行 func(*args, **kwargs)，失败后按指数退避重试。

    参数:
        retries:    最大尝试次数（含首次），<1 表示只执行一次。
        base_delay: 首次重试的等待秒数，之后每次翻倍。
        max_delay:  单次等待的上限，避免退避时间无限增长。
        exceptions: 命中这些异常才重试；其余异常直接抛出。
        on_retry:   每次重试前的回调 (异常, 第几次重试)，用于打点/上报。
    """
    delay = base_delay
    last_exc: BaseException | None = None
    for attempt in range(max(retries, 1)):
        try:
            return await func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            # 加入随机抖动，避免多个调用方同频重试再次互相挤压
            sleep_for = min(delay, max_delay) * (0.5 + random.random())
            logger.warning(
                "调用 %s 第 %d 次失败（%s: %s），%.2fs 后重试",
                getattr(func, "__name__", repr(func)), attempt + 1,
                type(exc).__name__, exc, sleep_for,
            )
            if on_retry is not None:
                on_retry(exc, attempt + 1)
            await asyncio.sleep(sleep_for)
            delay *= 2
    assert last_exc is not None
    raise last_exc


def async_retry(
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """把 retry_async 包装成装饰器，参数含义同上。"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(
                func, *args,
                retries=retries, base_delay=base_delay, max_delay=max_delay,
                exceptions=exceptions, **kwargs,
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
