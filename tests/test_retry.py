# -*- coding: utf-8 -*-
"""重试工具 (src/qiming/utils/retry.py) 的单元测试。"""

import asyncio

import pytest

from src.qiming.utils.retry import async_retry, retry_async


class TestRetryAsync:
    def test_success_first_try(self):
        calls = []

        async def ok():
            calls.append(1)
            return "done"

        assert asyncio.run(retry_async(ok)) == "done"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        calls = []

        async def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("闪断")
            return "recovered"

        result = asyncio.run(retry_async(flaky, retries=3, base_delay=0.01))
        assert result == "recovered"
        assert len(calls) == 3

    def test_gives_up_after_max_retries(self):
        calls = []

        async def always_fail():
            calls.append(1)
            raise TimeoutError("超时")

        with pytest.raises(TimeoutError):
            asyncio.run(retry_async(always_fail, retries=3, base_delay=0.01))
        assert len(calls) == 3

    def test_non_matching_exception_not_retried(self):
        calls = []

        async def wrong_type():
            calls.append(1)
            raise KeyError("不在重试列表里")

        with pytest.raises(KeyError):
            asyncio.run(retry_async(wrong_type, retries=3, base_delay=0.01))
        assert len(calls) == 1

    def test_custom_exception_list(self):
        calls = []

        async def locked():
            calls.append(1)
            raise KeyError("locked")

        # KeyError 在自定义重试列表里：重试耗尽后最终抛出
        with pytest.raises(KeyError):
            asyncio.run(
                retry_async(locked, retries=2, base_delay=0.01, exceptions=(KeyError,))
            )
        assert len(calls) == 2

    def test_backoff_delays_are_capped(self):
        import time

        async def always_fail():
            raise ConnectionError("x")

        start = time.perf_counter()
        with pytest.raises(ConnectionError):
            asyncio.run(retry_async(always_fail, retries=4, base_delay=0.01, max_delay=0.02))
        # 4 次尝试 3 次等待，若退避被封顶则总耗时远小于 0.01+0.02+0.04 的未封顶值
        assert time.perf_counter() - start < 0.5

    def test_on_retry_callback_invoked(self):
        seen = []

        async def fail_always():
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            asyncio.run(retry_async(
                fail_always, retries=2, base_delay=0.01,
                on_retry=lambda exc, n: seen.append(n),
            ))
        # 重试了 1 次，回调也只触发 1 次
        assert seen == [1]


class TestAsyncRetryDecorator:
    def test_decorator_retries(self):
        calls = []

        @async_retry(retries=3, base_delay=0.01)
        async def flaky(value: str) -> str:
            calls.append(value)
            if len(calls) < 2:
                raise ConnectionError("闪断")
            return value.upper()

        assert asyncio.run(flaky("abc")) == "ABC"
        assert len(calls) == 2
