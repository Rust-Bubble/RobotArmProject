# -*- coding: utf-8 -*-
"""系统级端点与全局异常处理的集成测试。"""

from fastapi import APIRouter
from fastapi.testclient import TestClient
import pytest

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_root_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"message": "启明智手 API 服务运行中"}

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "database": "up"}

    def test_openapi_docs_available(self, client):
        # Swagger 文档自动生成：openapi.json / docs / redoc 都可访问
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    def test_openapi_contains_metadata(self, client):
        spec = client.get("/openapi.json").json()
        assert spec["info"]["title"] == "启明智手 API"
        # 标签分组与统一错误响应模型都已写入文档
        tags = {t["name"] for t in spec["tags"]}
        assert {"chat", "authentication", "system"} <= tags
        assert "ErrorResponse" in spec["components"]["schemas"]


class TestGlobalErrorHandling:
    def test_unhandled_exception_returns_structured_500(self, client):
        # 临时挂一条必然抛异常的路由，验证全局异常处理与统一错误结构
        router = APIRouter()

        @router.get("/boom")
        async def boom():
            raise RuntimeError("测试爆炸")

        app.include_router(router)
        try:
            resp = client.get("/boom")
            assert resp.status_code == 500
            body = resp.json()
            assert body["code"] == 500
            assert "服务内部错误" in body["message"]
            # 内部异常细节不应泄漏给客户端
            assert "测试爆炸" not in body["message"]
        finally:
            app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/boom"]

    def test_validation_error_structured(self, client):
        # 缺字段 + 邮箱格式错误的统一 422 结构
        resp = client.post("/auth/register", json={"username": "x"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == 422
        assert body["message"] == "请求参数不合法"
        fields = {e["field"] for e in body["detail"]}
        assert "email" in fields and "password" in fields

    def test_auth_401_structured(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401
        body = resp.json()
        assert body["code"] == 401
        assert body["message"]
