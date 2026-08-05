# src/api/schemas.py
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

# 注册请求
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

# 登录请求
class UserLogin(BaseModel):
    username: str
    password: str

# Token 响应
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

# 刷新 token 请求
class RefreshTokenRequest(BaseModel):
    refresh_token: str

# 用户信息响应（不包含密码）
class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True