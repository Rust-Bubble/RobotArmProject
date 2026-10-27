# src/api/auth.py
"""用户认证端点：注册 / 登录 / Token 刷新 / 当前用户信息。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from src.api.dependencies import get_current_user
from src.api.docs import ERROR_RESPONSES, ErrorResponse
from src.api.schemas import RefreshTokenRequest, Token, UserLogin, UserOut, UserRegister
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册新用户",
    description="用户名或邮箱已被注册时返回 400。",
    responses={400: {"model": ErrorResponse, "description": "用户名或邮箱已被注册"}, **ERROR_RESPONSES},
)
async def register(user_data: UserRegister):
    async with db_manager.session() as session:
        # 检查用户名或邮箱是否已存在
        stmt = select(User).where(
            (User.username == user_data.username) | (User.email == user_data.email)
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()
        if existing:
            logger.info("注册失败：用户名或邮箱已被注册（%s）", user_data.username)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名或邮箱已被注册"
            )
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=get_password_hash(user_data.password)
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        logger.info("新用户注册成功：id=%s username=%s", new_user.id, new_user.username)
        return new_user


@router.post(
    "/login",
    response_model=Token,
    summary="登录获取 Token",
    description="校验用户名密码，签发 access / refresh token。",
    responses={401: {"model": ErrorResponse, "description": "用户名或密码错误"}, **ERROR_RESPONSES},
)
async def login(user_data: UserLogin):
    async with db_manager.session() as session:
        stmt = select(User).where(User.username == user_data.username)
        result = await session.execute(stmt)
        user = result.scalars().first()
        if not user or not verify_password(user_data.password, user.hashed_password):
            logger.info("登录失败：用户名或密码错误（%s）", user_data.username)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )
        access_token = create_access_token({"sub": str(user.id), "username": user.username})
        refresh_token = create_refresh_token({"sub": str(user.id), "username": user.username})
        logger.info("用户 %s 登录成功", user.username)
        return {"access_token": access_token, "refresh_token": refresh_token}


@router.post(
    "/refresh",
    response_model=Token,
    summary="刷新 Token",
    description="用 refresh token 换发新的 access / refresh token。",
    responses={401: {"model": ErrorResponse, "description": "无效的 refresh token"}, **ERROR_RESPONSES},
)
async def refresh_token(request: RefreshTokenRequest):
    payload = decode_token(request.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 refresh token"
        )
    sub, username = payload.get("sub"), payload.get("username")
    if not sub or not username:
        # 旧版本签发的 token 可能缺字段，避免 KeyError 直接 500
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 refresh token"
        )
    # 重新生成 access 和 refresh
    new_access = create_access_token({"sub": sub, "username": username})
    new_refresh = create_refresh_token({"sub": sub, "username": username})
    logger.info("用户 %s 的 Token 已刷新", username)
    return {"access_token": new_access, "refresh_token": new_refresh}


@router.get(
    "/me",
    response_model=UserOut,
    summary="当前用户信息",
    responses=ERROR_RESPONSES,
)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user
