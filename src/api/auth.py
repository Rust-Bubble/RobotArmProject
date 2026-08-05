# src/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import timedelta
from src.api.dependencies import get_current_user
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User
from src.api.schemas import UserRegister, UserLogin, Token, UserOut, RefreshTokenRequest
from src.api.auth_utils import (
    verify_password, get_password_hash,
    create_access_token, create_refresh_token,
    decode_token, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    async with db_manager.session() as session:
        # 检查用户名或邮箱是否已存在
        stmt = select(User).where(
            (User.username == user_data.username) | (User.email == user_data.email)
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()
        if existing:
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
        return new_user

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    async with db_manager.session() as session:
        stmt = select(User).where(User.username == user_data.username)
        result = await session.execute(stmt)
        user = result.scalars().first()
        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )
        access_token = create_access_token({"sub": str(user.id), "username": user.username})
        refresh_token = create_refresh_token({"sub": str(user.id), "username": user.username})
        return {"access_token": access_token, "refresh_token": refresh_token}

@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    payload = decode_token(request.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 refresh token"
        )
    # 重新生成 access 和 refresh
    new_access = create_access_token({"sub": payload["sub"], "username": payload["username"]})
    new_refresh = create_refresh_token({"sub": payload["sub"], "username": payload["username"]})
    return {"access_token": new_access, "refresh_token": new_refresh}

@router.get("/me", response_model=UserOut)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user