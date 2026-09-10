// 登录/注册相关 REST 请求，走 vite 代理转发到 FastAPI /auth

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

const TOKEN_KEY = "qiming_access_token";

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function storeToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export async function login(
  username: string,
  password: string,
): Promise<TokenPair> {
  const res = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? "登录失败");
  }
  const data: TokenPair = await res.json();
  storeToken(data.access_token);
  return data;
}

export async function register(
  username: string,
  email: string,
  password: string,
): Promise<void> {
  const res = await fetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? "注册失败");
  }
}
