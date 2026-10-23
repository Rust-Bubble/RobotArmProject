// 登录/注册相关 REST 请求，走 vite 代理转发到 FastAPI /auth

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

const TOKEN_KEY = "qiming_access_token";

/** 把后端返回的错误转成可读的中文提示（含 422 校验错误的字段定位）。 */
async function readError(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === "string" && detail) return detail;
  // 422 时 FastAPI 返回校验错误数组，取第一条转成人话
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    const field = Array.isArray(first?.loc)
      ? String(first.loc[first.loc.length - 1])
      : "";
    if (field === "email") return "邮箱格式不正确（需包含 @，例如 user@example.com）";
    if (first?.type === "missing") return `缺少必填字段：${field}`;
    return `字段 ${field} 校验失败：${first?.msg ?? "格式不正确"}`;
  }
  return fallback;
}

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
    throw new Error(await readError(res, "登录失败"));
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
    throw new Error(await readError(res, "注册失败"));
  }
}
