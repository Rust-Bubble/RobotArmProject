import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: {
      // REST 接口（登录/注册）转发到 FastAPI
      "/auth": "http://127.0.0.1:8000",
      // 对话 WebSocket 转发到 FastAPI
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
