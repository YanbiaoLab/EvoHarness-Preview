import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Build straight into evoweb/static so the stdlib server ships the bundle;
// `npm run dev` proxies /api to a running `python -m evoweb`.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../static", emptyOutDir: true },
  server: {
    proxy: { "/api": "http://127.0.0.1:7861" },
  },
});
