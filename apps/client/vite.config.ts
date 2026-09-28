import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import path from "path"


// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    // Pinned, because the API only allows this exact origin. Without
    // strictPort, Vite silently moves to 5174 when 5173 is busy, the browser
    // then gets no CORS header from the API, and the chat looks broken.
    port: 5173,
    strictPort: true,
  },
})
