import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // The backend's CORS_ORIGINS allows http://localhost:5173 only, so fail instead of moving to another port
  server: { port: 5173, strictPort: true },
  css: {
    preprocessorOptions: {
      // Bootstrap 5.3 still uses @import and global Sass functions; hide its deprecation warnings
      scss: { quietDeps: true, silenceDeprecations: ['import', 'global-builtin', 'color-functions', 'if-function'] },
    },
  },
})
