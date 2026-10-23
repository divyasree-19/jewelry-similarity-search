# Test harness: build the static site with Vite, serve it with nginx.
# VITE_* values are compiled into the bundle, so changing VITE_API_URL means rebuilding this image.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_API_URL=http://localhost:8001
ARG VITE_MAX_UPLOAD_MB=10
ENV VITE_API_URL=$VITE_API_URL VITE_MAX_UPLOAD_MB=$VITE_MAX_UPLOAD_MB
RUN npm run build

# `docker build --target test frontend` runs the unit tests (used by scripts/run_tests.sh)
FROM build AS test
RUN npm test

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO /dev/null http://127.0.0.1/ || exit 1
