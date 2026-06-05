# Multi-stage build: Astro frontend + nginx
# Build:  docker build -t agent-nginx -f deploy/nginx.Dockerfile .
# Run:    docker run -p 80:80 agent-nginx

# Stage 1 — Build Astro frontend
FROM node:22-alpine AS builder

WORKDIR /app
COPY ui/package.json ui/pnpm-lock.yaml ui/pnpm-workspace.yaml ./
RUN npm install -g pnpm && CI=true pnpm install --frozen-lockfile
COPY ui/ .
RUN CI=true pnpm run build

# Stage 2 — nginx
FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx config
COPY deploy/nginx-docker.conf /etc/nginx/conf.d/agent.conf

# Copy built frontend from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
