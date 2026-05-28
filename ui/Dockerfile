# Multi-stage build for Astro frontend
FROM node:22-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./

# Install pnpm
RUN npm install -g pnpm

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source code
COPY . .

# Build the Astro site
RUN pnpm run build

# Final stage - serve with a simple HTTP server
FROM node:22-alpine

WORKDIR /app

# Install serve to run the static site
RUN npm install -g serve

# Copy built assets from builder
COPY --from=builder /app/dist ./dist

# Expose port
EXPOSE 4321

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:4321/ || exit 1

# Serve the static site
CMD ["serve", "-s", "dist", "-l", "4321"]
