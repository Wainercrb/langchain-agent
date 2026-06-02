# Docker Setup for LangChain Agent

This directory contains Docker configuration for running the entire LangChain Agent application (frontend + backend + databases) in containers.

## Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- External PostgreSQL database (managed service)
- Supabase account (for vector storage with pgvector)
- API keys: Google, OpenAI

## Quick Start

### 1. Set up environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your:
- `SUPABASE_URL` and `SUPABASE_KEY` (or use local Supabase)
- `GOOGLE_API_KEY` (for embeddings)
- `OPENAI_API_KEY` (for LLM)

### 2. Start all services (Production)

```bash
docker-compose up -d
```

Services will be available at:
- **Frontend**: http://localhost:4321
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database (PostgreSQL)**: localhost:5432



## Services

### Backend (FastAPI)
- **Port**: 8000
- **Dockerfile**: `agent/Dockerfile`
- **Health Check**: GET `/v1/health`
- **Docs**: GET `/docs` (Swagger UI)

### Frontend (Astro)
- **Port**: 4321
- **Dockerfile**: `ui/Dockerfile`
- **Build Tool**: Vite + Astro

### External Services (Third-party)
- **PostgreSQL**: Your external database host (configure in `.env`)
- **Supabase**: Cloud-hosted pgvector for embeddings

## Common Commands

```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Run specific service
docker-compose up backend frontend

# Rebuild image
docker-compose build backend
docker-compose build frontend

# Remove all containers and volumes
docker-compose down -v

# Access database directly
docker-compose exec postgres psql -U postgres -d langchain_agent

# Run migrations
docker-compose exec backend alembic upgrade head

# Execute command in running container
docker-compose exec backend python -m pytest tests/

# Stop all services
docker-compose stop

# Start stopped services
docker-compose start
```

## Network Architecture

Only 2 local containers communicate with external third-party services:

```
┌──────────────────────────────────────┐
│   Docker Bridge Network              │
│   (langchain-network)                │
│                                      │
│  ┌──────────────┐  ┌──────────────┐ │
│  │  Frontend    │  │   Backend    │ │
│  │  :4321       ├─►│   :8000      │ │
│  └──────────────┘  └──────┬───────┘ │
│                           │         │
└───────────────────────────┼─────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   PostgreSQL         Supabase            Google/OpenAI
   (External)      (Vector Store)         (APIs)
```

## Troubleshooting

### Port already in use
```bash
# Find and stop containers using ports
lsof -i :8000  # Backend
lsof -i :4321  # Frontend

docker-compose down
```

### Backend can't reach external database
- Check `DATABASE_URL` in `.env` file
- Verify PostgreSQL server is running and accessible
- Test connection: `psql $DATABASE_URL`

### Frontend can't reach backend
- Check backend is running: `curl http://localhost:8000/v1/health`
- Check CORS is configured in backend
- Check `VITE_API_URL` environment variable in frontend

## Environment Variables

Create `.env` file with your external service credentials:

```env
# External PostgreSQL Database
DATABASE_URL=postgresql://user:password@your-db-host:5432/langchain_agent

# Supabase (Vector Storage)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret

# Google APIs
GOOGLE_API_KEY=your-key

# OpenAI
OPENAI_API_KEY=your-key

# App Config
LOG_LEVEL=INFO
DEBUG=false
```

## Production Considerations

For production deployment:

1. **Use environment-specific compose files**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
   ```

2. **Use a reverse proxy** (Nginx, Traefik)

3. **Set up proper secrets management** (Docker Secrets, HashiCorp Vault)

4. **Use managed databases** (AWS RDS, Google Cloud SQL)

5. **Enable container logging** (ELK stack, Datadog, New Relic)

6. **Set resource limits**
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '0.5'
             memory: 512M
   ```

7. **Use health checks** (already configured)

## Architecture Decisions

### Multi-stage builds
- Reduces final image size
- Separates build dependencies from runtime
- Backend: ~150MB, Frontend: ~200MB

### Network isolation
- All services communicate via named Docker network
- No exposed ports between services (except to host)

### Health checks
- Backend: Checks `/v1/health` endpoint every 30s
- Frontend: Checks HTTP response every 30s
- Database: Checks `pg_isready` every 10s

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices for Python Docker Images](https://docs.docker.com/language/python/build-images/)
- [Astro Deployment](https://docs.astro.build/en/guides/deploy/)
