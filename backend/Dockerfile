FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: Redis, Postgres, Java (for Neo4j), Supervisor, and build tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    software-properties-common \
    supervisor \
    redis-server \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Add PostgreSQL 15 repository and install
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && echo "deb http://apt.postgresql.org/pub/repos/apt/ bullseye-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y postgresql-15 postgresql-client-15 \
    && rm -rf /var/lib/apt/lists/*

# Install Neo4j 5
ENV NEO4J_HOME=/opt/neo4j
RUN wget -qO- https://dist.neo4j.org/neo4j-community-5.18.1-unix.tar.gz | tar -xz -C /opt/ \
    && mv /opt/neo4j-community-* /opt/neo4j \
    && sed -i 's/#dbms.default_listen_address=0.0.0.0/dbms.default_listen_address=0.0.0.0/' /opt/neo4j/conf/neo4j.conf

# Setup User (Hugging Face Spaces runs as user 1000 by default)
RUN useradd -m -u 1000 user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Create directories and set permissions for user 1000
RUN mkdir -p /app/backend/data /var/log/supervisor /var/run/supervisor /var/lib/postgresql/data \
    && chown -R 1000:1000 /app /var/log/supervisor /var/run/supervisor /var/lib/postgresql /var/run/postgresql /opt/neo4j /etc/supervisor

USER 1000
WORKDIR /app

# Install Python requirements
COPY --chown=1000:1000 requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application files
COPY --chown=1000:1000 . /app/backend
RUN chmod +x /app/backend/start.sh

# Environment variables for the application
ENV HOST=0.0.0.0
ENV PORT=7860
ENV POSTGRES_USER=test
ENV POSTGRES_PASSWORD=test
ENV POSTGRES_DB=test
ENV DATABASE_URL=postgresql+asyncpg://test:test@localhost:5432/test
ENV REDIS_URL=redis://localhost:6379/0
ENV NEO4J_URL=bolt://localhost:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=test

EXPOSE 7860
CMD ["/app/backend/start.sh"]
