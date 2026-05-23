#!/bin/bash
set -e

# Initialize PostgreSQL database if it doesn't exist
if [ ! -d "/var/lib/postgresql/data/PG_VERSION" ]; then
    echo "Initializing PostgreSQL..."
    /usr/lib/postgresql/15/bin/initdb -D /var/lib/postgresql/data
    
    # Start postgres temporarily to create user and db
    /usr/lib/postgresql/15/bin/pg_ctl -D /var/lib/postgresql/data -w start
    
    # Create the user and database (using default test credentials or env vars)
    psql -c "CREATE USER ${POSTGRES_USER:-test} WITH PASSWORD '${POSTGRES_PASSWORD:-test}';"
    psql -c "CREATE DATABASE ${POSTGRES_DB:-test} OWNER ${POSTGRES_USER:-test};"
    
    /usr/lib/postgresql/15/bin/pg_ctl -D /var/lib/postgresql/data -w stop
fi

# Initialize Neo4j password if needed
if [ ! -f "/opt/neo4j/data/dbms/auth" ]; then
    echo "Setting initial Neo4j password..."
    /opt/neo4j/bin/neo4j-admin dbms set-initial-password "${NEO4J_PASSWORD:-test}"
fi

# Run database migrations
echo "Starting supervisord to manage all processes..."
exec /usr/bin/supervisord -c /app/backend/supervisord.conf
