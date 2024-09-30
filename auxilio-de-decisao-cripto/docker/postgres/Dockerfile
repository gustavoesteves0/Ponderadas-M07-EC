# Use the official Postgres image
FROM postgres:14

# Set environment variables for default database setup
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres23
ENV POSTGRES_DB=crypto_model_db

# Copy any initialization SQL scripts to the Docker image
COPY ./init.sql /docker-entrypoint-initdb.d/