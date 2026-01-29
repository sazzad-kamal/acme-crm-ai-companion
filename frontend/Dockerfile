FROM node:20-slim AS build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

# Build arg for API URL — set in Railway service settings
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# Production server with optional basic auth
FROM node:20-slim

WORKDIR /app

RUN npm install express

COPY --from=build /app/dist ./dist
COPY frontend/server.js ./server.js

ENV PORT=3000
EXPOSE ${PORT}

CMD ["node", "server.js"]
