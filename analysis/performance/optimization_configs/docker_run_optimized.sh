docker run -d \
  --name algotrading-optimized \
  --cpus="2.0" \
  --memory="16g" \
  --restart=unless-stopped \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  algotrading-agent:latest