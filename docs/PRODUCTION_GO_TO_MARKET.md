# NSM — Production & Go-to-Market

## 1) النشر
```bash
docker compose up -d
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml --profile social up -d
```

- Streamlit: `:8501`
- FastAPI: `:5000` — ضع GPU على `nsm-api`

## 2) الفوترة
```bash
export STRIPE_SECRET_KEY=sk_live_...
export STRIPE_PRICE_PRO=price_...
```
- `POST /billing/checkout`
- بدون مفاتيح: وضع demo

## 3) MCP
```bash
pip install mcp
python mcp_server/server.py
```

## 4) السرب 24/7
```bash
python3 scripts/social_swarm_daemon.py --loop --interval 3600
```

## 5) الصحة
- `GET /health`
- `python3 scripts/train_background_worker.py --once`
