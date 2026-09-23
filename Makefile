.PHONY: backend frontend test docker

# 🚀 Run the FastAPI backend with auto-reload
backend:
	cd backend && PYTHONPATH=. ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 💻 Run the React Vite frontend
frontend:
	cd frontend && npm run dev

# 🧪 Run full automated test suite (15 tests)
test:
	cd backend && PYTHONPATH=. ../.venv/bin/pytest tests/ -v

# 🐳 Run with Docker Compose
docker:
	docker-compose up --build
