web: gunicorn app:create_app()
worker: python -m celery -A app.celery worker --loglevel=debug --concurrency=1 --max-tasks-per-child=50 