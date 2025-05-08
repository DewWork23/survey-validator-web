web: gunicorn app:create_app()
worker: celery -A app.celery worker --loglevel=debug --concurrency=2 --max-tasks-per-child=50 