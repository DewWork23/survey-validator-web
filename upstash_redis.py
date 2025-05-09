import os
import requests

UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
    "Content-Type": "application/json"
}

def redis_command(command):
    resp = requests.post(
        UPSTASH_REDIS_REST_URL,
        headers=HEADERS,
        json={"command": command}
    )
    resp.raise_for_status()
    return resp.json()

def redis_set(key, value):
    return redis_command(["SET", key, value])

def redis_get(key):
    return redis_command(["GET", key]) 