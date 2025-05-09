import os
import requests

UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
    "Content-Type": "application/json"
} if UPSTASH_REDIS_REST_TOKEN else {}

def redis_command(command):
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        print("Warning: Upstash Redis configuration is missing")
        return None
        
    try:
        # Ensure all command arguments are strings
        command = [str(arg) for arg in command]
        resp = requests.post(
            UPSTASH_REDIS_REST_URL,
            headers=HEADERS,
            json={"command": command}
        )
        if not resp.ok:
            print(f"Upstash error response: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Upstash Redis error: {str(e)}")
        return None

def redis_set(key, value):
    result = redis_command(["SET", key, value])
    return result is not None

def redis_get(key):
    result = redis_command(["GET", key])
    return result.get('result') if result else None 