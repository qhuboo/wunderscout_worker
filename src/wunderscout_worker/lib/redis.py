import os
import redis

redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

