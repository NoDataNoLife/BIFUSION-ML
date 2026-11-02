"""Job existence and atomic pending -> running/cancelled transitions."""
from redis import Redis
from app.core.config import settings


def connection():
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)


def key(job_id):
    return f"bifusion:job:{job_id}"


def register(job_id, kind):
    connection().hset(key(job_id), mapping={"kind": kind, "state": "PENDING"})


def lookup(job_id):
    return connection().hgetall(key(job_id))


def transition(job_id, expected, target):
    # Start and cancellation compete in the same Redis operation; only one wins.
    return bool(connection().eval(
        "if redis.call('HGET', KEYS[1], 'state') == ARGV[1] then "
        "redis.call('HSET', KEYS[1], 'state', ARGV[2]); return 1 end; return 0",
        1, key(job_id), expected, target,
    ))
