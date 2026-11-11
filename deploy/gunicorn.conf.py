"""Optional self-hosted WSGI configuration. Vercel manages its own workers."""
import os

bind = os.environ.get("BIND", "0.0.0.0:8080")
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
threads = 2
timeout = 30
graceful_timeout = 30
preload_app = True
accesslog = None  # Do not record request paths, query strings, or client addresses.
errorlog = "-"
loglevel = "warning"
forwarded_allow_ips = ""  # Trust no externally supplied forwarding headers by default.
