"""No-op compatibility hook; public runs do not checkpoint retry prefixes."""
ACTIVE = None

def record(*args, **kwargs):
    pass

