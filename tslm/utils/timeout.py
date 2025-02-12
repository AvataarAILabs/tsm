import signal

def process_timeout(timeout, func, args=(), kwargs={}):
    """Execute a function with a timeout using signal handlers instead of ProcessPoolExecutor"""
    
    def handler(signum, frame):
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

    # Set up the timeout handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        result = func(*args, **kwargs)
        return result
    finally:
        # Restore the old handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
