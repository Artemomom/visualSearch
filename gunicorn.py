import os

bind = '0.0.0.0:5000'
workers = int(os.environ.get('NUM_WORKERS', 1))
