import requests
from config import GDELT_BASE, HTTP_TIMEOUT

def gdelt_get(params):
    r = requests.get(GDELT_BASE, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()