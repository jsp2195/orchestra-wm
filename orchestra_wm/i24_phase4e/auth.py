"""Drive bearer selection. Secrets are supplied only by the execution environment."""
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


_lock = threading.Lock()
_cached = None
_expires_at = 0.0


def access_token():
    """Mint before expiry when a refresh grant is configured; never log tokens."""
    global _cached, _expires_at
    keys = ('GOOGLE_DRIVE_REFRESH_TOKEN', 'GOOGLE_DRIVE_CLIENT_ID',
            'GOOGLE_DRIVE_CLIENT_SECRET')
    values = [os.environ.get(k) for k in keys]
    if not any(values):
        return os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
    if not all(values):
        raise RuntimeError('Incomplete secure Drive refresh configuration')
    with _lock:
        if _cached and time.monotonic() < _expires_at:
            return _cached
        body = urllib.parse.urlencode(dict(refresh_token=values[0],
                                           client_id=values[1], client_secret=values[2],
                                           grant_type='refresh_token')).encode()
        req = urllib.request.Request('https://oauth2.googleapis.com/token', body,
                                     {'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise RuntimeError(f'Drive refresh HTTP {status}; preserve partial work') from None
        except urllib.error.URLError:
            raise RuntimeError('Drive refresh transport failure; preserve partial work') from None
        token = result.get('access_token')
        lifetime = result.get('expires_in')
        if not isinstance(token, str) or not token or not isinstance(lifetime, int) or lifetime <= 60:
            raise RuntimeError('Invalid Drive refresh response; preserve partial work')
        _cached = token
        _expires_at = time.monotonic() + lifetime - 60
        return token
