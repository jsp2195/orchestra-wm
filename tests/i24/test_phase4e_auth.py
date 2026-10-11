import io
from unittest.mock import patch

import pytest

from orchestra_wm.i24_phase4e import auth


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_refresh_is_cached_and_partial_grant_fails_closed(monkeypatch):
    for key in ('GOOGLE_DRIVE_REFRESH_TOKEN', 'GOOGLE_DRIVE_CLIENT_ID',
                'GOOGLE_DRIVE_CLIENT_SECRET'):
        monkeypatch.setenv(key, 'private-test-value')
    auth._cached = None
    with patch.object(auth.urllib.request, 'urlopen',
                      return_value=Response(b'{"access_token":"fresh","expires_in":3600}')) as call:
        assert auth.access_token() == 'fresh'
        assert auth.access_token() == 'fresh'
        assert call.call_count == 1
    monkeypatch.delenv('GOOGLE_DRIVE_CLIENT_SECRET')
    with pytest.raises(RuntimeError, match='Incomplete secure Drive refresh'):
        auth.access_token()
