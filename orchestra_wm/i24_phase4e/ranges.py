"""Bounded authenticated HTTP ranges for standard-library ZIP inspection.

Every response must match the exact requested interval and immutable total size.
No redirects, credential logging, authentication retries, or full-file fallback.
"""
import io
import os
import urllib.error
import urllib.request


class SourceBlocked(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class DriveRanges(io.RawIOBase):
    def __init__(self, file_id, size, *, opener=None, block_bytes=8*1024*1024):
        self.size = int(size)
        self.position = 0
        self.block_bytes = block_bytes
        self.cache_start = 0
        self.cache = b''
        self.url = 'https://www.googleapis.com/drive/v3/files/' + file_id + '?alt=media'
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.requests = 0
        self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2): raise ValueError('Invalid seek origin')
        position = (0, self.position, self.size)[whence] + offset
        if position < 0: raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            raise SourceBlocked('Unbounded remote read forbidden')
        size = min(size, max(0, self.size-self.position))
        out = bytearray()
        while len(out) < size:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                first = self.position
                last = min(self.size-1, first + self.block_bytes-1)
                token = os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
                if not token: raise SourceBlocked('Missing secure source token binding')
                req = urllib.request.Request(self.url, headers={
                    'Authorization': 'Bearer '+token, 'Accept-Encoding': 'identity',
                    'Range': f'bytes={first}-{last}'})
                try:
                    with self.opener.open(req, timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{self.size}':
                            raise SourceBlocked('Server did not honor exact HTTP range')
                        if response.headers.get('Content-Encoding', 'identity') != 'identity':
                            raise SourceBlocked('Encoded range response rejected')
                        data = response.read(last-first+2)
                        if len(data) != last-first+1:
                            raise SourceBlocked('Truncated or oversized range')
                except urllib.error.HTTPError as e:
                    status=e.code; e.close()
                    raise SourceBlocked(f'Source HTTP {status}; no retry; preserve partial work') from None
                except urllib.error.URLError:
                    raise SourceBlocked('Source transport failure; preserve partial work') from None
                self.cache_start, self.cache = first, data
                self.requests += 1
                self.transferred += len(data)
            offset = self.position-self.cache_start
            chunk = self.cache[offset:offset+size-len(out)]
            out.extend(chunk)
            self.position += len(chunk)
        return bytes(out)
