#!/usr/bin/env python3
"""Static file server with HTTP Range support (needed for smooth video seeking on iOS/Safari)."""
import os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
PORT = int(sys.argv[2] if len(sys.argv) > 2 else 8080)

MIME = {'.mp4': 'video/mp4', '.html': 'text/html; charset=utf-8', '.jpg': 'image/jpeg',
        '.png': 'image/png', '.json': 'application/json', '.wav': 'audio/wav',
        '.css': 'text/css', '.js': 'application/javascript'}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _resolve(self):
        p = unquote(self.path.split('?')[0]).lstrip('/')
        full = os.path.abspath(os.path.join(ROOT, p))
        if not full.startswith(ROOT):
            return None
        if os.path.isdir(full):
            full = os.path.join(full, 'player.html')
        return full if os.path.isfile(full) else None

    def do_HEAD(self):
        self._send(head_only=True)

    def do_GET(self):
        self._send(head_only=False)

    def _send(self, head_only=False):
        path = self._resolve()
        if not path:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        ctype = MIME.get(ext, 'application/octet-stream')
        start, end = 0, size - 1
        status = 200
        rng = self.headers.get('Range')
        if rng:
            m = re.match(r'bytes=(\d*)-(\d*)', rng.strip())
            if m:
                a, b = m.group(1), m.group(2)
                if a == '' and b:                      # suffix range
                    start = max(0, size - int(b)); end = size - 1
                else:
                    start = int(a or 0)
                    if b:
                        end = min(int(b), size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.send_header('Access-Control-Allow-Origin', '*')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head_only:
            return
        with open(path, 'rb') as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(256 * 1024, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


if __name__ == '__main__':
    print(f'serving {ROOT} on 0.0.0.0:{PORT} (range enabled)', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
