import os
import posixpath
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote
import urllib

# modify this to add additional routes
ROUTES = ['/about_us', './']  # empty string for the 'default' match



class RequestHandler(SimpleHTTPRequestHandler):

    def translate_path(self, path):
        """translate path given routes"""

        # set default root to cwd
        root = os.getcwd()

        # look up routes and set root directory accordingly

        if path.startswith('/about_us'):
            # found match!
            path = '/'  # consume path up to pattern len

        # normalize path and prepend root directory
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = posixpath.normpath(unquote(path))
        words = path.split('/')
        words = filter(None, words)

        path = root
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)

        return path


if __name__ == '__main__':
    myServer = HTTPServer(('0.0.0.0', 25748), RequestHandler)
    print("Ready to begin serving files.")
    try:
        myServer.serve_forever()
    except KeyboardInterrupt:
        pass

    myServer.server_close()
    print("Exiting.")
