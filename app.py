import logging
import os

from service import create_app

logging.basicConfig(level=logging.INFO)

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('DEBUG', 'false') == 'true',
    )
