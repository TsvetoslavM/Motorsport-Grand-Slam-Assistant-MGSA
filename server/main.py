import logging
import uvicorn

from server.app import create_app
from server.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MGSA Server - Laptop Backend (Field Test Ready)")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print("Starting server on http://0.0.0.0:8000")
    print("CTRL+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")

