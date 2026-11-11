import pytest
import shutil
import threading
from werkzeug.serving import make_server

from healthdl.data import ROOT, load_dataset
from healthdl.inference import Predictor
from main import create_app


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture
def predictor():
    return Predictor(ROOT / "artifacts")


@pytest.fixture
def app():
    return create_app({"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def live_url(app):
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def browser_page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        try:
            browser = driver.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        except playwright.Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Install Chromium with: python -m playwright install chromium")
            raise
        context = browser.new_context()
        page = context.new_page()
        yield page
        context.close()
        browser.close()
