import pytest

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
