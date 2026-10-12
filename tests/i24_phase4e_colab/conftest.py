import pytest
from orchestra_wm.i24_phase4e_colab import storage


@pytest.fixture(autouse=True)
def small_fixture_storage(monkeypatch):
    # Tiny software fixtures must work on small /tmp filesystems. Production
    # runner retains the frozen 10 GB reserve; its rejection is tested separately.
    monkeypatch.setattr(storage,'FREE_RESERVE_BYTES',0)
