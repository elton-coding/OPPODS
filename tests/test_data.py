from pathlib import Path

import numpy as np

from oppods.data import ChannelMemmap

DATA = Path("ziliao/data_train/H_train.npz")


def test_channel_memmap_metadata_and_read() -> None:
    data = ChannelMemmap(DATA)
    assert data.shape == (100000, 2, 2, 16, 144)
    assert data.storage_dtype == np.dtype("float16")
    batch = data.read([0, 99999])
    assert batch.shape == (2, 2, 2, 16, 144)
    assert batch.dtype == np.complex64
    assert np.isfinite(batch).all()
