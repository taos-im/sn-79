"""The validator compresses each miner's state update on its batch path when compression.parallel_workers is not 0,
and a miner decompresses it with the engine the synapse names. So the batch path names the engine it used: with zstd
or zlib a miner decompresses what it was sent, and with lz4, the default, nothing changes."""
import types

import pytest

from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.utils.compress import compress, compress_batch, decompress

VERSION = 45
BOOKS = {0: {"bids": [[100.0, 1.0]], "asks": [[100.1, 2.0]]}}


def _synapse():
    return types.SimpleNamespace(books=BOOKS, pools=None, accounts={5: {"0": {"base": 1.0}}}, notices={5: []},
                                 config={"book_count": 1}, response=None, compressed=None,
                                 compression_engine=MarketSimulationStateUpdate.model_fields["compression_engine"].default)


@pytest.mark.parametrize("engine", ["zstd", "zlib", "lz4"])
def test_a_miner_decompresses_what_the_batch_path_sent_with_each_engine(engine):
    synapses = {5: _synapse()}
    books = compress(BOOKS, level=1, engine=engine, version=VERSION)
    compress_batch(synapses, [5], books, level=1, engine=engine, version=VERSION)
    assert synapses[5].compression_engine == engine
    got = decompress(synapses[5].compressed, synapses[5].compression_engine, VERSION)
    assert got["accounts"] and got["books"], got
