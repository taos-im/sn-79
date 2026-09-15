"""A fresh parquet OBJECT is not the same as fresh training DATA.

`collect()` decided freshness from S3 LastModified alone. An aggregator restart flushes rows that were
ingested long before the run into brand-new objects, so the data gate goes green on data the run did
not produce: A pass can report training intervals minutes into a run, counting pages that span
exchange time accumulated before it started.

Page filenames carry the data span (`<start>-<end>.parquet`, ddHHMMSS in the aggregator's own clock),
so the data gate can require the span to ADVANCE past a watermark taken at run start. Gradients keep
using write time: A .grad file genuinely is new when it appears.
"""
import datetime as _dt

import pytest

# THE SHIM IS INTERNAL AND IS NOT PUBLISHED, so this import cannot resolve in the carved tree. A bare
# import there is not a skipped test, it is a COLLECTION ERROR that takes the whole public suite down
# with it: Pytest exits rc=2 having run nothing.
_shim = pytest.importorskip(
    "runbooks.testing.sn79_acceptance_check_shim",
    reason="the acceptance shim is internal and absent from the published tree",
)
data_end_tag = _shim.data_end_tag
is_fresh_data = _shim.is_fresh_data


def _lm(sec):
    return _dt.datetime.fromtimestamp(sec, _dt.timezone.utc)


def test_end_tag_parsed_from_page_name():
    assert data_end_tag("p/data/0/2/intervals/18032836-18041312.parquet") == "18041312"
    assert data_end_tag("p/data/0/2/intervals/00001906-00002202.parquet") == "00002202"


def test_unparseable_name_is_not_counted_as_advancing():
    assert data_end_tag("p/data/0/2/intervals/latest.parquet") is None


def test_restart_flush_of_old_rows_does_not_count():
    """The measured failure: New object, old span."""
    key = "p/data/0/2/intervals/18032836-18041312.parquet"
    assert not is_fresh_data(key, _lm(2_000_000), since=1_000_000, min_end_tag="18041312")


def test_a_span_that_advances_counts():
    key = "p/data/0/2/intervals/18041312-18045000.parquet"
    assert is_fresh_data(key, _lm(2_000_000), since=1_000_000, min_end_tag="18041312")


def test_without_a_watermark_behaviour_is_unchanged():
    """Optional flag: Absent, the gate is exactly the old write-time rule."""
    key = "p/data/0/2/intervals/18032836-18041312.parquet"
    assert is_fresh_data(key, _lm(2_000_000), since=1_000_000, min_end_tag=None)
    assert not is_fresh_data(key, _lm(500_000), since=1_000_000, min_end_tag=None)
