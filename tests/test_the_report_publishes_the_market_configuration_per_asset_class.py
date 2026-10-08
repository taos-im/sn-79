"""The dashboard's Simulation Config table shows one row per asset class: simulation_config_rows returns, per
class, its book range, its realization count and the class's own market configuration, and a single market is
one row named market."""
from types import SimpleNamespace

from taos.im.validator.report import simulation_config_rows


class _Cfg:
    def __init__(self, **kw):
        self._d = kw

    def model_dump(self):
        return dict(self._d)


def _cls(name, books, **cfg):
    return SimpleNamespace(name=name, books=books, config=_Cfg(**cfg))


def test_two_classes_give_two_rows_with_their_book_ranges_and_realizations():
    sim = SimpleNamespace(asset_classes=lambda: [
        _cls("simulation_0", list(range(0, 96)), book_count=16, init_price=300.0, min_order_size=0.25, priceDecimals=2),
        _cls("simulation_1", list(range(96, 128)), book_count=16, init_price=29.57, min_order_size=2.5364, priceDecimals=2),
    ])
    rows = simulation_config_rows(sim)
    assert list(rows) == ["simulation_0", "simulation_1"]
    assert rows["simulation_0"]["books"] == "0-95" and rows["simulation_0"]["_book_ids"] == list(range(96))
    assert rows["simulation_1"]["books"] == "96-127" and rows["simulation_1"]["_book_ids"] == list(range(96, 128))
    assert "realizations" not in rows["simulation_0"]
    assert rows["simulation_1"]["simulation_min_order_size"] == "2.5364"
    assert rows["simulation_1"]["simulation_init_price"] == "29.57"


def test_a_single_market_is_one_row_named_market():
    sim = SimpleNamespace(asset_classes=lambda: [_cls("market", list(range(0, 64)), book_count=64, init_price=300.0)])
    rows = simulation_config_rows(sim)
    assert list(rows) == ["market"] and rows["market"]["books"] == "0-63"


def test_log_dir_and_the_raw_fee_policy_dump_stay_out_of_the_row():
    """The log directory is a host path. The fee policy enters the row only through its own serializer, as the
    simulation_fee_policy_* labels the Fee Policy table reads (8 October 2026); the raw dump key does not."""
    sim = SimpleNamespace(asset_classes=lambda: [_cls("market", [0, 1], book_count=2, logDir="/x", fee_policy={"a": 1})])
    row = simulation_config_rows(sim)["market"]
    assert "simulation_logDir" not in row and "simulation_fee_policy" not in row
