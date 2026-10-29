"""MinerTrainingService multi-UID routing and budget split.

One instance trains for several UIDs against a shared base model, splitting the
round budget and uploading a distinct gradient per UID. Training itself is
mocked here; these tests pin the routing/budget/registration logic.

Run: pytest GenTRX/tests/test_miner_service.py -v
"""

from unittest.mock import MagicMock


from GenTRX.src.miner_training_service import MinerTrainingService, MinerTrainingConfig


def _svc(tmp_path, uid=7, uids=None, budget=240.0):
    cfg = MinerTrainingConfig(
        uid=uid, uids=uids or [], output_dir=tmp_path, round_budget_s=budget
    )
    svc = MinerTrainingService(cfg)
    svc.model = object()          # non-None so no checkpoint download is attempted
    svc._clear_s3_cache = lambda: None
    svc._prune_gradients = lambda: None
    return svc


def test_registered_uids_seeded(tmp_path):
    svc = _svc(tmp_path, uid=7, uids=[12, 40])
    assert svc.state.registered_uids == {7, 12, 40}


def test_submit_tags_uid(tmp_path):
    svc = _svc(tmp_path)
    svc.state.training_in_progress = True  # prevent the kick from draining
    svc.submit_assignment({"round": 1, "data": [], "books": []})
    svc.submit_assignment({"round": 1, "miner_uid": 99, "data": [], "books": []})
    uids = [a["miner_uid"] for a in svc.state.pending_assignments]
    assert uids == [7, 99]  # missing → cfg.uid; explicit preserved


def test_registered_uids_split_budget_evenly(tmp_path):
    svc = _svc(tmp_path, uid=7, uids=[12, 40], budget=240.0)
    calls = []
    svc._train_one_uid = lambda uid, a, v, budget: (calls.append((uid, budget)) or 10.0)

    by_uid = {7: [{"round": 1}], 12: [{"round": 1}], 40: [{"round": 1}]}
    svc._train_all_uids_background(by_uid, target_v=0)

    assert {u for u, _ in calls} == {7, 12, 40}
    assert all(abs(b - 80.0) < 1e-6 for _, b in calls)  # 240 / 3


def test_unregistered_uid_gets_leftover_and_is_promoted(tmp_path):
    svc = _svc(tmp_path, uid=7, uids=[], budget=240.0)  # only 7 registered
    calls = []
    svc._train_one_uid = lambda uid, a, v, budget: (calls.append((uid, budget)) or 100.0)

    svc._train_all_uids_background({7: [{"round": 1}], 99: [{"round": 1}]}, target_v=0)

    budgets = dict(calls)
    assert budgets[7] == 240.0                 # sole registered → full slice
    assert abs(budgets[99] - 140.0) < 1e-6     # leftover: 240 - 100 spent
    assert 99 in svc.state.registered_uids     # promoted for next round


def test_prune_gradients_prunes_each_served_uid(tmp_path):
    cfg = MinerTrainingConfig(uid=7, uids=[12, 40], output_dir=tmp_path)
    svc = MinerTrainingService(cfg)
    store = MagicMock()
    store.prune_keep_latest.return_value = 0
    svc._write_store = store  # one shared bucket, {uid} in the path

    svc._prune_gradients()

    prefixes = {c.args[0] for c in store.prune_keep_latest.call_args_list}
    assert prefixes == {"gradients/7/", "gradients/12/", "gradients/40/"}
    for c in store.prune_keep_latest.call_args_list:
        assert c.kwargs["keep"] == cfg.keep_gradients
        assert c.kwargs["suffix"] == ".grad"


def test_prune_gradients_routes_per_uid_store(tmp_path):
    cfg = MinerTrainingConfig(uid=7, uids=[12], output_dir=tmp_path)
    svc = MinerTrainingService(cfg)
    shared, dedicated = MagicMock(), MagicMock()
    shared.prune_keep_latest.return_value = 0
    dedicated.prune_keep_latest.return_value = 0
    svc._write_store = shared
    svc._write_stores = {12: dedicated}  # uid 12 has its own bucket

    svc._prune_gradients()

    shared.prune_keep_latest.assert_called_once()
    assert shared.prune_keep_latest.call_args.args[0] == "gradients/7/"
    dedicated.prune_keep_latest.assert_called_once()
    assert dedicated.prune_keep_latest.call_args.args[0] == "gradients/12/"


def test_prune_gradients_disabled_when_keep_zero(tmp_path):
    cfg = MinerTrainingConfig(uid=7, uids=[12], output_dir=tmp_path, keep_gradients=0)
    svc = MinerTrainingService(cfg)
    store = MagicMock()
    svc._write_store = store

    svc._prune_gradients()

    store.prune_keep_latest.assert_not_called()


def _stub_training(svc, monkeypatch):
    """Real _train_one_uid, with page download and the training step stubbed."""
    import torch

    from GenTRX.src.gradient import GradientDelta, GradientMetadata

    svc._download_assignment_data = lambda a: ["page.parquet"]
    svc._build_page_loaders = lambda files: [object()]
    delta = GradientDelta(delta={"w": torch.ones(4)}, metadata=GradientMetadata(loss_before=1.0, loss_after=0.9))
    monkeypatch.setattr("GenTRX.src.distributed.train_incremental", lambda model, loaders, cfg, device: delta)


def test_trained_gradients_are_uploaded_to_shared_store(tmp_path, monkeypatch):
    svc = _svc(tmp_path, uid=7, uids=[12])
    _stub_training(svc, monkeypatch)
    store = MagicMock()
    svc._write_store = store  # one shared bucket, {uid} in the path

    svc._train_all_uids_background({7: [{"round": 3}], 12: [{"round": 3}]}, target_v=1)

    uploads = {c.kwargs["miner_uid"]: c.kwargs for c in store.put_gradient.call_args_list}
    assert set(uploads) == {7, 12}
    assert all(kw["round_id"] == 3 and kw["data"] for kw in uploads.values())
    assert svc.state.last_uploaded_round == 3


def test_upload_uses_per_uid_store_when_set(tmp_path, monkeypatch):
    svc = _svc(tmp_path, uid=7, uids=[12])
    _stub_training(svc, monkeypatch)
    shared, dedicated = MagicMock(), MagicMock()
    svc._write_store = shared
    svc._write_stores = {12: dedicated}

    svc._train_one_uid(7, [{"round": 3}], target_v=1, budget_s=10.0)
    svc._train_one_uid(12, [{"round": 3}], target_v=1, budget_s=10.0)

    assert shared.put_gradient.call_args.kwargs["miner_uid"] == 7
    assert dedicated.put_gradient.call_args.kwargs["miner_uid"] == 12


def test_failing_uid_does_not_skip_the_rest(tmp_path):
    svc = _svc(tmp_path, uid=7, uids=[12, 40])
    calls = []

    def _train(uid, assignments, target_v, budget):
        calls.append(uid)
        if uid == 12:
            raise RuntimeError("upload store unavailable")
        return 10.0

    svc._train_one_uid = _train

    svc._train_all_uids_background({7: [{"round": 1}], 12: [{"round": 1}], 40: [{"round": 1}]}, target_v=0)

    assert calls == [7, 12, 40]
    assert svc.state.training_in_progress is False
