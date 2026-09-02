"""اختبارات حماية البيانات الخاصة أثناء التعلم الجماعي."""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def mk(tmp, name):
    import ai.living_mesh as lm
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    return lm.LivingMeshNode(node_id=name, host="127.0.0.1", port=0)


def test_reject_raw_data_in_payload():
    from ai.private_federated_learning import PrivateFederatedLearning, PrivacyViolation
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "n0")
        pfl = PrivateFederatedLearning(n)
        try:
            pfl.sanitize_outgoing({"partial_weights": [0.1], "raw_data": [[1, 2, 3]]})
            assert False, "should raise"
        except PrivacyViolation as e:
            assert "raw_data" in str(e)
        print("✅ rejects raw_data field")


def test_protect_update_no_raw_flag():
    from ai.private_federated_learning import PrivateFederatedLearning
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "n1")
        pfl = PrivateFederatedLearning(n, clip_norm=1.0, noise_multiplier=0.1)
        big = [10.0, 0.0, 0.0]  # norm=10 → clip to 1
        out = pfl.protect_update(big, add_noise=False)
        assert out["raw_data_included"] is False
        norm = sum(v*v for v in out["update"]) ** 0.5
        assert abs(norm - 1.0) < 1e-5, norm
        print("✅ clip works, no raw flag")


def test_secure_aggregate_masks_cancel():
    from ai.private_federated_learning import PrivateFederatedLearning
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = mk(tmp, "A"), mk(tmp, "B"), mk(tmp, "C")
        peers = ["A", "B", "C"]
        round_id = "rnd_test"
        shares = []
        true_updates = []
        for node, w0 in zip((a, b, c), ([1.0, 1.0], [3.0, 3.0], [5.0, 5.0])):
            pfl = PrivateFederatedLearning(node, noise_multiplier=0.0)
            # force known update via protect without noise then mask
            prot = pfl.protect_update(w0, add_noise=False)
            true_updates.append(prot["update"])
            mask = pfl.generate_pairwise_masks(round_id, node.node_id, peers, dim=2)
            masked = pfl.mask_update(prot["update"], mask)
            shares.append({"masked_update": masked, "raw_data_included": False, "final_loss": 0.1})
        pfl_a = PrivateFederatedLearning(a, noise_multiplier=0.0)
        agg = pfl_a.aggregate_shares(shares)
        assert agg["ok"]
        expected = [(true_updates[0][i] + true_updates[1][i] + true_updates[2][i]) / 3 for i in range(2)]
        for i in range(2):
            assert abs(agg["partial_weights"][i] - expected[i]) < 1e-5, (agg, expected)
        print("✅ secure aggregate masks cancel", agg["partial_weights"])


def test_local_train_never_exports_samples():
    from ai.private_federated_learning import PrivateFederatedLearning
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "train")
        pfl = PrivateFederatedLearning(n)
        out = pfl.local_private_train_step(seed_weights=[0.2]*4, steps=2)
        assert "raw_data" not in out
        assert out["update_meta"]["raw_data_included"] is False
        assert "partial_weights" in out
        print("✅ local train exports weights only")


def test_private_round_with_vcen():
    from ai.private_federated_learning import PrivateFederatedLearning
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    with tempfile.TemporaryDirectory() as tmp:
        a, b, v = mk(tmp, "wa"), mk(tmp, "wb"), mk(tmp, "ver")
        peers = ["wa", "wb"]
        rid = "fl_priv_1"
        shares = []
        for node in (a, b):
            pfl = PrivateFederatedLearning(node, noise_multiplier=0.01)
            shares.append(pfl.build_private_share(rid, peers, seed_weights=[0.1]*4, steps=2))
        pfl_a = PrivateFederatedLearning(a)
        va = VerifiableCognitiveNet(a, quorum=1, require_independent=True)
        vv = VerifiableCognitiveNet(v, quorum=1, require_independent=True)
        out = pfl_a.private_round_to_vcen_claim(va, shares, verifier_vcens=[vv])
        assert out["ok"] is True, out
        assert out["aggregate"]["raw_data_included"] is False
        assert out["verdict"]["accepted"] is True
        print("✅ private aggregate → VCEN model_update accepted")


def test_aggregate_drops_share_with_raw():
    from ai.private_federated_learning import PrivateFederatedLearning
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "agg")
        pfl = PrivateFederatedLearning(n)
        shares = [
            {"masked_update": [1.0, 2.0], "raw_data_included": False},
            {"masked_update": [3.0, 4.0], "samples": [[0, 1]], "raw_data_included": False},
        ]
        # samples key triggers violation → drop
        agg = pfl.aggregate_shares(shares)
        assert agg["ok"] and agg["n_shares"] == 1
        print("✅ drops shares containing samples key")


if __name__ == "__main__":
    test_reject_raw_data_in_payload()
    test_protect_update_no_raw_flag()
    test_secure_aggregate_masks_cancel()
    test_local_train_never_exports_samples()
    test_private_round_with_vcen()
    test_aggregate_drops_share_with_raw()
    print("🏆 Private FL tests passed")
