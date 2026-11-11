"""Comprehensive end-to-end test suite for Never-AI v5.0.0 Sovereign upgrades."""

from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from never_ai.cortex.scene_graph import ExperienceAnchorStore, extract_scene_graph
from never_ai.memory.conversations import ConversationStore
from never_ai.server.app import app
from never_ai.updater.ota import APP_VERSION, OTAUpdater


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_scene_graph_extraction_french_rp() -> None:
    msgs = [
        {"role": "user", "content": "Nous sommes dans la chambre sous l'orage nocturne."},
        {
            "role": "never",
            "content": (
                "*Assise sur le fauteuil de cuir, je retire mon manteau sombre humide et referme la porte.*\n"
                "« Ferme à clé, on n'a plus beaucoup de temps. »"
            ),
        },
    ]
    sg = extract_scene_graph(msgs)
    assert sg["active"] is True
    assert "chambre" in sg["location"].lower() or "fauteuil" in sg["location"].lower()
    assert "manteau" in sg["attire"]
    assert len(sg["recent_actions"]) >= 1
    assert "Lieu:" in sg["summary"]


def test_experience_anchor_store_and_lora_export(tmp_path: Path) -> None:
    store = ExperienceAnchorStore(tmp_path / "sovereign_experiences.jsonl")
    rec = store.record(
        prompt="Approche-toi du comptoir et réponds franchement.",
        preferred_reply="*Je pose mes deux paumes sur le zinc froid.* « Dis-moi tout, sans filtre. »",
        kind="user_edit",
    )
    assert rec["ok"] is True
    assert rec["total_anchors"] == 1
    assert (tmp_path / "sovereign_experiences.jsonl").exists()
    assert (tmp_path / "sovereign_lora_dataset.jsonl").exists()

    hits = store.retrieve_similar("Approche-toi du comptoir et parle sans détour", top_k=2, min_sim=0.10)
    assert len(hits) == 1
    assert "zinc froid" in hits[0]["reply"]


def test_conversation_store_edit_and_fork(tmp_path: Path) -> None:
    cs = ConversationStore(tmp_path / "threads.json")
    th = cs.create_thread(title="Test RP")
    tid = th["id"]
    cs.append_exchange(
        tid,
        "Premier tour",
        {"visible_answer": "Réponse initiale"},
    )
    cs.append_exchange(
        tid,
        "Deuxième tour",
        {"visible_answer": "Suite"},
    )

    forked = cs.fork_thread(tid, up_to_index=1)
    assert forked is not None
    assert len(forked["messages"]) == 2
    assert forked["forked_from"] == tid

    updated = cs.update_thread_messages(
        tid,
        [
            {"role": "user", "content": "Premier tour édité"},
            {"role": "never", "content": "Variante choisie", "variants": ["Initial", "Variante choisie"], "activeVariant": 1},
        ],
        title="Titre Édité",
    )
    assert updated["title"] == "Titre Édité"
    assert len(updated["messages"]) == 2
    assert updated["messages"][1]["variants"][1] == "Variante choisie"


def test_fastapi_v5_endpoints_and_ota_rollback(tmp_path: Path) -> None:
    assert APP_VERSION == "5.0.0"
    client = TestClient(app)

    # 1. Create thread, PUT sync, GET scene, POST fork
    r_new = client.post("/api/threads", json={"title": "Scène Atelier"})
    assert r_new.status_code == 200
    tid = r_new.json()["thread"]["id"]

    r_put = client.put(
        f"/api/threads/{tid}",
        json={
            "title": "Scène Nocturne",
            "messages": [
                {"role": "user", "content": "Nous sommes dans le salon sous la pluie."},
                {"role": "never", "content": "*Je retire ma veste.* « Viens par ici. »", "variants": ["*Je retire ma veste.* « Viens par ici. »"]},
            ],
        },
    )
    assert r_put.status_code == 200
    assert r_put.json()["thread"]["title"] == "Scène Nocturne"

    r_scene = client.get(f"/api/threads/{tid}/scene")
    assert r_scene.status_code == 200
    assert r_scene.json()["scene"]["active"] is True

    r_fork = client.post(f"/api/threads/{tid}/fork", json={"up_to_index": 0})
    assert r_fork.status_code == 200
    assert len(r_fork.json()["thread"]["messages"]) == 1

    # 2. Experience anchor endpoint
    r_exp = client.post(
        "/api/experience/record",
        json={"prompt": "Parle vrai", "preferred_reply": "« Toujours. »", "kind": "variant_win"},
    )
    assert r_exp.status_code == 200
    assert r_exp.json()["ok"] is True

    # 3. E2EE Zero-Knowledge Vault push & pull
    vid = "test_vault_sovereign_01"
    r_vpush = client.post(
        "/api/vault/push",
        json={"vault_id": vid, "salt_b64": "AAECAw==", "iv_b64": "BAUGBw==", "ciphertext_b64": "CAkKCwwNDg8="},
    )
    assert r_vpush.status_code == 200
    assert r_vpush.json()["ok"] is True

    r_vpull = client.get(f"/api/vault/pull/{vid}")
    assert r_vpull.status_code == 200
    assert r_vpull.json()["ciphertext_b64"] == "CAkKCwwNDg8="

    # Cleanup test vault artifact
    vfp = REPO_ROOT / "memory_store" / "encrypted_vaults" / f"{vid}.json"
    vfp.unlink(missing_ok=True)

    # 4. OTA backup + rollback lifecycle
    ota_root = tmp_path / "ota_app"
    (ota_root / "src" / "never_ai").mkdir(parents=True)
    (ota_root / "src" / "never_ai" / "sample.py").write_text("X = 1\n", encoding="utf-8")
    updater = OTAUpdater(app_root=ota_root)
    bpath = updater._create_rollback_snapshot("1111111")
    assert bpath is not None and bpath.exists()
    (ota_root / "src" / "never_ai" / "sample.py").write_text("X = 999\n", encoding="utf-8")
    rb = updater.rollback_last_update()
    assert rb["rolled_back"] is True
    assert (ota_root / "src" / "never_ai" / "sample.py").read_text(encoding="utf-8") == "X = 1\n"


def test_compiled_pwa_v5_bundle_and_js_syntax() -> None:
    targets = [
        REPO_ROOT / "src" / "never_ai" / "server" / "static" / "index.html",
        REPO_ROOT / "mobile_pwa" / "index.html",
        REPO_ROOT / "docs" / "index.html",
        REPO_ROOT / "app" / "index.html",
    ]
    ref = targets[0].read_text(encoding="utf-8")
    for t in targets[1:]:
        assert t.read_text(encoding="utf-8") == ref, f"Out-of-sync PWA target: {t}"

    # Verify all 16 v5.0 modules are compiled in and no raw markers remain
    assert "/* __STYLES_V5__ */" not in ref
    assert "/* __MODULES_V5__ */" not in ref
    for symbol in (
        "const NeverAIVault =",
        "_EDGE_REFUSAL_RE",
        "Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
        "function extractSceneGraphJS(",
        "function formatRpTypography(",
        "async function generateVariantAt(",
        "function startInlineEdit(",
        "async function forkThreadAt(",
        "async function illustrateTurn(",
        "async function openVoiceMode(",
        "function openHtmlAtelier(",
        "async function runPythonAtelier(",
        "async function pushEncryptedCloudVault(",
        "async function rollbackOtaNow(",
    ):
        assert symbol in ref, f"Missing v5.0 feature symbol in compiled bundle: {symbol}"

    # Validate compiled <script> via node --check
    s_start = ref.index("<script>") + len("<script>")
    s_end = ref.rindex("</script>")
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as tf:
        tf.write(ref[s_start:s_end])
        tmp_js = Path(tf.name)
    try:
        subprocess.run(["node", "--check", str(tmp_js)], check=True)
    finally:
        tmp_js.unlink(missing_ok=True)

    # Validate Service Worker version in NeverAI-PWA.zip
    with zipfile.ZipFile(REPO_ROOT / "NeverAI-PWA.zip", "r") as zf:
        sw_txt = zf.read("NeverAI-PWA/sw.js").decode("utf-8")
        assert "5.0.0-sovereign" in sw_txt
