# exports every tinker checkpoint as lora-only sampler weights, uploads them to hf, verifies sha256, then deletes them on tinker.
# uv run -m scripts.migrate_tinker_to_hf --output_dir /workspace/migrate --repo_id sohv/lasr --workers 6 --delete

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tinker
from huggingface_hub import HfApi, hf_hub_download
from safetensors.numpy import load_file

LOGGER = logging.getLogger(__name__)
EXPORT_TTL_SECONDS = 3 * 86400


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def list_checkpoints(rest) -> list[dict]:
    out = []
    while True:
        r = rest.list_user_checkpoints(limit=100, offset=len(out)).result()
        out += [c.model_dump(mode="json") for c in r.checkpoints]
        if not r.checkpoints or len(out) >= r.cursor.total_count:
            return out


def run_and_slot(checkpoint_id: str) -> tuple[str, str]:
    """(run name, folder under the run) for a checkpoint id like weights/<run>_step100 or sampler_weights/<run>."""
    kind, name = checkpoint_id.split("/", 1)
    if kind == "sampler_weights":
        return name, ""
    run, _, step = name.rpartition("_step")
    if run and step.isdigit():
        return run, f"step{step}"
    return name, "final_training"


def export_and_fetch(service, rest, ckpt: dict, dest: Path) -> dict:
    """Sampler-weights archive of one checkpoint unpacked into dest; training checkpoints are exported first."""
    source = ckpt["tinker_path"]
    export = None
    if ckpt["checkpoint_type"] == "training":
        name = ckpt["checkpoint_id"].split("/", 1)[1] + "_export"
        export = service.create_training_client_from_state(source).save_weights_for_sampler(name, ttl_seconds=EXPORT_TTL_SECONDS).result().path
    url = rest.get_checkpoint_archive_url_from_tinker_path(export or source).result().url
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "archive.tar"
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    adapter = next(dest.rglob("adapter_model.safetensors"))
    if adapter.parent != dest:
        for f in adapter.parent.iterdir():
            f.rename(dest / f.name)
    files = sorted(p.name for p in dest.iterdir() if p.is_file())
    record = {"source": source, "export": export, "checkpoint_type": ckpt["checkpoint_type"], "time": ckpt["time"],
              "source_size_bytes": ckpt["size_bytes"], "files": files, "sha256": sha256(dest / "adapter_model.safetensors")}
    (dest / "tinker_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    LOGGER.info(f"fetched {source} -> {dest} {record['sha256'][:12]}")
    return record


def remote_sha(api: HfApi, repo_id: str, path: str) -> str | None:
    info = api.get_paths_info(repo_id, [path], expand=True)
    return info[0].lfs.sha256 if info and getattr(info[0], "lfs", None) else None


def tensors_equal(hf_file: str, local_file: Path) -> bool:
    """Same lora tensors up to the renames the earlier hf uploads applied (unembed -> lm_head, language_model segment)."""
    hf, local = load_file(hf_file), load_file(str(local_file))
    renamed = {k.replace("model.unembed_tokens", "lm_head").replace("model.model.layers", "model.model.language_model.layers"): k for k in local}
    if set(hf) != set(local) and set(hf) != set(renamed):
        return False
    names = {k: k for k in local} if set(hf) == set(local) else renamed
    return all(np.array_equal(hf[k], local[names[k]]) for k in hf)


def migrate_run(run: str, ckpts: list[dict], args, service, rest, api: HfApi, manifest) -> None:
    stage = Path(args.output_dir) / "stage" / run
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda c: (c, export_and_fetch(service, rest, c, stage / (run_and_slot(c["checkpoint_id"])[1] or "_root"))), ckpts))
    for c, r in records:
        r["slot"] = run_and_slot(c["checkpoint_id"])[1]
        r["hf_path"] = f"adapters/{run}/{r['slot']}".rstrip("/")
        r["verify"] = "sha256"
    root = next((r for c, r in records if c["checkpoint_type"] == "sampler"), None)
    existing = remote_sha(api, args.repo_id, f"adapters/{run}/adapter_model.safetensors")
    if root and existing:
        if existing == root["sha256"]:
            root["verify"] = "sha256, already on hf"
        elif tensors_equal(hf_hub_download(args.repo_id, f"adapters/{run}/adapter_model.safetensors", local_dir=str(stage / "_hf")), stage / "_root" / "adapter_model.safetensors"):
            root["verify"] = "tensors equal to the renamed copy already on hf"
        else:
            LOGGER.warning(f"{run}: hf root adapter has different tensors from the tinker final sampler; uploading it as tinker_final_sampler")
            root.update(slot="tinker_final_sampler", hf_path=f"adapters/{run}/tinker_final_sampler")
            (stage / "_root").rename(stage / "tinker_final_sampler")
        shutil.rmtree(stage / "_hf", ignore_errors=True)
        if root["verify"] != "sha256":
            shutil.rmtree(stage / "_root")
    for c, r in records:
        # the final training checkpoint holds the same lora as the final sampler; keep one copy
        if r["slot"] == "final_training" and root and r["sha256"] == root["sha256"]:
            r.update(hf_path=root["hf_path"], verify=f"same sha256 as the final sampler ({root['verify']})")
            shutil.rmtree(stage / "final_training")
    if (stage / "_root").exists():
        for f in (stage / "_root").iterdir():
            f.rename(stage / f.name)
        (stage / "_root").rmdir()
    if any(stage.rglob("adapter_model.safetensors")):
        api.upload_folder(repo_id=args.repo_id, folder_path=str(stage), path_in_repo=f"adapters/{run}",
                          commit_message=f"Tinker migration: {run} ({len(records)} checkpoints, lora only)")
    for c, r in records:
        r["hf_sha256"] = remote_sha(api, args.repo_id, f"{r['hf_path']}/adapter_model.safetensors")
        r["verified"] = r["hf_sha256"] == r["sha256"] or (r["verify"] != "sha256" and r["hf_sha256"] is not None)
        if not r["verified"]:
            LOGGER.error(f"{c['tinker_path']}: hf sha {r['hf_sha256']} != local {r['sha256']}, not deleting")
        elif args.delete:
            for path in filter(None, (r["source"], r["export"])):
                rest.delete_checkpoint_from_tinker_path(path).result()
            r["deleted_on_tinker"] = True
        manifest.write(json.dumps({"id": c["checkpoint_id"], "run": run, **r}) + "\n")
        manifest.flush()
    shutil.rmtree(stage)
    LOGGER.info(f"{run}: {sum(r['verified'] for _, r in records)}/{len(records)} verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")], force=True)
    for name in ("httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    api = HfApi()
    ckpts = list_checkpoints(rest)
    # leftovers from an earlier attempt of this script are re-exported from their source, not migrated
    ckpts = [c for c in ckpts if not c["checkpoint_id"].endswith("_export")]
    by_run = defaultdict(list)
    for c in ckpts:
        by_run[run_and_slot(c["checkpoint_id"])[0]].append(c)
    runs = args.runs or sorted(by_run)
    LOGGER.info(f"{len(ckpts)} checkpoints in {len(by_run)} runs; migrating {runs}")
    started = time.time()
    with (out / "manifest.jsonl").open("a") as manifest:
        for run in runs:
            migrate_run(run, by_run[run], args, service, rest, api, manifest)
    print(f"manifest saved to {out / 'manifest.jsonl'} ({round((time.time() - started) / 60)} min)")


if __name__ == "__main__":
    main()
