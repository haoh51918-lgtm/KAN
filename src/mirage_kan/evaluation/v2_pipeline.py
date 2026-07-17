"""One-way S2a v2 development opening and five-arm Quanta orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq
import yaml

from mirage_kan.artifacts.topology import TopologyTransaction
from mirage_kan.data import PitPanel
from mirage_kan.data.pit import RAW_FIELDS, sha256_file
from mirage_kan.evaluation.v2_runner import (
    ARMS,
    EvaluationIdentityPins,
    StagedArmEvaluation,
    stage_v2_arm,
)
from mirage_kan.governance.authority import AuthorityGuard, AuthoritySuperseded
from mirage_kan.governance.implementation_lock import verify_implementation_lock
from mirage_kan.governance.openings import (
    consume_development_opening,
    consume_rebound_development_opening,
    verify_development_opening,
)
from mirage_kan.governance.mining_rebind import verify_mining_rebind_receipt
from mirage_kan.protocol import BASE_LOCK


@dataclass(frozen=True)
class _FrozenDevelopmentContext:
    root: Path
    base_lock_path: Path
    base_lock: Mapping[str, object]
    config_path: Path
    config: Mapping[str, object]
    implementation_lock_path: Path
    development_opening_path: Path
    cache_path: Path
    cache_sha256: str


@dataclass(frozen=True)
class PendingDevelopmentDecision:
    """Published five-arm evidence with the decision target still claimed."""

    workspace: Path
    topology: TopologyTransaction
    authority_guard: AuthorityGuard
    pins: EvaluationIdentityPins
    arm_manifest_sha256: Mapping[str, str]

    @property
    def decision_target(self) -> Path:
        """Return the claimed, deliberately unpublished decision directory."""
        return self.topology.targets[self.topology.top_key]

    def stage_decision(self, staging_path: Path | str):
        """Invoke the evidence-only decision assembler without publishing it."""
        from mirage_kan.evaluation.v2_decision_assembler import (
            stage_v2_decision_artifact,
        )

        try:
            return stage_v2_decision_artifact(self.workspace, staging_path)
        except AuthoritySuperseded as error:
            try:
                self.topology.terminalize(
                    {
                        "failure_class": "superseded_authority",
                        "error": str(error),
                    }
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"decision topology cleanup also failed: {cleanup_error}"
                )
            raise
        except BaseException as error:
            self.topology.terminalize(
                {
                    "failure_class": "s2a_v2_decision_assembly_failure",
                    "error": str(error),
                }
            )
            raise

    def publish_decision(self, staged_decision: object) -> Path:
        """Publish an assembled decision under both frozen authority boundaries."""
        from mirage_kan.evaluation.v2_decision_assembler import (
            StagedDecisionArtifact,
        )

        verified_staging = False
        try:
            if not isinstance(staged_decision, StagedDecisionArtifact):
                raise TypeError(
                    "decision publication requires a staged v2 decision artifact"
                )
            staging_path = staged_decision.path.resolve(strict=True)
            if (
                not staging_path.is_dir()
                or staging_path.is_symlink()
                or not staging_path.is_relative_to(self.workspace)
                or not staging_path.name.endswith(".staging")
            ):
                raise ValueError("decision staging directory is unsafe")
            manifest_path = staging_path / "manifest.json"
            if (
                not manifest_path.is_file()
                or manifest_path.is_symlink()
                or sha256_file(manifest_path) != staged_decision.manifest_sha256
                or _read_json(manifest_path, label="staged decision manifest")
                != staged_decision.manifest
            ):
                raise ValueError("staged decision manifest changed after assembly")
            verified_staging = True
            context = _load_frozen_context(self.workspace)
            verify_implementation_lock(self.workspace)
            if (
                sha256_file(context.implementation_lock_path)
                != self.pins.implementation_lock_sha256
            ):
                raise ValueError(
                    "decision publication implementation lock differs from the opening"
                )
            _verify_manifest_files(
                staging_path,
                staged_decision.manifest,
                label="decision staging",
                exact_files={"decision.json"},
            )
            recorded = staged_decision.manifest.get("evaluation_manifest_sha256")
            if recorded != dict(self.arm_manifest_sha256):
                raise ValueError(
                    "decision artifact does not bind the published five arms"
                )
            if (
                staged_decision.manifest.get("development_topology_sha256")
                != self.topology.topology_sha256
            ):
                raise ValueError("decision artifact belongs to another topology")
            publication = self.authority_guard.revalidate(
                "before_each_artifact_publication", arm=self.topology.top_key
            )
            final = self.authority_guard.revalidate("before_final_decision_publication")
            self.topology.publish_top_bundle(
                staged_decision.path,
                authority_guard=self.authority_guard,
                authority_capability=publication.capability,
                final_decision_capability=final.capability,
            )
            published_manifest = _read_json(
                self.decision_target / "manifest.json",
                label="published decision manifest",
            )
            _verify_manifest_files(
                self.decision_target,
                published_manifest,
                label="published decision",
                exact_files={"decision.json"},
            )
        except AuthoritySuperseded as error:
            try:
                self.topology.terminalize(
                    {
                        "failure_class": "superseded_authority",
                        "error": str(error),
                    }
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"decision topology cleanup also failed: {cleanup_error}"
                )
            if verified_staging:
                try:
                    _remove_staging(self.workspace, staged_decision.path)
                except BaseException as cleanup_error:
                    error.add_note(
                        f"decision staging cleanup also failed: {cleanup_error}"
                    )
            raise
        except BaseException as error:
            self.topology.terminalize(
                {
                    "failure_class": "s2a_v2_decision_publication_failure",
                    "error": str(error),
                }
            )
            if verified_staging:
                _remove_staging(self.workspace, staged_decision.path)
            raise
        return self.decision_target


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _mapping(value, label=label)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return value


def _verify_manifest_files(
    path: Path,
    manifest: Mapping[str, object],
    *,
    label: str,
    exact_files: set[str] | None = None,
) -> None:
    files = _mapping(manifest.get("files"), label=f"{label} files")
    if exact_files is not None and set(files) != exact_files:
        raise ValueError(f"{label} manifest has the wrong exact file set")
    entries = tuple(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} must contain only flat regular files")
    if {entry.name for entry in entries} != set(files) | {"manifest.json"}:
        raise ValueError(f"{label} file set differs from its manifest")
    for filename, raw_record in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{label} manifest has an unsafe filename")
        expected_bytes: int | None = None
        if isinstance(raw_record, Mapping):
            expected_sha256 = _sha256(
                raw_record.get("sha256"), label=f"{label} {filename}"
            )
            raw_bytes = raw_record.get("bytes")
            if type(raw_bytes) is not int or raw_bytes < 0:
                raise ValueError(f"{label} manifest has an invalid byte count")
            expected_bytes = raw_bytes
        else:
            expected_sha256 = _sha256(raw_record, label=f"{label} {filename}")
        file_path = path / filename
        if sha256_file(file_path) != expected_sha256:
            raise ValueError(f"{label} file hash mismatch: {filename}")
        if expected_bytes is not None and file_path.stat().st_size != expected_bytes:
            raise ValueError(f"{label} file byte count mismatch: {filename}")


def _contained_file(root: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a nonempty string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the workspace")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"{label} is not a contained regular file")
    return resolved


def _contained_future_file(root: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a nonempty string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the workspace")
    path = root / relative
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(root) or path.is_symlink():
        raise ValueError(f"{label} path escapes the workspace")
    return path


def _load_frozen_context(workspace: Path | str) -> _FrozenDevelopmentContext:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("development workspace is not a directory")
    base_lock_path = _contained_file(root, str(BASE_LOCK), label="base lock")
    base_lock = _read_json(base_lock_path, label="base lock")
    protocol = _mapping(base_lock.get("protocol"), label="base-lock protocol")
    config_path = _contained_file(root, protocol.get("path"), label="frozen config")
    if sha256_file(config_path) != protocol.get("sha256"):
        raise ValueError("frozen config hash differs from the base lock")
    try:
        config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("frozen config is not valid YAML") from error
    config = _mapping(config_value, label="frozen config")
    if config.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("frozen config protocol differs from the base lock")
    paths = _mapping(config.get("artifact_paths"), label="artifact paths")
    implementation = _contained_file(
        root, paths.get("implementation_lock"), label="implementation lock"
    )
    opening = _contained_future_file(
        root, paths.get("development_opening"), label="development opening"
    )
    data = _mapping(base_lock.get("data"), label="base-lock data")
    raw_cache = data.get("cache_path")
    if not isinstance(raw_cache, str) or not raw_cache:
        raise ValueError("base lock lacks the PIT cache path")
    cache = Path(raw_cache)
    cache = (
        cache.resolve(strict=True)
        if cache.is_absolute()
        else (root / cache).resolve(strict=True)
    )
    if not cache.is_file() or cache.is_symlink():
        raise ValueError("PIT cache must be a regular file")
    cache_sha256 = _sha256(data.get("cache_sha256"), label="PIT cache")
    return _FrozenDevelopmentContext(
        root=root,
        base_lock_path=base_lock_path,
        base_lock=base_lock,
        config_path=config_path,
        config=config,
        implementation_lock_path=implementation,
        development_opening_path=opening,
        cache_path=cache,
        cache_sha256=cache_sha256,
    )


def _verify_published_mining_topology(
    topology: TopologyTransaction,
) -> str:
    if topology.phase != "mining":
        raise ValueError("mining verification received another topology phase")
    manifests: dict[str, str] = {}
    records: dict[str, Mapping[str, object]] = {}
    for key, path in topology.targets.items():
        if (
            not path.is_dir()
            or path.is_symlink()
            or (path / ".INCOMPLETE").exists()
            or (path / "terminal_failure.json").exists()
        ):
            raise ValueError(f"mining target is not immutably published: {key}")
        manifest_path = path / "manifest.json"
        record = _read_json(manifest_path, label=f"mining manifest {key}")
        if (
            record.get("topology_sha256") != topology.topology_sha256
            or record.get("topology_key") != key
        ):
            raise ValueError(f"mining manifest ownership differs: {key}")
        manifests[key] = sha256_file(manifest_path)
        records[key] = record
    top = records[topology.top_key]
    if (
        top.get("schema_version") != "mirage_s2a_v2_staging_bundle_v1"
        or top.get("role") != "mining_top_bundle"
    ):
        raise ValueError("mining top has an invalid v2 schema or role")
    expected = {key: manifests[key] for key in topology.child_keys}
    if top.get("child_manifests") != expected:
        raise ValueError("mining top does not bind final topology child manifests")
    if top.get("child_manifest_sha256") != expected:
        raise ValueError("mining top does not bind runner child manifest identities")
    return manifests[topology.top_key]


def _pins_from_opening(
    context: _FrozenDevelopmentContext,
    development_topology: TopologyTransaction,
    opening: Mapping[str, object],
    mining_manifest_sha256: str,
) -> EvaluationIdentityPins:
    pins = _mapping(opening.get("identity_pins"), label="development identity pins")
    if pins.get("mining_manifest_sha256") != mining_manifest_sha256:
        raise ValueError("development opening names another mining top manifest")
    if opening.get("topology_sha256") not in (
        None,
        development_topology.topology_sha256,
    ):
        raise ValueError("development opening names another development topology")
    return EvaluationIdentityPins(
        base_lock_sha256=_sha256(
            pins.get("base_lock_sha256"), label="opening base lock"
        ),
        implementation_lock_sha256=_sha256(
            pins.get("implementation_lock_sha256"), label="opening implementation"
        ),
        mining_manifest_sha256=_sha256(
            pins.get("mining_manifest_sha256"), label="opening mining manifest"
        ),
        development_opening_sha256=sha256_file(context.development_opening_path),
        development_topology_sha256=development_topology.topology_sha256,
        provider_identity=_mapping(
            pins.get("provider_identity"), label="opening provider identity"
        ),
        mining_rebind_receipt_sha256=(
            None
            if pins.get("mining_rebind_receipt_sha256") is None
            else _sha256(
                pins.get("mining_rebind_receipt_sha256"),
                label="opening mining rebind receipt",
            )
        ),
    )


def _load_raw_pit_panel(context: _FrozenDevelopmentContext) -> PitPanel:
    """Rehash and load every raw PIT row while physically excluding labels."""
    observed_sha256 = sha256_file(context.cache_path)
    if observed_sha256 != context.cache_sha256:
        raise ValueError("live PIT cache hash differs from the base lock")
    schema = set(pq.ParquetFile(context.cache_path).schema_arrow.names)
    columns = ["datetime", "instrument", *RAW_FIELDS, "in_universe"]
    if "tradable" in schema:
        columns.append("tradable")
    missing = set(columns).difference(schema)
    if missing:
        raise ValueError(f"PIT cache lacks raw development columns: {sorted(missing)}")
    frame = pd.read_parquet(context.cache_path, columns=columns)
    if frame.empty or list(frame.columns) != columns:
        raise ValueError("raw PIT development panel is empty or reordered")
    return PitPanel.from_frame(
        frame,
        source_path=context.cache_path,
        source_sha256=observed_sha256,
    )


def _validate_staged_arm(root: Path, arm: str, staged: StagedArmEvaluation) -> None:
    if not isinstance(staged, StagedArmEvaluation):
        raise TypeError(f"{arm} did not return a staged arm evaluation")
    if staged.arm != arm or staged.topology_key != f"evaluation:{arm}":
        raise ValueError(f"{arm} returned another arm's staging identity")
    path = staged.staging_path.resolve(strict=True)
    if (
        not path.is_dir()
        or path.is_symlink()
        or not path.is_relative_to(root)
        or not path.name.endswith(".staging")
    ):
        raise ValueError(f"{arm} returned an unsafe staging directory")
    manifest = path / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"{arm} staging lacks a regular manifest")
    if sha256_file(manifest) != staged.manifest_sha256:
        raise ValueError(f"{arm} staging manifest changed after execution")


def _remove_staging(root: Path, path: Path | str) -> None:
    raw = Path(path)
    if not raw.exists() or raw.is_symlink():
        return
    resolved = raw.resolve(strict=True)
    if (
        resolved.is_dir()
        and resolved.is_relative_to(root)
        and resolved.name.endswith(".staging")
    ):
        shutil.rmtree(resolved)


def _stage_five_arms(
    context: _FrozenDevelopmentContext,
    panel: PitPanel,
    pins: EvaluationIdentityPins,
    guard: AuthorityGuard,
    development_capability: str,
) -> dict[str, StagedArmEvaluation]:
    completed: dict[str, StagedArmEvaluation] = {}
    for arm in ARMS:
        staged: StagedArmEvaluation | None = None
        try:
            capability = guard.revalidate(
                "before_each_scientific_or_control_arm", arm=arm
            ).capability
            staged = stage_v2_arm(
                context.root,
                arm=arm,
                panel=panel,
                pins=pins,
                authority_guard=guard,
                development_capability=development_capability,
                capability=capability,
                staging_parent=context.root,
            )
            _validate_staged_arm(context.root, arm, staged)
            completed[arm] = staged
        except BaseException as error:
            cleanup_paths: list[Path] = []
            if staged is not None:
                cleanup_paths.append(staged.staging_path)
            cleanup_paths.extend(result.staging_path for result in completed.values())
            for path in cleanup_paths:
                try:
                    _remove_staging(context.root, path)
                except BaseException as cleanup_error:
                    error.add_note(
                        "arm staging cleanup also failed for "
                        f"{path}: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
    if set(completed) != set(ARMS):
        raise RuntimeError("five-arm staging completed with an incomplete arm set")
    return completed


def run_s2a_v2_development(
    workspace: Path | str,
) -> PendingDevelopmentDecision:
    """Open development once, execute five official Quanta arms, publish children."""
    context = _load_frozen_context(workspace)
    verify_implementation_lock(context.root)
    mining_source = context.config.get("mining_source")
    if isinstance(mining_source, Mapping) and mining_source.get("mode") == (
        "verified_cross_protocol_rebind"
    ):
        binding = verify_mining_rebind_receipt(
            context.root,
            target_base_lock_path=context.base_lock_path,
            target_implementation_lock_path=context.implementation_lock_path,
        )
        source = _mapping(binding.get("source"), label="rebound mining source")
        artifacts = _mapping(
            source.get("artifacts"), label="rebound mining artifacts"
        )
        mining_manifest_sha256 = _sha256(
            _mapping(
                artifacts.get("mining_run"), label="rebound mining top"
            ).get("manifest_sha256"),
            label="rebound mining top manifest",
        )
        mining_topology = None
    else:
        mining_topology = TopologyTransaction.from_frozen_config(
            context.root, phase="mining"
        )
        mining_manifest_sha256 = _verify_published_mining_topology(mining_topology)
    development_topology = TopologyTransaction.from_frozen_config(
        context.root, phase="development"
    )
    staged: dict[str, StagedArmEvaluation] = {}
    claimed = False
    try:
        development_topology.preclaim()
        development_topology.claim_all()
        claimed = True
        guard = AuthorityGuard(context.root)
        development_receipt = guard.revalidate("before_development_opening")
        if mining_topology is None:
            opening = consume_rebound_development_opening(
                context.root,
                development_topology,
                guard,
                development_receipt.capability,
            )
        else:
            opening = consume_development_opening(
                context.root,
                mining_topology,
                development_topology,
                guard,
                development_receipt.capability,
            )
        if verify_development_opening(context.root) != opening:
            raise ValueError(
                "development opening changed immediately after consumption"
            )
        pins = _pins_from_opening(
            context,
            development_topology,
            opening,
            mining_manifest_sha256,
        )
        panel = _load_raw_pit_panel(context)
        staged = _stage_five_arms(
            context,
            panel,
            pins,
            guard,
            development_receipt.capability,
        )
        final_hashes: dict[str, str] = {}
        for arm in ARMS:
            key = f"evaluation:{arm}"
            publication = guard.revalidate("before_each_artifact_publication", arm=key)
            development_topology.publish_child(
                key,
                staged[arm].staging_path,
                authority_guard=guard,
                authority_capability=publication.capability,
            )
            final_hashes[arm] = sha256_file(
                development_topology.targets[key] / "manifest.json"
            )
        return PendingDevelopmentDecision(
            workspace=context.root,
            topology=development_topology,
            authority_guard=guard,
            pins=pins,
            arm_manifest_sha256=final_hashes,
        )
    except AuthoritySuperseded as error:
        if claimed:
            try:
                development_topology.terminalize(
                    {
                        "failure_class": "superseded_authority",
                        "error": str(error),
                    }
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"development topology cleanup also failed: {cleanup_error}"
                )
        raise
    except BaseException as error:
        if claimed:
            try:
                development_topology.terminalize(
                    {
                        "failure_class": "s2a_v2_development_failure",
                        "error": str(error),
                    }
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"development topology cleanup also failed: {cleanup_error}"
                )
        raise
    finally:
        for result in staged.values():
            _remove_staging(context.root, result.staging_path)


__all__ = ["PendingDevelopmentDecision", "run_s2a_v2_development"]
