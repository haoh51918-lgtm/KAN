"""Identity-bound, topology-neutral Quanta execution for one S2a v2 arm."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zlib
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd
import torch
import yaml

from mirage_kan.artifacts.library import verify_library
from mirage_kan.artifacts.topology import TopologyTransaction
from mirage_kan.data import PitPanel
from mirage_kan.data.pit import RAW_FIELDS
from mirage_kan.data.pit import sha256_file
from mirage_kan.dsl import AstNode, evaluate
from mirage_kan.evaluation.quanta import QuantaAdapter
from mirage_kan.governance.authority import AuthorityGuard
from mirage_kan.governance.mining_rebind import verify_mining_rebind_receipt
from mirage_kan.mining.e3 import (
    PROFILE_SPECS,
    atom_manifest_sha256,
    build_profile_atom_bank,
)
from mirage_kan.mining.e3_runner import (
    BOOTSTRAP_SEED_BASE,
    TRAINING_STEPS,
    AtomPanel,
    draw_training_bootstrap,
    materialize_atom_panel,
)
from mirage_kan.mining.mlp_control import (
    LEARNING_RATE,
    MLP_SEED_BASE,
    MLPControlReceipt,
    MLPTrainingStepReceipt,
    replay_control_on_atom_panel,
)
from mirage_kan.protocol import BASE_LOCK, PROTOCOL_ID

ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)
_FACTOR_LIBRARY_ARMS = frozenset(
    {
        "kan_e3_selected",
        "typed_gp_sr_control",
        "kan_e3_permutation_control",
    }
)
_INPUT_KINDS = {
    "alpha158_replay": "official_alpha158",
    "kan_e3_selected": "verified_factor_library",
    "typed_gp_sr_control": "verified_factor_library",
    "matched_blackbox_control": "computed_factor_control",
    "kan_e3_permutation_control": "verified_factor_library",
}
_ARTIFACT_KEYS = {
    "kan_e3_selected": "kan_library",
    "typed_gp_sr_control": "gp_control_library",
    "matched_blackbox_control": "blackbox_control",
    "kan_e3_permutation_control": "permutation_control_library",
}
_CHILD_KEYS = frozenset(
    {
        "kan_library",
        "gp_control_library",
        "permutation_control_library",
        "blackbox_control",
        "mechanism_cards",
        "blind_review_package",
    }
)
_ARM_CHILD_KEYS = {
    "kan_e3_selected": "kan_library",
    "typed_gp_sr_control": "gp_control_library",
    "matched_blackbox_control": "blackbox_control",
    "kan_e3_permutation_control": "permutation_control_library",
}
_PROVIDER_FIELDS = (
    "path",
    "tree_sha256",
    "stat_inventory_sha256",
    "file_count",
    "total_bytes",
)
_HASH_FIELDS = (
    "base_lock_sha256",
    "implementation_lock_sha256",
    "mining_manifest_sha256",
    "development_opening_sha256",
    "development_topology_sha256",
)
_DEFAULT_BASE_LOCK = BASE_LOCK


class QuantaArmAdapter(Protocol):
    """The already-pinned Quanta adapter surface needed by this runner."""

    identity: dict[str, object]

    def initialize_and_verify_provider(self) -> None: ...

    def evaluate_alpha158(self, **kwargs: object) -> dict[str, Any]: ...

    def evaluate_panel(
        self, panel: pd.DataFrame, **kwargs: object
    ) -> dict[str, Any]: ...

    def write_portfolio_diagnostics(
        self, destination: Path | str
    ) -> dict[str, str]: ...


AdapterFactory = Callable[[Path, dict[str, object]], QuantaArmAdapter]


@dataclass(frozen=True)
class EvaluationIdentityPins:
    """Opening-time identities that one arm must re-prove before Quanta access."""

    base_lock_sha256: str
    implementation_lock_sha256: str
    mining_manifest_sha256: str
    development_opening_sha256: str
    development_topology_sha256: str
    provider_identity: Mapping[str, object]
    mining_rebind_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in _HASH_FIELDS:
            _require_sha256(getattr(self, field), label=field.replace("_", " "))
        if self.mining_rebind_receipt_sha256 is not None:
            _require_sha256(
                self.mining_rebind_receipt_sha256,
                label="mining rebind receipt",
            )
        _validate_provider(self.provider_identity, label="pinned provider identity")


@dataclass(frozen=True)
class StagedArmEvaluation:
    """A verified flat staging bundle awaiting ``TopologyTransaction`` publication."""

    arm: str
    topology_key: str
    staging_path: Path
    manifest: dict[str, object]
    manifest_sha256: str


@dataclass(frozen=True)
class _ExecutionContext:
    protocol_id: str
    config: Mapping[str, object]
    base_lock: Mapping[str, object]
    base_lock_path: Path
    implementation_lock_path: Path
    mining_manifest_path: Path
    mining_manifest: Mapping[str, object]
    development_opening_path: Path
    development_opening: Mapping[str, object]
    development_dates: tuple[pd.Timestamp, ...]
    provider_identity: dict[str, object]
    mining_child_paths: Mapping[str, str]
    mining_source_protocol_id: str
    mining_rebind_receipt_sha256: str | None


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} lacks string {key}")
    return value


def _library_size_bounds(config: Mapping[str, object]) -> tuple[int, int]:
    admission = _mapping(config.get("admission"), label="frozen admission")
    minimum = admission.get("minimum_library_size")
    maximum = admission.get("library_cap")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or minimum < 1
        or maximum < minimum
    ):
        raise ValueError("frozen admission has invalid library-size bounds")
    return minimum, maximum


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _mapping(value, label=label)


def _workspace_file(root: Path, raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the workspace")
    candidate = root.joinpath(relative)
    if candidate.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"{label} must be a contained regular file")
    return resolved


def _workspace_directory(root: Path, raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the workspace")
    candidate = root.joinpath(relative)
    if candidate.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError(f"{label} must be a contained directory")
    return resolved


def _validate_provider(
    observed: Mapping[str, object], *, label: str
) -> dict[str, object]:
    provider = dict(observed)
    for field in _PROVIDER_FIELDS:
        if field not in provider:
            raise ValueError(f"{label} lacks {field}")
    if not isinstance(provider["path"], str) or not provider["path"]:
        raise ValueError(f"{label} path is invalid")
    for field in ("tree_sha256", "stat_inventory_sha256"):
        _require_sha256(provider[field], label=f"{label} {field}")
    for field in ("file_count", "total_bytes"):
        if type(provider[field]) is not int or provider[field] < 0:
            raise ValueError(f"{label} {field} is invalid")
    return {field: provider[field] for field in _PROVIDER_FIELDS}


def _calendar_sha256(dates: tuple[pd.Timestamp, ...]) -> str:
    payload = [date_value.isoformat() for date_value in dates]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_live_panel(
    root: Path,
    panel: PitPanel,
    base_lock: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.Timestamp, ...]:
    data = _mapping(base_lock.get("data"), label="base lock data")
    cache_path = Path(_string(data, "cache_path", label="base lock data")).resolve(
        strict=True
    )
    expected_cache_sha256 = _require_sha256(
        data.get("cache_sha256"), label="base lock PIT cache"
    )
    live_cache_sha256 = sha256_file(cache_path)
    if live_cache_sha256 != expected_cache_sha256:
        raise ValueError("live PIT cache hash differs from the base lock")
    data_config_path = _workspace_file(
        root, data.get("config_path"), label="frozen PIT cache config"
    )
    if sha256_file(data_config_path) != data.get("config_sha256"):
        raise ValueError("frozen PIT cache config hash mismatch")
    data_config = _read_json(data_config_path, label="frozen PIT cache config")
    if (
        data_config.get("cache_path") != str(cache_path)
        or data_config.get("cache_sha256") != live_cache_sha256
    ):
        raise ValueError("PIT cache config differs from the live locked cache")
    if (
        panel.source_path is None
        or Path(panel.source_path).resolve(strict=True) != cache_path
        or panel.source_sha256 != live_cache_sha256
    ):
        raise ValueError("PIT panel source attributes differ from the live cache")

    columns = ["datetime", "instrument", *RAW_FIELDS, "in_universe"]
    if panel.tradability is not None:
        columns.append("tradable")
    replay = PitPanel.from_frame(
        pd.read_parquet(cache_path, columns=columns),
        source_path=cache_path,
        source_sha256=live_cache_sha256,
    )
    try:
        pd.testing.assert_frame_equal(panel.raw, replay.raw, check_exact=True)
        pd.testing.assert_series_equal(
            panel.membership, replay.membership, check_exact=True
        )
        for field in RAW_FIELDS:
            pd.testing.assert_series_equal(
                panel.observed[field], replay.observed[field], check_exact=True
            )
        if panel.tradability is None:
            if replay.tradability is not None:
                raise AssertionError("unexpected replay tradability")
        elif replay.tradability is None:
            raise AssertionError("missing replay tradability")
        else:
            pd.testing.assert_series_equal(
                panel.tradability, replay.tradability, check_exact=True
            )
    except AssertionError as error:
        raise ValueError(
            "PIT panel content differs from exact live-cache replay"
        ) from error

    protocol_data = _mapping(config.get("data"), label="frozen protocol data")
    period = protocol_data.get("development_test")
    if not isinstance(period, list) or len(period) != 2:
        raise ValueError("frozen protocol lacks the development-test period")
    start, end = (pd.Timestamp(value) for value in period)
    if start > end:
        raise ValueError("frozen development-test period is reversed")
    observed_dates = pd.DatetimeIndex(
        panel.raw.index.get_level_values("datetime").unique()
    ).sort_values()
    dates = tuple(
        pd.Timestamp(value)
        for value in observed_dates[(observed_dates >= start) & (observed_dates <= end)]
    )
    if not dates:
        raise ValueError("live PIT panel has no development-test calendar")
    return dates


def _verify_development_opening(
    root: Path,
    pins: EvaluationIdentityPins,
    config: Mapping[str, object],
    protocol_id: str,
    provider_identity: Mapping[str, object],
    development_dates: tuple[pd.Timestamp, ...],
) -> tuple[Path, Mapping[str, object]]:
    artifact_paths = _mapping(
        config.get("artifact_paths"), label="frozen artifact paths"
    )
    opening_path = _workspace_file(
        root,
        artifact_paths.get("development_opening"),
        label="development opening",
    )
    if sha256_file(opening_path) != pins.development_opening_sha256:
        raise ValueError(
            "development opening identity differs from the consumed opening"
        )
    opening = _read_json(opening_path, label="development opening")
    expected_schema = (
        "mirage_s2a_development_opening_v3"
        if pins.mining_rebind_receipt_sha256 is not None
        else "mirage_s2a_development_opening_v2"
    )
    if (
        opening.get("schema_version") != expected_schema
        or opening.get("protocol_id") != protocol_id
        or opening.get("state") != "consumed_before_first_development_access"
    ):
        raise ValueError(
            "development opening has an invalid schema, protocol, or state"
        )
    topology = TopologyTransaction.from_frozen_config(root, phase="development")
    if (
        topology.topology_sha256 != pins.development_topology_sha256
        or opening.get("topology_sha256") != topology.topology_sha256
    ):
        raise ValueError("development opening has the wrong frozen topology")
    evaluations = _mapping(
        artifact_paths.get("evaluations"), label="frozen evaluation paths"
    )
    if opening.get("evaluation_paths") != dict(evaluations) or set(evaluations) != set(
        ARMS
    ):
        raise ValueError("development opening does not bind the exact five arms")
    expected_identities = {
        "base_lock_sha256": pins.base_lock_sha256,
        "implementation_lock_sha256": pins.implementation_lock_sha256,
        "mining_manifest_sha256": pins.mining_manifest_sha256,
        "provider_identity": dict(provider_identity),
    }
    if pins.mining_rebind_receipt_sha256 is not None:
        expected_identities["mining_rebind_receipt_sha256"] = (
            pins.mining_rebind_receipt_sha256
        )
        if opening.get("mining_authorization_kind") != (
            "verified_cross_protocol_rebind"
        ):
            raise ValueError("development opening lacks rebind authorization")
    if opening.get("identity_pins") != expected_identities:
        raise ValueError("development opening identities differ from arm pins")
    if (
        opening.get("development_calendar_sha256")
        != _calendar_sha256(development_dates)
        or opening.get("development_calendar_count") != len(development_dates)
        or opening.get("development_calendar_start") != development_dates[0].isoformat()
        or opening.get("development_calendar_end") != development_dates[-1].isoformat()
    ):
        raise ValueError("development opening has the wrong exact calendar identity")
    return opening_path, opening


def _verify_flat_manifest_files(
    path: Path, manifest: Mapping[str, object], *, label: str
) -> None:
    files = _mapping(manifest.get("files"), label=f"{label} files")
    entries = tuple(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} must contain only flat regular files")
    if {entry.name for entry in entries} != set(files) | {"manifest.json"}:
        raise ValueError(f"{label} file set differs from its manifest")
    for filename, expected_sha256 in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{label} manifest has an unsafe filename")
        _require_sha256(expected_sha256, label=f"{label} file {filename}")
        if sha256_file(path / filename) != expected_sha256:
            raise ValueError(f"{label} file hash mismatch: {filename}")


def _verify_mining_top_cross_links(
    root: Path,
    config: Mapping[str, object],
    base_lock: Mapping[str, object],
    pins: EvaluationIdentityPins,
    mining_path: Path,
    manifest: Mapping[str, object],
    *,
    source_artifact_paths: Mapping[str, object] | None = None,
    source_protocol_sha256: str | None = None,
    source_implementation_sha256: str | None = None,
) -> None:
    if (
        manifest.get("schema_version") != "mirage_s2a_v2_staging_bundle_v1"
        or manifest.get("role") != "mining_top_bundle"
        or manifest.get("topology_key") != "mining_run"
    ):
        raise ValueError(
            "mining top manifest has an invalid schema, role, or topology key"
        )
    topology_sha256 = _require_sha256(
        manifest.get("topology_sha256"), label="mining top topology"
    )
    if manifest.get("published_child_topology_sha256") != topology_sha256:
        raise ValueError("mining top and published children have different topology")
    _verify_flat_manifest_files(mining_path, manifest, label="mining top")
    identities = _mapping(manifest.get("identities"), label="mining top identities")
    protocol = _mapping(base_lock.get("protocol"), label="base lock protocol")
    expected_protocol_sha256 = (
        protocol.get("sha256")
        if source_protocol_sha256 is None
        else source_protocol_sha256
    )
    expected_implementation_sha256 = (
        pins.implementation_lock_sha256
        if source_implementation_sha256 is None
        else source_implementation_sha256
    )
    if (
        identities.get("protocol_sha256") != expected_protocol_sha256
        or identities.get("implementation_sha256")
        != expected_implementation_sha256
    ):
        raise ValueError("mining top protocol or implementation identity is stale")
    _require_sha256(identities.get("authority_sha256"), label="mining authority")

    child_hashes = _mapping(
        manifest.get("child_manifest_sha256"), label="mining child hashes"
    )
    child_paths = _mapping(
        manifest.get("published_child_paths"), label="mining child paths"
    )
    if set(child_hashes) != _CHILD_KEYS or set(child_paths) != _CHILD_KEYS:
        raise ValueError("mining top does not register every frozen child")
    artifact_paths = (
        _mapping(config.get("artifact_paths"), label="frozen artifact paths")
        if source_artifact_paths is None
        else source_artifact_paths
    )
    for child_key in _CHILD_KEYS:
        configured = artifact_paths.get(child_key)
        if not isinstance(configured, str):
            raise ValueError(f"frozen artifact paths lack {child_key}")
        expected_path = root.joinpath(configured).resolve(strict=True)
        recorded_path = Path(str(child_paths[child_key])).resolve(strict=True)
        if recorded_path != expected_path:
            raise ValueError(f"mining top has the wrong published path for {child_key}")
        child_manifest_path = expected_path / "manifest.json"
        expected_hash = _require_sha256(
            child_hashes[child_key], label=f"mining child {child_key}"
        )
        if sha256_file(child_manifest_path) != expected_hash:
            raise ValueError(f"mining child manifest identity mismatch: {child_key}")
        child_manifest = _read_json(
            child_manifest_path, label=f"published child {child_key}"
        )
        if (
            child_manifest.get("topology_key") != child_key
            or child_manifest.get("topology_sha256") != topology_sha256
        ):
            raise ValueError(
                f"published child topology binding is invalid: {child_key}"
            )

    lineage = _mapping(
        manifest.get("kan_selected_lineage"), label="selected KAN lineage"
    )
    minimum, maximum = _library_size_bounds(config)
    if not minimum <= len(lineage) <= maximum:
        raise ValueError("selected KAN lineage count violates frozen admission bounds")
    global_indices: list[int] = []
    for factor_id, raw_record in lineage.items():
        if not isinstance(factor_id, str) or not factor_id:
            raise ValueError("selected KAN lineage has an invalid factor ID")
        record = _mapping(raw_record, label=f"selected KAN lineage {factor_id}")
        _require_sha256(record.get("canonical_hash"), label="selected KAN AST")
        global_index = record.get("global_attempt_index")
        if type(global_index) is not int or not 0 <= global_index < 256:
            raise ValueError("selected KAN lineage has an invalid global attempt index")
        global_indices.append(global_index)
    if len(set(global_indices)) != len(global_indices):
        raise ValueError("selected KAN lineage reuses a global attempt index")

    cards_path = root.joinpath(str(artifact_paths["mechanism_cards"])).resolve(
        strict=True
    )
    cards_manifest_path = cards_path / "manifest.json"
    cards_sha256 = sha256_file(cards_manifest_path)
    if (
        manifest.get("mechanism_cards_manifest_sha256") != cards_sha256
        or child_hashes.get("mechanism_cards") != cards_sha256
    ):
        raise ValueError("mechanism-card identity differs from the mining top")
    cards_manifest = _read_json(cards_manifest_path, label="mechanism-card manifest")
    if (
        cards_manifest.get("schema_version") != "mirage_s2a_v2_staging_bundle_v1"
        or cards_manifest.get("role") != "kan_mechanism_evidence_pending_human_review"
        or cards_manifest.get("output_kind") != "mechanism_cards"
        or cards_manifest.get("selected_factor_ids") != list(lineage)
        or cards_manifest.get("card_count") != len(lineage)
    ):
        raise ValueError("mechanism cards do not match selected KAN factors")
    _verify_flat_manifest_files(cards_path, cards_manifest, label="mechanism cards")
    card_rows = [
        json.loads(line)
        for line in (cards_path / "mechanism_cards.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if [row.get("factor_id") for row in card_rows] != list(lineage):
        raise ValueError("mechanism-card rows differ from selected KAN factor order")
    for row in card_rows:
        factor_id = row["factor_id"]
        card = _mapping(row.get("card"), label=f"mechanism card {factor_id}")
        identity = _mapping(
            card.get("identity_and_canonical_ast"),
            label=f"mechanism card identity {factor_id}",
        )
        if identity.get("factor_id") != factor_id or identity.get(
            "canonical_hash"
        ) != _mapping(lineage[factor_id], label="KAN lineage").get("canonical_hash"):
            raise ValueError("mechanism-card AST identity differs from KAN lineage")


def _verify_execution_context(
    root: Path, panel: PitPanel, pins: EvaluationIdentityPins
) -> _ExecutionContext:
    base_lock_path = root / _DEFAULT_BASE_LOCK
    if not base_lock_path.is_file() or base_lock_path.is_symlink():
        raise ValueError("base lock is not a regular file")
    if sha256_file(base_lock_path) != pins.base_lock_sha256:
        raise ValueError("base lock identity differs from the development opening")
    base_lock = _read_json(base_lock_path, label="base lock")
    protocol = _mapping(base_lock.get("protocol"), label="base lock protocol")
    protocol_id = _string(protocol, "protocol_id", label="base lock protocol")
    if protocol_id != PROTOCOL_ID:
        raise ValueError("base lock has the wrong S2a v2 protocol")
    config_path = _workspace_file(
        root, protocol.get("path"), label="frozen protocol config"
    )
    if sha256_file(config_path) != protocol.get("sha256"):
        raise ValueError("frozen protocol config hash mismatch")
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(config_value, label="frozen protocol config")
    if config.get("protocol_id") != protocol_id:
        raise ValueError("frozen protocol config has the wrong protocol")
    controls = _mapping(config.get("controls"), label="frozen controls")
    if tuple(controls.get("arms", ())) != ARMS:
        raise ValueError("frozen config does not contain the exact five S2a v2 arms")
    artifact_paths = _mapping(
        config.get("artifact_paths"), label="frozen artifact paths"
    )

    implementation_lock_path = _workspace_file(
        root,
        artifact_paths.get("implementation_lock"),
        label="implementation lock",
    )
    if sha256_file(implementation_lock_path) != pins.implementation_lock_sha256:
        raise ValueError(
            "implementation lock identity differs from the development opening"
        )
    implementation = _read_json(implementation_lock_path, label="implementation lock")
    if implementation.get("schema_version") != "mirage_s2_implementation_lock_v2":
        raise ValueError("unsupported S2a v2 implementation lock")
    if implementation.get("protocol_id") != protocol_id:
        raise ValueError("implementation lock has the wrong protocol")
    locked_provider = _validate_provider(
        _mapping(
            implementation.get("qlib_provider"),
            label="implementation lock provider",
        ),
        label="implementation lock provider identity",
    )
    pinned_provider = _validate_provider(
        pins.provider_identity, label="pinned provider identity"
    )
    if locked_provider != pinned_provider:
        raise ValueError("implementation-lock provider identity differs from opening")

    mining_source = config.get("mining_source")
    rebind_sha256: str | None = None
    if isinstance(mining_source, Mapping) and mining_source.get("mode") == (
        "verified_cross_protocol_rebind"
    ):
        binding = verify_mining_rebind_receipt(
            root,
            target_base_lock_path=base_lock_path,
            target_implementation_lock_path=implementation_lock_path,
        )
        source = _mapping(binding.get("source"), label="rebound mining source")
        source_artifacts = _mapping(
            source.get("artifacts"), label="rebound mining artifacts"
        )
        mining_child_paths = {
            key: _string(
                _mapping(value, label=f"rebound artifact {key}"),
                "path",
                label=f"rebound artifact {key}",
            )
            for key, value in source_artifacts.items()
        }
        receipt_relative = _string(
            mining_source,
            "rebind_receipt",
            label="rebound mining source",
        )
        receipt_path = _workspace_file(
            root, receipt_relative, label="mining rebind receipt"
        )
        rebind_sha256 = sha256_file(receipt_path)
        if pins.mining_rebind_receipt_sha256 != rebind_sha256:
            raise ValueError(
                "mining rebind receipt identity differs from the development opening"
            )
        source_config = _mapping(
            source.get("config"), label="rebound source config"
        )
        source_implementation = _mapping(
            source.get("implementation_lock"),
            label="rebound source implementation",
        )
        source_protocol_sha256 = _require_sha256(
            source_config.get("sha256"), label="rebound source protocol"
        )
        source_implementation_sha256 = _require_sha256(
            source_implementation.get("sha256"),
            label="rebound source implementation",
        )
        source_protocol_id = _string(
            source, "protocol_id", label="rebound source"
        )
    else:
        mining_child_paths = {
            key: _string(
                artifact_paths, key, label="frozen artifact paths"
            )
            for key in _CHILD_KEYS | {"mining_run"}
        }
        source_artifacts = None
        source_protocol_sha256 = None
        source_implementation_sha256 = None
        source_protocol_id = protocol_id
        if pins.mining_rebind_receipt_sha256 is not None:
            raise ValueError("direct mining cannot carry a rebind receipt identity")
    mining_directory = _workspace_directory(
        root, mining_child_paths["mining_run"], label="mining run"
    )
    mining_manifest_path = mining_directory / "manifest.json"
    if not mining_manifest_path.is_file() or mining_manifest_path.is_symlink():
        raise ValueError("mining manifest is not a regular file")
    if sha256_file(mining_manifest_path) != pins.mining_manifest_sha256:
        raise ValueError(
            "mining manifest identity differs from the development opening"
        )
    mining_manifest = _read_json(mining_manifest_path, label="mining manifest")
    if mining_manifest.get("protocol_id") not in (None, protocol_id):
        raise ValueError("mining manifest has the wrong protocol")
    _verify_mining_top_cross_links(
        root,
        config,
        base_lock,
        pins,
        mining_directory,
        mining_manifest,
        source_artifact_paths=mining_child_paths,
        source_protocol_sha256=source_protocol_sha256,
        source_implementation_sha256=source_implementation_sha256,
    )

    development_dates = _verify_live_panel(root, panel, base_lock, config)
    development_opening_path, development_opening = _verify_development_opening(
        root,
        pins,
        config,
        protocol_id,
        pinned_provider,
        development_dates,
    )

    return _ExecutionContext(
        protocol_id=protocol_id,
        config=config,
        base_lock=base_lock,
        base_lock_path=base_lock_path,
        implementation_lock_path=implementation_lock_path,
        mining_manifest_path=mining_manifest_path,
        mining_manifest=mining_manifest,
        development_opening_path=development_opening_path,
        development_opening=development_opening,
        development_dates=development_dates,
        provider_identity=pinned_provider,
        mining_child_paths=mining_child_paths,
        mining_source_protocol_id=source_protocol_id,
        mining_rebind_receipt_sha256=rebind_sha256,
    )


def _registered_input(
    root: Path, arm: str, context: _ExecutionContext
) -> tuple[dict[str, object], Path | None]:
    if arm == "alpha158_replay":
        binding: dict[str, object] = {
            "kind": "official_alpha158",
            "path": None,
            "manifest_sha256": None,
        }
        return binding, None

    expected_relative = context.mining_child_paths.get(_ARTIFACT_KEYS[arm])
    input_path = _workspace_directory(root, expected_relative, label=f"{arm} input")
    manifest_path = input_path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"{arm} input manifest is not a regular file")
    child_hashes = _mapping(
        context.mining_manifest.get("child_manifest_sha256"),
        label="mining child hashes",
    )
    expected_manifest_sha256 = _require_sha256(
        child_hashes.get(_ARM_CHILD_KEYS[arm]),
        label=f"registered {arm} input manifest",
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"{arm} registered input manifest identity mismatch")
    binding = {
        "kind": _INPUT_KINDS[arm],
        "path": expected_relative,
        "manifest_sha256": expected_manifest_sha256,
        "source_protocol_id": context.mining_source_protocol_id,
        "mining_rebind_receipt_sha256": context.mining_rebind_receipt_sha256,
    }
    return binding, input_path


def _mining_panel(
    panel: PitPanel, context: _ExecutionContext
) -> tuple[PitPanel, pd.Timestamp]:
    protocol_data = _mapping(context.config.get("data"), label="frozen protocol data")
    train = protocol_data.get("train")
    validation = protocol_data.get("validation")
    if (
        not isinstance(train, list)
        or len(train) != 2
        or not isinstance(validation, list)
        or len(validation) != 2
    ):
        raise ValueError("frozen protocol lacks train or validation periods")
    train_start, train_end = (pd.Timestamp(value) for value in train)
    validation_start, validation_end = (pd.Timestamp(value) for value in validation)
    if (
        train_start > train_end
        or train_end >= validation_start
        or validation_start > validation_end
    ):
        raise ValueError("frozen train and validation periods are invalid")
    warmup = _mapping(
        protocol_data.get("feature_warmup"), label="frozen feature warm-up"
    )
    if (
        warmup.get("trading_dates") != 60
        or warmup.get("raw_only") is not True
        or warmup.get("labels_outside_objective_split_are_null") is not True
    ):
        raise ValueError(
            "frozen feature warm-up contract is not exact last-60 raw-only"
        )
    dates = pd.DatetimeIndex(panel.raw.index.get_level_values("datetime"))
    unique_dates = dates.unique().sort_values()
    warmup_dates = unique_dates[unique_dates < train_start][-60:]
    if len(warmup_dates) != 60:
        raise ValueError("full PIT panel lacks the exact 60-date feature warm-up")
    objective_dates = unique_dates[
        (unique_dates >= train_start) & (unique_dates <= validation_end)
    ]
    selected_dates = warmup_dates.append(objective_dates)
    index = panel.raw.index[dates.isin(selected_dates)]
    if index.empty or objective_dates.empty or not bool((dates > validation_end).any()):
        raise ValueError(
            "full PIT panel must contain both mining and post-opening development rows"
        )
    return (
        PitPanel(
            raw=panel.raw.loc[index],
            membership=panel.membership.loc[index],
            observed={
                name: values.loc[index] for name, values in panel.observed.items()
            },
            tradability=(
                None if panel.tradability is None else panel.tradability.loc[index]
            ),
            source_path=panel.source_path,
            source_sha256=panel.source_sha256,
        ),
        validation_end,
    )


def _verify_mining_artifact_index(
    values: pd.DataFrame,
    mining_panel: PitPanel,
    validation_end: pd.Timestamp,
    *,
    label: str,
) -> None:
    if (
        not isinstance(values.index, pd.MultiIndex)
        or list(values.index.names) != ["datetime", "instrument"]
        or values.empty
    ):
        raise ValueError(f"{label} has an invalid panel index")
    dates = pd.DatetimeIndex(pd.to_datetime(values.index.get_level_values("datetime")))
    if bool((dates > validation_end).any()):
        raise ValueError(f"{label} contains predictions after validation end")
    if not values.index.equals(mining_panel.raw.index):
        raise ValueError(f"{label} does not exactly cover the mining panel")


def _load_factor_library(
    arm: str, path: Path, panel: PitPanel, context: _ExecutionContext
) -> tuple[pd.DataFrame, dict[str, object]]:
    mining_panel, validation_end = _mining_panel(panel, context)
    stored_values = pd.read_parquet(path / "factor_panel.parquet")
    _verify_mining_artifact_index(
        stored_values,
        mining_panel,
        validation_end,
        label=f"{arm} factor artifact",
    )
    verification = verify_library(path, mining_panel)
    manifest = _read_json(path / "manifest.json", label=f"{arm} library manifest")
    if manifest.get("schema_version") != "mirage_factor_library_v1":
        raise ValueError(f"{arm} has an unsupported factor-library schema")
    if manifest.get("library_role") != arm:
        raise ValueError(f"{arm} factor library has the wrong exact arm role")
    expected_kan_mined = arm == "kan_e3_selected"
    if manifest.get("kan_mined") is not expected_kan_mined:
        raise ValueError(f"{arm} factor library has the wrong KAN-mined status")
    factors = _mapping(manifest.get("factors"), label=f"{arm} factor records")
    factor_ids = tuple(stored_values.columns)
    minimum, maximum = _library_size_bounds(context.config)
    if (
        manifest.get("factor_count") != len(factor_ids)
        or not minimum <= len(factor_ids) <= maximum
        or len(set(factor_ids)) != len(factor_ids)
        or set(factors) != set(factor_ids)
        or verification.get("verified") is not True
        or verification.get("factor_count") != len(factor_ids)
    ):
        raise ValueError(f"{arm} factor count or selected IDs are invalid")
    recorded_selected = manifest.get("selected_candidate_ids")
    if recorded_selected is not None and recorded_selected != list(factor_ids):
        raise ValueError(f"{arm} recorded selected IDs differ from its panel")
    for factor_id in factor_ids:
        record = _mapping(factors[factor_id], label=f"{arm} factor {factor_id}")
        canonical_hash = _require_sha256(
            record.get("canonical_hash"), label=f"{arm} factor AST lineage"
        )
        program = AstNode.from_dict(
            _mapping(record.get("ast"), label=f"{arm} factor AST")
        )
        if program.identity != canonical_hash:
            raise ValueError(f"{arm} factor AST differs from its canonical hash")
    if arm == "kan_e3_selected":
        lineage = _mapping(
            context.mining_manifest.get("kan_selected_lineage"),
            label="selected KAN lineage",
        )
        if list(factor_ids) != list(lineage):
            raise ValueError("KAN library selected IDs differ from mining lineage")
        for factor_id in factor_ids:
            factor_record = _mapping(factors[factor_id], label="KAN factor record")
            lineage_record = _mapping(lineage[factor_id], label="KAN lineage")
            if factor_record.get("canonical_hash") != lineage_record.get(
                "canonical_hash"
            ) or factor_record.get("global_attempt_index") != lineage_record.get(
                "global_attempt_index"
            ):
                raise ValueError("KAN library AST identity differs from mining lineage")
    replayed: dict[str, pd.Series] = {}
    for factor_id in factor_ids:
        record = _mapping(factors[factor_id], label=f"{arm} factor {factor_id}")
        program = AstNode.from_dict(
            _mapping(record.get("ast"), label=f"{arm} factor AST")
        )
        result = evaluate(program, panel)
        replayed[factor_id] = result.values.where(result.support & panel.membership)
    values = pd.DataFrame(replayed, index=panel.raw.index)
    return values, {
        "kind": "verified_factor_library",
        "path": str(path),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "library_role": arm,
        "kan_mined": expected_kan_mined,
        "factor_library": True,
        "factor_count": len(factor_ids),
        "selected_factor_ids": list(factor_ids),
        "verification": dict(verification),
        "artifact_panel_end": validation_end.isoformat(),
        "artifact_panel_rows": len(stored_values),
        "development_values_source": "independent_ast_replay_from_raw_pit",
    }


def _blackbox_pairing_matches_lineage(
    selected_factor_ids: list[object],
    paired_indices: list[object],
    lineage: Mapping[str, object],
) -> bool:
    observed = dict(zip(selected_factor_ids, paired_indices, strict=True))
    expected = {
        factor_id: _mapping(record, label="KAN lineage").get("global_attempt_index")
        for factor_id, record in lineage.items()
    }
    return observed == expected


def _load_blackbox_control(
    path: Path,
    context: _ExecutionContext,
    panel: PitPanel,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path, label="matched blackbox control manifest")
    if manifest.get("schema_version") != "mirage_matched_blackbox_control_v2":
        raise ValueError("unsupported matched blackbox control schema")
    expected_fields = {
        "arm": "matched_blackbox_control",
        "role": "falsification_control_never_production",
        "output_kind": "control_panel_not_factor_library",
        "promotion_eligible": False,
        "factor_library_publication_allowed": False,
        "kan_mined": False,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            if field == "factor_library_publication_allowed":
                raise ValueError("matched blackbox control cannot be a factor library")
            raise ValueError(f"matched blackbox control has invalid {field}")
    forbidden = {"library_role", "factors"}.intersection(manifest)
    if forbidden:
        raise ValueError(
            "matched blackbox control cannot masquerade as a factor library"
        )
    files = _mapping(manifest.get("files"), label="blackbox control files")
    expected_files = set(files) | {"manifest.json"}
    entries = tuple(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("matched blackbox control must contain flat regular files")
    if {entry.name for entry in entries} != expected_files:
        raise ValueError("matched blackbox control file set differs from its manifest")
    for filename, expected_sha256 in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("blackbox control manifest has an unsafe filename")
        _require_sha256(expected_sha256, label=f"blackbox file {filename}")
        if sha256_file(path / filename) != expected_sha256:
            raise ValueError(f"blackbox control file hash mismatch: {filename}")
    if "prediction_panel.parquet" not in files:
        raise ValueError("matched blackbox control lacks prediction_panel.parquet")
    if "control_receipts.jsonl" not in files:
        raise ValueError("matched blackbox control lacks control_receipts.jsonl")
    for required in ("control_trajectories.jsonl", "tensor_evidence.zlib"):
        if required not in files:
            raise ValueError(f"matched blackbox control lacks {required}")
    mining_panel, validation_end = _mining_panel(panel, context)
    prediction = pd.read_parquet(path / "prediction_panel.parquet")
    _verify_mining_artifact_index(
        prediction,
        mining_panel,
        validation_end,
        label="matched blackbox artifact",
    )
    minimum, maximum = _library_size_bounds(context.config)
    if not minimum <= prediction.shape[1] <= maximum:
        raise ValueError("matched blackbox prediction panel has invalid shape or index")
    if manifest.get("control_count") != prediction.shape[1]:
        raise ValueError("matched blackbox control count differs from predictions")
    selected_factor_ids = manifest.get("selected_kan_factor_ids")
    paired_indices = manifest.get("paired_kan_global_attempt_indices")
    if (
        not isinstance(selected_factor_ids, list)
        or not isinstance(paired_indices, list)
        or len(selected_factor_ids) != prediction.shape[1]
        or len(paired_indices) != prediction.shape[1]
        or len(set(selected_factor_ids)) != len(selected_factor_ids)
        or len(set(paired_indices)) != len(paired_indices)
    ):
        raise ValueError("matched blackbox KAN pairing identities are invalid")
    lineage = _mapping(
        context.mining_manifest.get("kan_selected_lineage"),
        label="selected KAN lineage",
    )
    if not _blackbox_pairing_matches_lineage(
        selected_factor_ids,
        paired_indices,
        lineage,
    ):
        raise ValueError("matched blackbox pairing differs from selected KAN lineage")
    receipt_rows = [
        json.loads(line)
        for line in (path / "control_receipts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(receipt_rows) != prediction.shape[1]:
        raise ValueError("matched blackbox receipt count differs from predictions")
    trajectory_lines = (
        (path / "control_trajectories.jsonl").read_bytes().splitlines(keepends=True)
    )
    if len(trajectory_lines) != prediction.shape[1]:
        raise ValueError("matched blackbox trajectory receipt count is invalid")
    try:
        tensor_archive = zlib.decompress((path / "tensor_evidence.zlib").read_bytes())
    except zlib.error as error:
        raise ValueError("matched blackbox tensor evidence is invalid") from error

    def tensor_from_reference(value: object, label: str) -> np.ndarray:
        reference = _mapping(value, label=label)
        _require_sha256(reference.get("sha256"), label=label)
        offset = reference.get("offset")
        nbytes = reference.get("nbytes")
        shape = reference.get("shape")
        dtype = reference.get("dtype")
        if (
            type(offset) is not int
            or type(nbytes) is not int
            or offset < 0
            or nbytes < 0
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or dtype not in {"|u1", "|i1", "<i2", "<i4", "<i8", "<f4", "<f8"}
        ):
            raise ValueError(f"{label} has invalid tensor bounds, shape, or dtype")
        payload = tensor_archive[offset : offset + nbytes]
        if len(payload) != nbytes or hashlib.sha256(
            payload
        ).hexdigest() != reference.get("sha256"):
            raise ValueError(f"{label} tensor payload hash mismatch")
        array = np.frombuffer(payload, dtype=np.dtype(dtype))
        if array.size != math.prod(shape):
            raise ValueError(f"{label} tensor byte count differs from shape")
        return np.array(array.reshape(shape), copy=True)

    control_ids: list[str] = []
    mining_atom_panels: dict[str, AtomPanel] = {}
    full_atom_panels: dict[str, AtomPanel] = {}
    full_predictions: dict[str, pd.Series] = {}
    for row_index, (row, factor_id, global_index) in enumerate(
        zip(receipt_rows, selected_factor_ids, paired_indices, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError("matched blackbox receipt row is not a record")
        control_id = row.get("control_id")
        if not isinstance(control_id, str) or not control_id:
            raise ValueError("matched blackbox receipt lacks a control ID")
        control_ids.append(control_id)
        if (
            row.get("kan_factor_id") != factor_id
            or row.get("kan_global_attempt_index") != global_index
            or type(global_index) is not int
            or not 0 <= global_index < len(PROFILE_SPECS) * 64
            or row.get("seed") != MLP_SEED_BASE + global_index
        ):
            raise ValueError(
                "matched blackbox receipt has the wrong KAN pairing or seed"
            )
        profile = tuple(PROFILE_SPECS)[global_index // 64]
        atom_count = len(build_profile_atom_bank(profile))
        if (
            row.get("profile") != profile
            or row.get("optimizer") != "Adam"
            or row.get("learning_rate") != LEARNING_RATE
            or row.get("scheduled_updates") != TRAINING_STEPS
            or row.get("completed_updates") != TRAINING_STEPS
            or row.get("input_atom_count") != atom_count
            or row.get("kan_parameter_count") != 2 * atom_count
            or row.get("mlp_parameter_count") != 2 * atom_count + 5
            or row.get("atom_manifest_sha256")
            != atom_manifest_sha256(build_profile_atom_bank(profile))
            or row.get("kan_mined") is not False
            or row.get("promotion_eligible") is not False
            or row.get("factor_library_publication_allowed") is not False
        ):
            raise ValueError(
                "matched blackbox receipt violates its frozen control budget"
            )
        _require_sha256(
            row.get("valid_support_sha256"),
            label="matched blackbox valid-support identity",
        )
        expected_gap = 5 / float(2 * atom_count)
        if not math.isclose(
            float(row.get("parameter_relative_gap", math.inf)),
            expected_gap,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("matched blackbox receipt has the wrong parameter gap")
        bootstrap = _mapping(row.get("bootstrap"), label="blackbox bootstrap receipt")
        date_count = bootstrap.get("date_count")
        if type(date_count) is not int:
            raise ValueError("matched blackbox bootstrap lacks a date count")
        bootstrap_receipt = draw_training_bootstrap(
            date_count, BOOTSTRAP_SEED_BASE + global_index
        )
        if bootstrap != _json_compatible(asdict(bootstrap_receipt)):
            raise ValueError("matched blackbox bootstrap differs from paired KAN")
        initial_parameters = tensor_from_reference(
            row.get("initial_parameters"), "blackbox initial-parameter reference"
        )
        final_parameters = tensor_from_reference(
            row.get("final_parameters"), "blackbox final-parameter reference"
        )
        first_gradient = tensor_from_reference(
            row.get("first_step_data_gradient"), "blackbox gradient reference"
        )
        if any(
            value.shape != (2 * atom_count + 5,)
            for value in (initial_parameters, final_parameters, first_gradient)
        ):
            raise ValueError("matched blackbox parameter evidence has the wrong shape")
        trajectory_reference = _mapping(
            row.get("trajectory"), label="blackbox trajectory reference"
        )
        if (
            trajectory_reference.get("file") != "control_trajectories.jsonl"
            or trajectory_reference.get("line") != row_index
            or trajectory_reference.get("sha256")
            != hashlib.sha256(trajectory_lines[row_index]).hexdigest()
        ):
            raise ValueError("matched blackbox trajectory reference is invalid")
        trajectory_record = json.loads(trajectory_lines[row_index])
        trajectory = trajectory_record.get("trajectory")
        if (
            trajectory_record.get("receipt_index") != row_index
            or trajectory_record.get("control_id") != control_id
            or not isinstance(trajectory, list)
            or len(trajectory) != TRAINING_STEPS
            or [step.get("update_index") for step in trajectory]
            != list(range(TRAINING_STEPS))
        ):
            raise ValueError("matched blackbox trajectory is incomplete")
        last_trajectory_parameters: np.ndarray | None = None
        replay_trajectory: list[MLPTrainingStepReceipt] = []
        for step in trajectory:
            parameters = tensor_from_reference(
                step.get("parameters"), "blackbox trajectory parameter reference"
            )
            total_loss = step.get("total_loss")
            mean_daily_ic = step.get("mean_daily_ic")
            if parameters.shape != final_parameters.shape:
                raise ValueError(
                    "matched blackbox trajectory parameter shape is invalid"
                )
            if (
                not isinstance(total_loss, (int, float))
                or isinstance(total_loss, bool)
                or not math.isfinite(float(total_loss))
                or not isinstance(mean_daily_ic, (int, float))
                or isinstance(mean_daily_ic, bool)
                or not math.isfinite(float(mean_daily_ic))
            ):
                raise ValueError("matched blackbox trajectory metrics are invalid")
            replay_trajectory.append(
                MLPTrainingStepReceipt(
                    update_index=int(step["update_index"]),
                    total_loss=float(total_loss),
                    mean_daily_ic=float(mean_daily_ic),
                    parameters=torch.from_numpy(parameters),
                )
            )
            last_trajectory_parameters = parameters
        if last_trajectory_parameters is None or not np.array_equal(
            last_trajectory_parameters, final_parameters
        ):
            raise ValueError(
                "matched blackbox last checkpoint differs from final parameters"
            )
        tensor_from_reference(
            row.get("training_prediction"), "blackbox training-prediction reference"
        )
        tensor_from_reference(
            row.get("training_prediction_mask"),
            "blackbox training-prediction-mask reference",
        )
        published_values = prediction.iloc[:, row_index].to_numpy(dtype=float)
        published_mask = np.isfinite(published_values)
        recorded_prediction = tensor_from_reference(
            row.get("prediction"), "blackbox prediction reference"
        )
        recorded_mask = tensor_from_reference(
            row.get("prediction_mask"), "blackbox prediction-mask reference"
        ).astype(bool)
        if not np.array_equal(recorded_mask, published_mask) or not np.array_equal(
            recorded_prediction,
            np.nan_to_num(published_values, nan=0.0),
        ):
            raise ValueError("matched blackbox prediction tensor differs from parquet")

        control_receipt = MLPControlReceipt(
            profile=profile,
            kan_global_attempt_index=global_index,
            seed=int(row["seed"]),
            bootstrap=bootstrap_receipt,
            optimizer=str(row["optimizer"]),
            learning_rate=float(row["learning_rate"]),
            scheduled_updates=int(row["scheduled_updates"]),
            completed_updates=int(row["completed_updates"]),
            input_atom_count=int(row["input_atom_count"]),
            kan_parameter_count=int(row["kan_parameter_count"]),
            mlp_parameter_count=int(row["mlp_parameter_count"]),
            parameter_relative_gap=float(row["parameter_relative_gap"]),
            atom_manifest_sha256=str(row["atom_manifest_sha256"]),
            valid_support_sha256=str(row["valid_support_sha256"]),
            initial_parameters=torch.from_numpy(initial_parameters),
            final_parameters=torch.from_numpy(final_parameters),
            first_step_data_gradient=torch.from_numpy(first_gradient),
            trajectory=tuple(replay_trajectory),
            prediction=torch.from_numpy(recorded_prediction),
            prediction_mask=torch.from_numpy(recorded_mask),
        )
        if profile not in mining_atom_panels:
            mining_atom_panels[profile] = materialize_atom_panel(mining_panel, profile)
            full_atom_panels[profile] = materialize_atom_panel(panel, profile)
        mining_replay = replay_control_on_atom_panel(
            control_receipt, mining_atom_panels[profile]
        ).reindex(prediction.index)
        try:
            np.testing.assert_allclose(
                prediction.iloc[:, row_index].to_numpy(dtype=float),
                mining_replay.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        except AssertionError as error:
            raise ValueError(
                "matched blackbox prediction differs from exact checkpoint replay"
            ) from error
        full_predictions[control_id] = replay_control_on_atom_panel(
            control_receipt, full_atom_panels[profile]
        ).rename(control_id)
        if row.get("receipt_index") not in (None, row_index):
            raise ValueError("matched blackbox receipt order is inconsistent")
    if (
        len(set(control_ids)) != len(control_ids)
        or list(prediction.columns) != control_ids
    ):
        raise ValueError("matched blackbox prediction columns differ from receipts")
    replayed_prediction = pd.DataFrame(full_predictions, index=panel.raw.index)
    return replayed_prediction, {
        "kind": "computed_factor_control",
        "path": str(path),
        "manifest_sha256": sha256_file(manifest_path),
        "role": expected_fields["role"],
        "output_kind": expected_fields["output_kind"],
        "control_count": prediction.shape[1],
        "selected_kan_factor_ids": selected_factor_ids,
        "paired_kan_global_attempt_indices": paired_indices,
        "factor_library": False,
        "promotion_eligible": False,
        "artifact_panel_end": validation_end.isoformat(),
        "artifact_panel_rows": len(prediction),
        "development_values_source": "independent_mlp_final_parameter_replay",
    }


def _verify_adapter_identity(
    adapter: QuantaArmAdapter, context: _ExecutionContext
) -> dict[str, object]:
    identity = dict(adapter.identity)
    if identity.get("verified") is not True:
        raise ValueError("Quanta adapter identity is not verified")
    quanta = _mapping(context.base_lock.get("quanta"), label="base lock Quanta")
    for key, label in (
        ("commit", "commit"),
        ("config_sha256", "config"),
        ("runner_sha256", "runner"),
    ):
        if identity.get(key) != quanta.get(key):
            raise ValueError(f"Quanta {label} identity differs from the base lock")
    if identity.get("effective_qlib_provider") != context.provider_identity["path"]:
        raise ValueError("effective Qlib provider identity differs from opening")
    if (
        identity.get("qlib_provider_tree_sha256")
        != context.provider_identity["tree_sha256"]
    ):
        raise ValueError("effective Qlib provider tree identity differs from opening")
    return identity


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation manifest cannot contain non-finite JSON metrics")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"evaluation manifest contains unsupported value: {type(value)}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(_json_compatible(value), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: object) -> None:
    data = _canonical_json_bytes(payload)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def _flat_file_index(staging: Path) -> dict[str, dict[str, object]]:
    entries = tuple(sorted(staging.iterdir(), key=lambda path: path.name))
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("evaluation staging must contain only flat regular files")
    return {
        entry.name: {"sha256": sha256_file(entry), "bytes": entry.stat().st_size}
        for entry in entries
    }


def _consume_arm_authority(root: Path, receipt: object) -> Path:
    protocol_id = getattr(receipt, "protocol_id")
    arm = getattr(receipt, "arm")
    if not isinstance(protocol_id, str) or arm not in ARMS:
        raise ValueError("arm authority receipt has an invalid protocol or arm")
    authority_root = root / "governance" / "authority" / protocol_id
    authority_root = authority_root.resolve(strict=True)
    if not authority_root.is_dir() or not authority_root.is_relative_to(root):
        raise ValueError("authority consumption root is invalid")
    consumption_directory = authority_root / "arm_consumptions"
    try:
        os.mkdir(consumption_directory, 0o700)
    except FileExistsError:
        if consumption_directory.is_symlink() or not consumption_directory.is_dir():
            raise ValueError("arm authority consumption path is invalid") from None
    record = {
        "schema_version": "mirage_s2a_arm_authority_consumption_v2",
        "protocol_id": protocol_id,
        "arm": arm,
        "receipt_sha256": getattr(receipt, "receipt_sha256"),
        "receipt_sequence": getattr(receipt, "sequence"),
        "boundary": getattr(receipt, "boundary"),
        "base_lock_sha256": getattr(receipt, "base_lock_sha256"),
        "state": "consumed_before_arm_identity_or_adapter_access",
    }
    destination = consumption_directory / f"{arm}.json"
    data = _canonical_json_bytes(record)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except FileExistsError as error:
        raise PermissionError(
            f"S2a v2 arm authority was already consumed: {arm}"
        ) from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("arm authority consumption write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        consumption_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination


def _verify_diagnostic_calendar(
    staging: Path, expected_dates: tuple[pd.Timestamp, ...]
) -> None:
    expected = pd.DatetimeIndex(expected_dates)
    schemas = {
        "qlib_report.parquet": ("return", "bench", "cost", "turnover"),
        "portfolio_daily.parquet": (
            "daily_excess_return",
            "turnover",
            "realized_cost",
        ),
        "prediction_coverage.parquet": (
            "total_predictions",
            "finite_predictions",
            "prediction_coverage",
        ),
    }
    for filename, required_columns in schemas.items():
        path = staging / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"exact Quanta diagnostic is absent: {filename}")
        frame = pd.read_parquet(path)
        try:
            observed = pd.DatetimeIndex(pd.to_datetime(frame.index))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{filename} has an invalid development calendar"
            ) from error
        if observed.has_duplicates or not observed.equals(expected):
            raise ValueError(
                f"{filename} calendar differs from the exact development calendar"
            )
        missing = set(required_columns).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{filename} lacks exact diagnostic columns: {sorted(missing)}"
            )
        try:
            values = frame.loc[:, required_columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{filename} diagnostics are not numeric") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{filename} contains non-finite diagnostics")
        if filename == "prediction_coverage.parquet":
            total = frame["total_predictions"].to_numpy()
            finite = frame["finite_predictions"].to_numpy()
            coverage = frame["prediction_coverage"].to_numpy(dtype=float)
            if (
                not np.equal(total, np.floor(total)).all()
                or not np.equal(finite, np.floor(finite)).all()
                or (total <= 0).any()
                or (finite < 0).any()
                or (finite > total).any()
                or not np.array_equal(coverage, finite / total)
            ):
                raise ValueError("prediction coverage diagnostics are inconsistent")


def _production_adapter(staging: Path, context: _ExecutionContext) -> QuantaAdapter:
    quanta_lock = _mapping(context.base_lock.get("quanta"), label="base lock Quanta")
    config_path = _workspace_file(
        context.base_lock_path.parent.parent,
        quanta_lock.get("pinned_config_path"),
        label="pinned Quanta config",
    )
    if sha256_file(config_path) != quanta_lock.get("pinned_config_sha256"):
        raise ValueError("pinned Quanta config identity mismatch")
    config = _read_json(config_path, label="pinned Quanta config")
    for key in ("commit", "config_sha256", "runner_sha256"):
        if config.get(key) != quanta_lock.get(key):
            raise ValueError(f"pinned Quanta config has the wrong {key}")
    return QuantaAdapter(
        _string(config, "repository", label="pinned Quanta config"),
        expected_commit=_string(config, "commit", label="pinned Quanta config"),
        expected_config_sha256=_string(
            config, "config_sha256", label="pinned Quanta config"
        ),
        expected_runner_sha256=_string(
            config, "runner_sha256", label="pinned Quanta config"
        ),
        expected_provider_identity=dict(context.provider_identity),
        output_dir=staging,
    )


def _stage_v2_arm(
    workspace: Path | str,
    *,
    arm: str,
    panel: PitPanel,
    pins: EvaluationIdentityPins,
    authority_guard: AuthorityGuard,
    development_capability: str,
    capability: str,
    staging_parent: Path | str,
    adapter_factory: Callable[[Path, _ExecutionContext], QuantaArmAdapter],
) -> StagedArmEvaluation:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evaluation workspace is not a directory")
    if arm not in ARMS:
        raise ValueError(f"unknown S2a v2 evaluation arm: {arm!r}")
    if not isinstance(panel, PitPanel):
        raise TypeError("evaluation panel must be a PitPanel")
    if not isinstance(pins, EvaluationIdentityPins):
        raise TypeError("evaluation pins must be EvaluationIdentityPins")
    if not isinstance(authority_guard, AuthorityGuard):
        raise TypeError("authority guard must be AuthorityGuard")
    if (
        authority_guard.workspace != root
        or authority_guard.base_lock_sha256 != pins.base_lock_sha256
    ):
        raise ValueError("authority guard is not bound to this workspace and base lock")
    development_receipt = authority_guard.verify_capability(
        development_capability,
        boundary="before_development_opening",
    )
    authority_receipt = authority_guard.verify_capability(
        capability,
        boundary="before_each_scientific_or_control_arm",
        arm=arm,
    )
    if authority_receipt.sequence <= development_receipt.sequence:
        raise ValueError("arm authority must be issued after development opening")
    authority_consumption_path = _consume_arm_authority(root, authority_receipt)
    context = _verify_execution_context(root, panel, pins)
    opening_authority = {
        "receipt_sha256": development_receipt.receipt_sha256,
        "sequence": development_receipt.sequence,
        "boundary": development_receipt.boundary,
        "authority_sha256": development_receipt.authority_sha256,
        "base_lock_sha256": development_receipt.base_lock_sha256,
        "capability_sha256": hashlib.sha256(
            development_capability.encode("utf-8")
        ).hexdigest(),
    }
    if context.development_opening.get("authority_receipt") != opening_authority:
        raise ValueError(
            "development opening is not bound to the consumed authority capability"
        )
    registered_input, input_path = _registered_input(root, arm, context)

    parent = Path(staging_parent).resolve(strict=True)
    if not parent.is_dir() or not parent.is_relative_to(root):
        raise ValueError("staging parent must be a contained existing directory")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{arm}.", suffix=".staging", dir=parent)
    ).resolve(strict=True)
    try:
        adapter = adapter_factory(staging, context)
        adapter.initialize_and_verify_provider()
        quanta_identity = _verify_adapter_identity(adapter, context)
        experiment_name = f"mirage_kan_{context.protocol_id}_{arm}"
        if arm == "alpha158_replay":
            input_record: dict[str, object] = {
                "kind": "official_alpha158",
                "path": None,
                "manifest_sha256": None,
                "factor_library": False,
            }
            metrics = adapter.evaluate_alpha158(
                experiment_name=experiment_name,
                output_name=arm,
                capture_report=True,
            )
        elif arm in _FACTOR_LIBRARY_ARMS:
            assert input_path is not None
            values, input_record = _load_factor_library(arm, input_path, panel, context)
            metrics = adapter.evaluate_panel(
                values,
                experiment_name=experiment_name,
                recorder_name=experiment_name,
                output_name=arm,
                capture_report=True,
            )
        else:
            assert input_path is not None
            values, input_record = _load_blackbox_control(input_path, context, panel)
            metrics = adapter.evaluate_panel(
                values,
                experiment_name=experiment_name,
                recorder_name=experiment_name,
                output_name=arm,
                capture_report=True,
            )
        if arm != "alpha158_replay":
            input_record["source_protocol_id"] = registered_input.get(
                "source_protocol_id"
            )
            input_record["mining_rebind_receipt_sha256"] = registered_input.get(
                "mining_rebind_receipt_sha256"
            )
        if not isinstance(metrics, Mapping):
            raise TypeError("Quanta arm metrics must be a mapping")
        diagnostic_files = adapter.write_portfolio_diagnostics(staging)
        _verify_diagnostic_calendar(staging, context.development_dates)
        artifact_index = _flat_file_index(staging)
        if set(diagnostic_files).difference(artifact_index):
            raise ValueError("Quanta diagnostic receipt names an absent staged file")
        for filename, expected_sha256 in diagnostic_files.items():
            if artifact_index[filename]["sha256"] != expected_sha256:
                raise ValueError(f"Quanta diagnostic hash mismatch: {filename}")
        evidence_class = context.config.get(
            "evidence_class", "prospective_development_screen"
        )
        if evidence_class not in {
            "prospective_development_screen",
            "corrective_adaptive_repeated_development_screen",
        }:
            raise ValueError("frozen config has an invalid evidence class")
        identity_pins: dict[str, object] = {
            "base_lock_sha256": pins.base_lock_sha256,
            "implementation_lock_sha256": pins.implementation_lock_sha256,
            "mining_manifest_sha256": pins.mining_manifest_sha256,
            "development_opening_sha256": pins.development_opening_sha256,
            "development_topology_sha256": pins.development_topology_sha256,
            "provider_identity": dict(context.provider_identity),
        }
        if pins.mining_rebind_receipt_sha256 is not None:
            identity_pins["mining_rebind_receipt_sha256"] = (
                pins.mining_rebind_receipt_sha256
            )
        manifest: dict[str, object] = {
            "schema_version": "mirage_s2a_quanta_evaluation_v2",
            "protocol_id": context.protocol_id,
            "stage": "S2a",
            "evidence_class": evidence_class,
            "arm": arm,
            "topology_key": f"evaluation:{arm}",
            "scientific_result": False,
            "final_claim_allowed": False,
            "formal_promotion_allowed": False,
            "input": input_record,
            "metrics": dict(metrics),
            "diagnostic_files": dict(diagnostic_files),
            "files": artifact_index,
            "identity_pins": identity_pins,
            "development_opening": {
                "path": str(context.development_opening_path),
                "sha256": pins.development_opening_sha256,
                "topology_sha256": pins.development_topology_sha256,
                "authority_receipt_sha256": development_receipt.receipt_sha256,
            },
            "authority_receipt": {
                "receipt_sha256": authority_receipt.receipt_sha256,
                "sequence": authority_receipt.sequence,
                "boundary": authority_receipt.boundary,
                "arm": authority_receipt.arm,
                "authority_sha256": authority_receipt.authority_sha256,
                "base_lock_sha256": authority_receipt.base_lock_sha256,
                "consumption_path": str(authority_consumption_path),
                "consumption_sha256": sha256_file(authority_consumption_path),
            },
            "quanta_identity": quanta_identity,
        }
        manifest_path = staging / "manifest.json"
        _write_exclusive(manifest_path, manifest)
        _flat_file_index(staging)
        return StagedArmEvaluation(
            arm=arm,
            topology_key=f"evaluation:{arm}",
            staging_path=staging,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def stage_v2_arm(
    workspace: Path | str,
    *,
    arm: str,
    panel: PitPanel,
    pins: EvaluationIdentityPins,
    authority_guard: AuthorityGuard,
    development_capability: str,
    capability: str,
    staging_parent: Path | str,
) -> StagedArmEvaluation:
    """Execute one production Quanta arm into an unpublished flat staging bundle."""
    return _stage_v2_arm(
        workspace,
        arm=arm,
        panel=panel,
        pins=pins,
        authority_guard=authority_guard,
        development_capability=development_capability,
        capability=capability,
        staging_parent=staging_parent,
        adapter_factory=_production_adapter,
    )


def _stage_v2_arm_for_test(
    workspace: Path | str,
    *,
    arm: str,
    panel: PitPanel,
    pins: EvaluationIdentityPins,
    authority_guard: AuthorityGuard,
    development_capability: str,
    capability: str,
    adapter_factory: AdapterFactory,
    staging_parent: Path | str,
) -> StagedArmEvaluation:
    """Inject a fake adapter only for hermetic tests; production cannot call this seam."""

    def adapt(staging: Path, context: _ExecutionContext) -> QuantaArmAdapter:
        return adapter_factory(staging, dict(context.provider_identity))

    return _stage_v2_arm(
        workspace,
        arm=arm,
        panel=panel,
        pins=pins,
        authority_guard=authority_guard,
        development_capability=development_capability,
        capability=capability,
        staging_parent=staging_parent,
        adapter_factory=adapt,
    )


__all__ = [
    "ARMS",
    "EvaluationIdentityPins",
    "QuantaArmAdapter",
    "StagedArmEvaluation",
    "stage_v2_arm",
]
