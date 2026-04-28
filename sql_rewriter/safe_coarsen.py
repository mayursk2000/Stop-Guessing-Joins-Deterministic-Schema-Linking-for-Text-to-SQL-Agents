"""SafeCoarsen — Hierarchy-Respecting Safe Query Repair.

The algorithm takes a set of output groups, a hierarchy, a safety predicate,
a merge function, and a label function. It returns a coarsening map from
original group values to coarsened labels such that every released group
satisfies the predicate.

The caller is responsible for defining what "safe" means (predicate),
how summaries combine when groups merge (merge), and what merged groups
are called (label). SafeCoarsen knows none of these things.
"""

from __future__ import annotations
from typing import Callable


# ---------------------------------------------------------------------------
# Result constants
# ---------------------------------------------------------------------------

SAFE      = "SAFE"
UNSAFE    = "UNSAFE"
SUPPRESSED = "SUPPRESSED"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def safe_coarsen(
    groups: list[dict],
    hierarchy: dict[str, str | None],
    predicate: Callable[[dict], bool],
    merge: Callable[[dict, dict], dict],
    label: Callable[[str, list[dict]], str],
) -> dict[str, str]:
    """Compute a hierarchy-respecting safe coarsening map.

    Args:
        groups:     List of group dicts. Each must have a "value" key
                    (the original grouping value) plus any summary fields
                    the predicate and merge functions need.
        hierarchy:  Dict mapping node name → parent name (or None for root).
                    Must contain entries for all group values plus any
                    intermediate and root nodes needed for merging.
        predicate:  S(group) → bool. Returns True if the group is safe to
                    release. The algorithm never inspects group summaries
                    directly — all safety logic lives here.
        merge:      merge(g1, g2) → g. Combines two groups into one.
                    The result must have a "value" key. All summary fields
                    needed by predicate must be present in the result.
        label:      label(parent_node, children) → str. Returns the display
                    name for a merged group. Called with the parent node name
                    and the list of child groups being merged.

    Returns:
        A dict mapping each original group value (str) to a coarsened label
        (str). Safe groups that were not merged map to their own value.
        Groups that could not be made safe even at the root are mapped to
        None (suppressed).
    """
    if not groups:
        return {}

    # ------------------------------------------------------------------
    # Step 0: build working state
    # Each node in the working tree has:
    #   value       — node name (matches hierarchy key)
    #   status      — SAFE / UNSAFE / SUPPRESSED
    #   summaries   — dict of aggregate summaries (from input or via merge)
    #   origin      — set of original group values this node covers
    #   emitted     — True once this node has been committed to output
    # ------------------------------------------------------------------

    nodes: dict[str, dict] = {}

    # Seed from input groups (leaves)
    for g in groups:
        v = g["value"]
        node = dict(g)  # copy so we don't mutate caller's data
        node["status"]   = SAFE if predicate(g) else UNSAFE
        node["origin"]   = {v}
        node["emitted"]  = False
        nodes[v] = node

    # ------------------------------------------------------------------
    # Step 1: collect all ancestors that appear in the hierarchy and
    # are needed to connect group leaves to a root. We build these
    # on-demand as we go bottom-up.
    # ------------------------------------------------------------------

    def get_parent(node_name: str) -> str | None:
        return hierarchy.get(node_name)

    def get_children_in_hierarchy(parent_name: str) -> list[str]:
        return [k for k, v in hierarchy.items() if v == parent_name]

    # Determine which hierarchy nodes are relevant (ancestors of input groups)
    relevant: set[str] = set()
    for g in groups:
        v = g["value"]
        current = v
        while current is not None:
            relevant.add(current)
            current = get_parent(current)

    # Build the bottom-up processing order: leaves first, root last
    # We do a topological sort by repeatedly finding nodes whose children
    # are all already ordered.
    def build_levels() -> list[list[str]]:
        """Return nodes grouped by level, leaves first."""
        # level[node] = max depth from any leaf
        level: dict[str, int] = {}

        def depth(n: str) -> int:
            if n in level:
                return level[n]
            children = [c for c in get_children_in_hierarchy(n) if c in relevant]
            if not children:
                level[n] = 0
            else:
                level[n] = 1 + max(depth(c) for c in children)
            return level[n]

        for n in relevant:
            depth(n)

        max_level = max(level.values()) if level else 0
        return [
            [n for n in relevant if level[n] == lvl]
            for lvl in range(max_level + 1)
        ]

    levels = build_levels()

    # ------------------------------------------------------------------
    # Step 2: bottom-up merge pass
    # For each parent (level > 0), look at its children that are leaves
    # (i.e., came from input groups or were already processed).
    # ------------------------------------------------------------------

    output_map: dict[str, str] = {}  # original_value → coarsened_label

    def emit(node: dict, coarsened_label: str) -> None:
        """Mark a node as emitted and record all its origins in output_map."""
        node["emitted"]  = True
        node["status"]   = SAFE
        for orig in node["origin"]:
            output_map[orig] = coarsened_label

    def merge_group_list(group_list: list[dict], parent_name: str) -> dict:
        """Merge a list of groups into one, labeled by parent."""
        if len(group_list) == 1:
            merged = dict(group_list[0])
        else:
            merged = group_list[0]
            for g in group_list[1:]:
                merged = merge(merged, g)

        merged["value"]  = parent_name
        merged["origin"] = set().union(*(g["origin"] for g in group_list))
        merged["status"] = SAFE if predicate(merged) else UNSAFE
        merged["emitted"] = False
        return merged

    # Process each level from leaves upward (skip level 0 — those are leaves
    # and get handled when their parents are processed)
    for level_nodes in levels[1:]:
        for parent_name in level_nodes:
            # Gather children of this parent that are in our working set
            child_names = [
                c for c in get_children_in_hierarchy(parent_name)
                if c in relevant
            ]

            if not child_names:
                continue

            # Ensure all children exist in nodes dict (may be intermediate)
            child_nodes = []
            for c in child_names:
                if c in nodes:
                    child_nodes.append(nodes[c])
                # If a child is not in nodes, it had no input groups beneath
                # it — skip it (it contributes nothing to merge)

            if not child_nodes:
                continue

            # Emit any already-safe, not-yet-emitted leaves directly
            safe_children   = [n for n in child_nodes if n["status"] == SAFE and not n["emitted"]]
            unsafe_children = [n for n in child_nodes if n["status"] == UNSAFE]

            if not unsafe_children:
                # All children safe — emit each at original granularity
                for n in safe_children:
                    emit(n, label(n["value"], [n]))
                continue

            # Try 1: merge only the unsafe siblings
            if len(unsafe_children) > 1:
                merged_unsafe = merge_group_list(unsafe_children, parent_name)
            else:
                merged_unsafe = dict(unsafe_children[0])
                merged_unsafe["value"]  = parent_name
                merged_unsafe["origin"] = unsafe_children[0]["origin"]
                merged_unsafe["status"] = SAFE if predicate(merged_unsafe) else UNSAFE
                merged_unsafe["emitted"] = False

            if merged_unsafe["status"] == SAFE:
                # Emit safe children at original granularity
                for n in safe_children:
                    emit(n, label(n["value"], [n]))
                # Emit merged unsafe group under parent label
                merged_label = label(parent_name, unsafe_children)
                emit(merged_unsafe, merged_label)
                nodes[parent_name] = merged_unsafe
                continue

            # Try 2: incrementally add safe siblings until predicate passes.
            # We add them in their original order — the caller can pre-sort
            # the groups list if a specific order is desired.
            incremental = dict(merged_unsafe)
            incremental["origin"] = set(merged_unsafe["origin"])
            pulled_in = []

            for safe_n in safe_children:
                incremental = merge(incremental, safe_n)
                incremental["origin"] = incremental.get("origin", set()) | safe_n["origin"]
                incremental["status"] = SAFE if predicate(incremental) else UNSAFE
                incremental["emitted"] = False
                pulled_in.append(safe_n)
                if incremental["status"] == SAFE:
                    break

            if incremental["status"] == SAFE:
                # Emit safe children NOT pulled in at original granularity
                remaining_safe = [n for n in safe_children if n not in pulled_in]
                for n in remaining_safe:
                    emit(n, label(n["value"], [n]))
                # Emit the merged group (unsafe + pulled-in safe siblings)
                merged_label = label(parent_name, [merged_unsafe] + pulled_in)
                emit(incremental, merged_label)
                for n in pulled_in:
                    n["emitted"] = True
                nodes[parent_name] = incremental
                continue

            # Try 3: merge ALL children (incremental already covers all safe children
            # since we iterated through all of them above — re-use if exhausted)
            all_children = safe_children + unsafe_children
            merged_all = merge_group_list(all_children, parent_name)

            if merged_all["status"] == SAFE:
                merged_label = label(parent_name, all_children)
                emit(merged_all, merged_label)
                # mark all constituent child nodes as emitted so step 3
                # doesn't re-emit them at original granularity
                for n in all_children:
                    n["emitted"] = True
                nodes[parent_name] = merged_all
                continue

            # Try 3: push parent up — mark parent as UNSAFE with combined summaries
            # Safe children that haven't been emitted yet lose their granularity
            merged_all["status"] = UNSAFE
            nodes[parent_name] = merged_all
            # (will be handled when grandparent is processed)

    # ------------------------------------------------------------------
    # Step 3: collect remaining unemitted nodes
    # Any node that reached this point without being emitted is either:
    # (a) a safe leaf with no parent in the hierarchy → emit at own label
    # (b) an unsafe root → suppress
    # ------------------------------------------------------------------

    for node_name, node in nodes.items():
        if node["emitted"]:
            continue

        if node["status"] == SAFE:
            emit(node, label(node_name, [node]))
        else:
            # Check if this is the root (no parent or parent not in relevant)
            parent = get_parent(node_name)
            if parent is None or parent not in relevant:
                # Root-level unsafe → suppress
                node["status"] = SUPPRESSED
                for orig in node["origin"]:
                    output_map[orig] = None

    # ------------------------------------------------------------------
    # Step 4: ensure every input group has an entry (paranoia check)
    # ------------------------------------------------------------------

    for g in groups:
        if g["value"] not in output_map:
            # Group was not in hierarchy at all — emit as-is if safe, else suppress
            output_map[g["value"]] = g["value"] if predicate(g) else None

    return output_map


# ---------------------------------------------------------------------------
# Helper: build hierarchy dict from a list of (node, parent) tuples
# ---------------------------------------------------------------------------

def build_hierarchy(rows: list[tuple[str, str | None]]) -> dict[str, str | None]:
    """Build a hierarchy dict from (node, parent) pairs.

    Args:
        rows: List of (node_name, parent_name) tuples. Root node has parent=None.

    Returns:
        Dict mapping node_name → parent_name.
    """
    return {node: parent for node, parent in rows}