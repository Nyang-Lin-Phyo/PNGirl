"""
Shared drag-and-drop state.

DragItem  — one PNG currently being dragged (hand or mouse)
FallItem  — one PNG currently falling off screen
AnchorSlot — what's snapped onto each anchor point

All mutation happens on the main thread (UI events + frame signal handler).
The worker reads these as needed each frame via get_drag_state_fn.
"""

from dataclasses import dataclass, field
from typing import Optional
import threading


@dataclass
class DragItem:
    png_path: str           # absolute path to the PNG
    cx: int                 # current center x in frame pixel space
    cy: int                 # current center y in frame pixel space
    source: str             # "hand" or "mouse"
    hand_id: Optional[int]  # 0 or 1 for hand source, None for mouse


@dataclass
class FallItem:
    png_path: str
    cx: float
    cy: float
    vy: float = 18.0        # pixels per frame, accelerates


# anchor keys match the settings keys prefix
ANCHOR_KEYS = ["head", "left_shoulder", "right_shoulder"]


class DragState:
    """Thread-safe container for all drag/fall/anchor state."""

    def __init__(self):
        self._lock = threading.Lock()

        # hand_id -> DragItem  (at most 2 simultaneous drags)
        self._hand_drags: dict[int, DragItem] = {}

        # mouse drag (at most 1)
        self._mouse_drag: Optional[DragItem] = None

        # anchor_key -> png_path or None
        self._anchor_slots: dict[str, Optional[str]] = {
            k: None for k in ANCHOR_KEYS
        }

        # list of falling PNGs
        self._falling: list[FallItem] = []

    # ── Drag start ────────────────────────────────────────────────────────

    def start_hand_drag(self, hand_id: int, png_path: str, cx: int, cy: int):
        with self._lock:
            self._hand_drags[hand_id] = DragItem(
                png_path=png_path, cx=cx, cy=cy,
                source="hand", hand_id=hand_id)

    def start_mouse_drag(self, png_path: str, cx: int, cy: int):
        with self._lock:
            self._mouse_drag = DragItem(
                png_path=png_path, cx=cx, cy=cy,
                source="mouse", hand_id=None)

    # ── Drag move ─────────────────────────────────────────────────────────

    def move_hand_drag(self, hand_id: int, cx: int, cy: int):
        with self._lock:
            if hand_id in self._hand_drags:
                self._hand_drags[hand_id].cx = cx
                self._hand_drags[hand_id].cy = cy

    def move_mouse_drag(self, cx: int, cy: int):
        with self._lock:
            if self._mouse_drag:
                self._mouse_drag.cx = cx
                self._mouse_drag.cy = cy

    # ── Drag release ──────────────────────────────────────────────────────

    def release_hand_drag(self, hand_id: int, anchor_positions: dict,
                          snap_threshold: int) -> Optional[str]:
        """Returns the anchor_key snapped to, or None if falling."""
        with self._lock:
            item = self._hand_drags.pop(hand_id, None)
            if item is None:
                return None
            return self._resolve_release(item, anchor_positions, snap_threshold)

    def release_mouse_drag(self, anchor_positions: dict,
                           snap_threshold: int) -> Optional[str]:
        with self._lock:
            item = self._mouse_drag
            self._mouse_drag = None
            if item is None:
                return None
            return self._resolve_release(item, anchor_positions, snap_threshold)

    def _resolve_release(self, item: DragItem, anchor_positions: dict,
                         snap_threshold: int) -> Optional[str]:
        """Must be called with lock held."""
        best_key, best_dist = None, float("inf")
        for key, (ax, ay) in anchor_positions.items():
            dist = ((item.cx - ax) ** 2 + (item.cy - ay) ** 2) ** 0.5
            if dist < snap_threshold and dist < best_dist:
                best_key, best_dist = key, dist

        if best_key:
            # Eject whatever was there before
            old = self._anchor_slots.get(best_key)
            if old:
                self._falling.append(FallItem(png_path=old,
                                               cx=anchor_positions[best_key][0],
                                               cy=anchor_positions[best_key][1]))
            self._anchor_slots[best_key] = item.png_path
            return best_key
        else:
            self._falling.append(FallItem(png_path=item.png_path,
                                           cx=item.cx, cy=item.cy))
            return None

    # ── Anchor direct set (for clearing) ─────────────────────────────────

    def clear_anchor(self, anchor_key: str):
        with self._lock:
            self._anchor_slots[anchor_key] = None

    # ── Fall tick (called each frame by worker) ───────────────────────────

    def tick_falling(self, frame_height: int):
        """Advance all falling items. Call with no lock — worker only."""
        with self._lock:
            for f in self._falling:
                f.vy += 1.2          # gravity
                f.cy += f.vy
            self._falling = [f for f in self._falling if f.cy < frame_height + 200]

    # ── Snapshot for worker (lock-free read of copies) ────────────────────

    def snapshot(self):
        with self._lock:
            return (
                list(self._hand_drags.values()) + ([self._mouse_drag] if self._mouse_drag else []),
                dict(self._anchor_slots),
                list(self._falling),
            )

    # ── Mouse drag query ──────────────────────────────────────────────────

    def has_mouse_drag(self) -> bool:
        with self._lock:
            return self._mouse_drag is not None

    def mouse_drag_path(self) -> Optional[str]:
        with self._lock:
            return self._mouse_drag.png_path if self._mouse_drag else None