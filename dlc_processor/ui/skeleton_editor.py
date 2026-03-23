"""Interactive skeleton editor dialog for defining keypoint connections.

Opens a popup where the user can visually connect bodypart nodes to define
skeleton edges.  Nodes are arranged in a circle and can be dragged freely.
Click two nodes sequentially to create an edge; right-click an edge to remove it.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ── Colours (dark theme) ──────────────────────────────────────────────────────

_BG          = QColor("#1e1e2e")
_NODE_FILL   = QColor("#313244")
_NODE_BORDER = QColor("#cba6f7")
_EDGE_COLOR  = QColor("#89b4fa")
_SELECT_CLR  = QColor("#f38ba8")
_TEXT_CLR    = QColor("#cdd6f4")

_NODE_RADIUS = 20


# ── Draggable node ────────────────────────────────────────────────────────────

class _BodypartNode(QGraphicsEllipseItem):
    """A draggable labelled circle representing a single bodypart."""

    def __init__(self, name: str, x: float, y: float, editor: "SkeletonEditorDialog"):
        r = _NODE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(x, y)
        self.name = name
        self._editor = editor

        self.setBrush(QBrush(_NODE_FILL))
        self.setPen(QPen(_NODE_BORDER, 2))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        # Label below the node
        self._label = QGraphicsSimpleTextItem(name, self)
        self._label.setBrush(QBrush(_TEXT_CLR))
        font = QFont("Segoe UI", 8)
        self._label.setFont(font)
        br = self._label.boundingRect()
        self._label.setPos(-br.width() / 2, r + 2)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self._editor._update_edge_positions()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._editor._on_node_clicked(self)
        super().mousePressEvent(event)

    def set_selected_look(self, selected: bool) -> None:
        self.setPen(QPen(_SELECT_CLR if selected else _NODE_BORDER, 2))


# ── Clickable edge ────────────────────────────────────────────────────────────

class _EdgeLine(QGraphicsLineItem):
    """A skeleton edge line that can be right-clicked to remove."""

    def __init__(self, bp1: str, bp2: str, editor: "SkeletonEditorDialog"):
        super().__init__()
        self.bp1 = bp1
        self.bp2 = bp2
        self._editor = editor
        self.setPen(QPen(_EDGE_COLOR, 2.5))
        self.setZValue(1)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._editor._remove_edge(self.bp1, self.bp2)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(_SELECT_CLR, 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(_EDGE_COLOR, 2.5))
        super().hoverLeaveEvent(event)


# ── Dialog ────────────────────────────────────────────────────────────────────

class SkeletonEditorDialog(QDialog):
    """Modal dialog for interactively editing skeleton connections."""

    def __init__(
        self,
        bodyparts: list[str],
        existing_edges: Optional[list[tuple[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Skeleton Editor")
        self.setMinimumSize(820, 520)
        self.setStyleSheet(
            f"QDialog {{ background: {_BG.name()}; }}"
            f"QLabel, QListWidget, QPushButton {{ color: #cdd6f4; }}"
            f"QListWidget {{ background: #313244; border: 1px solid #45475a;"
            f"  border-radius: 4px; }}"
            f"QPushButton {{ background: #313244; border: 1px solid #45475a;"
            f"  border-radius: 4px; padding: 6px 14px; }}"
            f"QPushButton:hover {{ background: #45475a; }}"
        )

        self._bodyparts = list(bodyparts)
        self._edges: list[tuple[str, str]] = []
        self._nodes: dict[str, _BodypartNode] = {}
        self._edge_items: dict[tuple[str, str], _EdgeLine] = {}
        self._selected_node: Optional[_BodypartNode] = None

        self._build_ui()
        self._place_nodes()

        if existing_edges:
            for bp1, bp2 in existing_edges:
                if bp1 in self._nodes and bp2 in self._nodes:
                    self._add_edge(bp1, bp2)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(8)

        # Left — bodypart list
        left = QVBoxLayout()
        left.addWidget(_section_label("Bodyparts"))
        self._bp_list = QListWidget()
        for bp in self._bodyparts:
            self._bp_list.addItem(bp)
        self._bp_list.setMaximumWidth(160)
        left.addWidget(self._bp_list, 1)
        root.addLayout(left)

        # Centre — canvas
        centre = QVBoxLayout()
        centre.addWidget(_section_label("Canvas  (click two nodes to connect)"))
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor("#11111b")))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumSize(400, 400)
        centre.addWidget(self._view, 1)
        root.addLayout(centre, 1)

        # Right — connections list + buttons
        right = QVBoxLayout()
        right.addWidget(_section_label("Connections"))
        self._conn_list = QListWidget()
        self._conn_list.setMaximumWidth(200)
        right.addWidget(self._conn_list, 1)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected_connection)
        right.addWidget(btn_remove)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_all_edges)
        right.addWidget(btn_clear)

        right.addSpacing(12)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        right.addLayout(btn_row)

        root.addLayout(right)

    def _place_nodes(self) -> None:
        """Arrange bodypart nodes in a circle on the canvas."""
        n = len(self._bodyparts)
        if n == 0:
            return
        cx, cy = 180.0, 180.0
        radius = 140.0

        for i, bp in enumerate(self._bodyparts):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            node = _BodypartNode(bp, x, y, self)
            self._scene.addItem(node)
            self._nodes[bp] = node

    # ── Node click logic ──────────────────────────────────────────────────────

    def _on_node_clicked(self, node: _BodypartNode) -> None:
        if self._selected_node is None:
            self._selected_node = node
            node.set_selected_look(True)
        elif self._selected_node is node:
            # Deselect
            node.set_selected_look(False)
            self._selected_node = None
        else:
            bp1 = self._selected_node.name
            bp2 = node.name
            self._selected_node.set_selected_look(False)
            self._selected_node = None
            self._add_edge(bp1, bp2)

    # ── Edge management ───────────────────────────────────────────────────────

    def _edge_key(self, bp1: str, bp2: str) -> tuple[str, str]:
        return (min(bp1, bp2), max(bp1, bp2))

    def _add_edge(self, bp1: str, bp2: str) -> None:
        key = self._edge_key(bp1, bp2)
        if key in self._edge_items:
            return  # already exists

        line = _EdgeLine(key[0], key[1], self)
        n1, n2 = self._nodes[key[0]], self._nodes[key[1]]
        line.setLine(n1.pos().x(), n1.pos().y(), n2.pos().x(), n2.pos().y())
        self._scene.addItem(line)

        self._edges.append(key)
        self._edge_items[key] = line
        self._refresh_conn_list()

    def _remove_edge(self, bp1: str, bp2: str) -> None:
        key = self._edge_key(bp1, bp2)
        item = self._edge_items.pop(key, None)
        if item is not None:
            self._scene.removeItem(item)
        if key in self._edges:
            self._edges.remove(key)
        self._refresh_conn_list()

    def _remove_selected_connection(self) -> None:
        row = self._conn_list.currentRow()
        if 0 <= row < len(self._edges):
            bp1, bp2 = self._edges[row]
            self._remove_edge(bp1, bp2)

    def _clear_all_edges(self) -> None:
        for key in list(self._edge_items.keys()):
            item = self._edge_items.pop(key)
            self._scene.removeItem(item)
        self._edges.clear()
        self._refresh_conn_list()

    def _update_edge_positions(self) -> None:
        """Redraw all edge lines after nodes have moved."""
        for key, line in self._edge_items.items():
            n1 = self._nodes[key[0]]
            n2 = self._nodes[key[1]]
            line.setLine(n1.pos().x(), n1.pos().y(), n2.pos().x(), n2.pos().y())

    def _refresh_conn_list(self) -> None:
        self._conn_list.clear()
        for bp1, bp2 in self._edges:
            self._conn_list.addItem(f"{bp1}  -  {bp2}")

    # ── Public result ─────────────────────────────────────────────────────────

    def get_edges(self) -> list[tuple[str, str]]:
        """Return the list of skeleton edges as (bodypart1, bodypart2) tuples."""
        return list(self._edges)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section_label(text: str) -> "QLabel":
    from PySide6.QtWidgets import QLabel
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
    return lbl
