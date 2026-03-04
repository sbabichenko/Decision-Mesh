from __future__ import annotations

from typing import TYPE_CHECKING

from ._helpers import _r, _rn

if TYPE_CHECKING:
    from .edge import Edge
    from .face import Face
    from .mesh import DecisionMesh


class TreeNode:
    _seq = 0

    def __init__(self, mesh: DecisionMesh, parent: TreeNode = None, face: Face = None):
        self._id = TreeNode._seq; TreeNode._seq += 1
        self.parent = parent
        self.split_criterion = None
        self.face = face
        self.mesh = mesh
        # children (set after split)
        self.p: TreeNode | None = None  # '+' child
        self.m: TreeNode | None = None  # '-' child
        if face:
            self.face.addnode(self)

    # ---------- ids & basic predicates ----------
    @property
    def sid(self) -> str:
        return f"N{self._id:03d}"

    def is_leaf(self) -> bool:
        return self.split_criterion is None

    # ---------- structure helpers ----------
    def children(self) -> dict[str, TreeNode]:
        out = {}
        if self.p is not None: out['+'] = self.p
        if self.m is not None: out['-'] = self.m
        return out

    @property
    def depth(self) -> int:
        d, cur = 0, self.parent
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    @property
    def path(self) -> str:
        """String of '+'/'-' from root to this node."""
        bits = []
        node = self
        while node.parent is not None:
            bits.append('+' if node.parent.p is node else '-')
            node = node.parent
        return ''.join(reversed(bits))

    # ---------- tree ops ----------
    def split(self, edge: Edge, plus: Face, minus: Face):
        self.split_criterion = {'normal': edge.normal, 'intercept': edge.intercept}
        self.p = TreeNode(self.mesh, self, plus)
        self.m = TreeNode(self.mesh, self, minus)
        # maintain leaf set
        self.mesh.leaves.remove(self)
        self.mesh.leaves.add(self.p)
        self.mesh.leaves.add(self.m)

    # ---------- navigation ----------
    def child_for(self, sign: str) -> TreeNode | None:
        if sign == '+': return self.p
        if sign == '-': return self.m
        raise ValueError("child_for(sign): sign must be '+' or '-'")

    def follow(self, path, *, strict: bool = True, default=None) -> TreeNode | None:
        """
        Follow a path like '+-++---' (or any iterable of '+'/'-') from this node.
        If strict=True, raise on errors; else return `default`.
        """
        # Accept strings or any iterable of chars
        steps = path if isinstance(path, str) else list(path)

        # Validate characters once up front
        bad = [c for c in steps if c not in ('+', '-')]
        if bad:
            if strict:
                raise ValueError(f"Invalid step(s) in path: {bad!r}")
            return default

        node = self
        for i, ch in enumerate(steps):
            if node.is_leaf():
                if strict:
                    raise ValueError(f"Stopped at depth {node.depth} ({node.sid} is a leaf); remaining='{''.join(steps[i:])}'")
                return default
            nxt = node.child_for(ch)
            if nxt is None:
                if strict:
                    raise KeyError(f"No child '{ch}' from {node.sid} at step {i}")
                return default
            node = nxt
        return node

    def __getitem__(self, path: str) -> TreeNode:
        return self.follow(path, strict=True)

    # ---------- repr ----------
    def __repr__(self):
        if self.is_leaf():
            face_str = getattr(self.face, "sid", None)
            return f"<{self.sid} Leaf depth={self.depth} path='{self.path}' face={face_str}>"

        sc = self.split_criterion or {}
        n = _rn(sc.get("normal", "?"))
        b = _r(sc.get("intercept", "?"))
        kids = ''.join(sorted(self.children().keys())) or "-"
        return (f"<{self.sid} Split depth={self.depth} path='{self.path}' "
                f"n={n} b={b} kids={kids}>")

    __str__ = __repr__
