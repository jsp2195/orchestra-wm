"""Directed route-lane graph; geometry is measured in synthetic metres."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Lane:
    id: int
    centerline: np.ndarray
    speed_preference: float = 10.0
    successors: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())

    def pose(self, s):
        segments = np.diff(self.centerline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        i = min(int(np.searchsorted(np.cumsum(lengths), s, side='right')), len(lengths)-1)
        local = s - lengths[:i].sum()
        direction = segments[i] / lengths[i]
        return self.centerline[i] + np.clip(local, 0, lengths[i]) * direction, direction

@dataclass
class RoadGraph:
    lanes: list
    intersections: np.ndarray
    family: str

def make_road(family='intersection', scale=1.0):
    paths = []
    if family == 'merge':
        paths = [[[-65, -18], [-20, -18], [5, 0], [65, 0]], [[-65, 8], [-20, 8], [5, 0], [65, 0]]]
        nodes = [[5, 0]]
    else:
        nodes = {'intersection': [[0, 0]], 'corridor': [[-25, 0], [25, 0]],
                 'grid': [[-25,-25],[-25,25],[25,-25],[25,25]]}[family]
        ys = sorted(set(y for _,y in nodes))
        xs = sorted(set(x for x,_ in nodes))
        for y in ys:
            paths += [[[-75,y-2],[75,y-2]], [[75,y+2],[-75,y+2]]]
        for x in xs:
            paths += [[[x+2,-75],[x+2,75]], [[x-2,75],[x-2,-75]]]
    lanes = [Lane(i,np.asarray(p,dtype=float)*scale) for i,p in enumerate(paths)]
    for lane in lanes:
        lane.conflicts = [other.id for other in lanes if other.id != lane.id]
    return RoadGraph(lanes,np.asarray(nodes,dtype=float)*scale,family)
