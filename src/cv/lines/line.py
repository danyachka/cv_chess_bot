from dataclasses import dataclass
import numpy as np
from typing import Final
import time

radian: Final[float] = np.pi

parts: Final[int] = 10
angle_threshold: Final[float] = radian / parts # rad

distance_threshold: Final[float] = 30 # pixels


@dataclass
class Line:
    r: int
    theta: float

    def __init__(self, line):
        self.r = line[0][0]
        self.theta = line[0][1]

    def get_diff_angle(self, theta: float) -> float:
        lower = min(theta, self.theta)
        higher = max(theta, self.theta)

        if higher - lower > radian/2:
            return (radian*2 + lower) - (radian + higher)
        return higher - lower

    def is_parallel(self, theta: float) -> bool:
        return self.get_diff_angle(theta) < angle_threshold/2
    
    def is_to_close(self, other) -> bool:
        return abs(self.r - other.r) < distance_threshold
    
    def calc_intersection(self, other) -> tuple[np.float64, np.float64]:
        A = np.array([
            [np.cos(self.theta), np.sin(self.theta)], 
            [np.cos(other.theta), np.sin(other.theta)]
        ])
        b = np.array([self.r, other.r])
        x, y = np.linalg.lstsq(A, b)[0]
        return x, y


@dataclass
class LineGroup:
    lines: list[Line]
    angle: float
    intersections: list[tuple[int, int]]

    def __init__(self, angle: float):
        self.lines = []
        self.angle = angle
        self.intersections = []

    def append(self, line: Line, shape) -> bool:
        if not line.is_parallel(self.angle): 
            return False
        
        for i, appended in enumerate(self.lines):
            inter = appended.calc_intersection(line)
            self.intersections.append((int(inter[0]), int(inter[1])))
            if not _is_non_parallel_intersection(inter, shape) and not appended.is_to_close(line):
                continue
            self.lines[i] = _get_closer_line(self.angle, appended, line)
            return True
        
        self.lines.append(line)
        return True
    
    def is_perpendicular(self, other) -> bool:
        lower = min(other.angle, self.theta)
        higher = max(other.angle, self.theta)
        return -angle_threshold/2 < higher - lower < angle_threshold/2


def group_lines(lines: list[Line], shape) -> list[LineGroup]:
    start = time.time()
    groups: list[LineGroup] = [
        LineGroup(0), LineGroup(radian/2)
        # LineGroup(i * radian/parts) for i in range(parts)
    ]

    for line in lines:
        for group in groups:
            if group.append(line, shape):
                break
    print(f"Grouping elapsed = {time.time() - start}")
            
    for group in groups:
        print(f'Group {group.angle}: {len(group.lines)}')
            
    return [groups[0], groups[int(-1)]]



def _is_non_parallel_intersection(inter, shape) -> bool:
    print(inter)
    x_s, y_s, _ = shape
    x, y = inter
    return (0 < x < x_s) and (0 < y < y_s)


def _get_closer_line(angle: float, line1: Line, line2: Line) -> Line:
    diff1 = line1.get_diff_angle(angle)
    diff2 = line2.get_diff_angle(angle)
    return line1 if diff1 < diff2 else line2
