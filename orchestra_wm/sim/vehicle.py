from dataclasses import dataclass

@dataclass
class Vehicle:
    id: int
    lane: int
    s: float
    speed: float
    connected: bool
    desired_speed: float
    aggressiveness: float
    reaction: float
    acceleration: float = 0.0
    active: bool = True
    maneuver: str = 'follow'
    age: float = 0.0
    generation: int = 0
    route: tuple = ()
