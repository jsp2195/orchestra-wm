import numpy as np

def desired_acceleration(vehicle, gap, lead_speed, advisory, yield_required):
    target = vehicle.desired_speed
    if vehicle.connected:
        target *= {-1: 0.15, 0: 0.7, 1: 1.15}[int(advisory)]
    elif yield_required:
        target = 1.0
    desired_gap = 4 + vehicle.reaction * vehicle.speed
    free = 1.8 * (1 - (vehicle.speed / max(target, 0.1)) ** 4)
    follow = 1.8 * (1 - (desired_gap / max(gap, 0.5)) ** 2) + 0.6*(lead_speed-vehicle.speed)
    return float(np.clip(min(free, follow) if gap < 35 else free, -4.0, 2.0))
