import numpy as np

def integrate(vehicle, acceleration, dt):
    # Bound jerk as well as acceleration; no direct steering controls.
    vehicle.acceleration += float(np.clip(acceleration-vehicle.acceleration, -3*dt, 3*dt))
    old = vehicle.speed
    vehicle.speed = float(np.clip(old + vehicle.acceleration*dt, 0, 18))
    vehicle.s += (old + vehicle.speed)*0.5*dt
    vehicle.age += dt
