"""
Topology
--------
  All active sensor heads (RADAR, RF1, RF2) sit at the coordinate ORIGIN (0,0,0).
  The 100 acoustic nodes form a 10×10 regular grid over the full scan area, placed
  at ground level (z=0) with known coordinates.  This layout lets the downstream
  preprocessing module estimate the direction-of-arrival from the acoustic network.

Scan area
---------
  Square, side = AREA_SIDE metres (default 15 000 m = 15 km).
  Drones fly at altitudes [AREA_Z_MIN, AREA_Z_MAX].
  Origin is the centre of the square.

Adversarial / spoofing injection
---------------------------------
  Two independent attack modes can be active simultaneously:
    1. RF2 identity spoofing  – the claimed_id is replaced by a random string
       and fing_similarity is set low.
    2. Ghost injection        – the RF2 worker occasionally emits observations
       that carry NO corresponding real drone (a "phantom" target injected by
       an attacker to saturate the tracker).  These observations have
       drone_id = "GHOST_xx" and telemetry coordinates outside any real track.
  The acoustic network is deliberately NOT affected by these attacks, making it
  the trusted fall-back modality when RF2 is unreliable.

Pipeline interface
------------------
  ObservationQueue is the single handover point between this module and any
  downstream processor.  Every observation type is a frozen dataclass that
  carries a `modality` tag and a `sensor_pos` field (acoustic nodes only) so
  the fusion module always knows where each observation was produced.

    from data_generator import SensorDataGenerator, ObservationQueue
    gen = SensorDataGenerator(n_drones=3, seed=0, sim_speed=5.0)
    q   = ObservationQueue()
    gen.start(q)
    while True:
        obs = q.get(timeout=1.0)   # blocks
        process(obs)
    gen.stop()
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────────────────

# Scan area – 15 km × 15 km square, origin at centre
AREA_SIDE   : float = 15_000.0   # m
HALF        : float = AREA_SIDE / 2.0

AREA_Z_MIN  : float = 20.0       # m  – min drone altitude
AREA_Z_MAX  : float = 1000.0      # m  – max drone altitude

# Acoustic grid – 10 × 10 = 100 nodes uniformly tiled over the area, z = 0
ACOUSTIC_GRID_N : int = 10        # nodes per side

# Sampling rates (different, non-integer ratios → temporal misalignment)
RADAR_DT    : float = 0.10       # 10 Hz
RF1_DT      : float = 0.25       #  4 Hz
RF2_DT      : float = 0.50       #  2 Hz
ACOUSTIC_DT : float = 0.05       # 20 Hz

# Noise parameters 
RADAR_RANGE_NOISE   : float = 5.0     # m
RADAR_AZ_NOISE      : float = 0.4     # deg
RADAR_EL_NOISE      : float = 0.4     # deg  (elevation)
RADAR_VEL_NOISE     : float = 0.3     # m/s
RADAR_SNR_BASE      : float = 25.0    # dB at 1 km

RF1_DOA_NOISE       : float = 1.5     # deg
RF1_SS_NOISE        : float = 2.5     # dBm

RF2_POS_NOISE       : float = 8.0     # m  
RF2_VEL_NOISE       : float = 0.4     # m/s
RF2_ACC_NOISE       : float = 0.15    # m/s²
RF2_SNR_BASE        : float = 18.0    # dB

ACOUSTIC_INT_NOISE  : float = 5.0     # dB SPL
ACOUSTIC_THRESHOLD  : float = 35.0    # dB SPL  (activation gate)
ACOUSTIC_SRC_LEVEL  : float = 108.0   # dB SPL @ 1 m  (effective multirotor source level)
ACOUSTIC_ARRAY_MAX_RANGE: float = 1000.0  # m  practical node activation range

# Packet loss probabilities
RADAR_DROP_PROB     : float = 0.02
RF1_DROP_PROB       : float = 0.04
RF2_DROP_PROB       : float = 0.06
ACOUSTIC_DROP_PROB  : float = 0.01

# Adversarial parameters
RF2_SPOOF_PROB      : float = 0.0    # per-packet identity-spoof probability
RF2_GHOST_PROB      : float = 0.0    # per-cycle ghost-injection probability

# Drone kinematics
MAX_SPEED       : float = 80.0    # m/s
MAX_ACC         : float = 9.0     # m/s²
WAYPOINT_RADIUS : float = 30.0    # m

# ─────────────────────────────────────────────────────────────────────────────
# Acoustic node positions
# ─────────────────────────────────────────────────────────────────────────────

def _build_acoustic_grid(n: int = ACOUSTIC_GRID_N) -> List[Tuple[str, np.ndarray]]:
    """Return list of (sensor_id, position_3d) for a regular n×n grid."""
    step = AREA_SIDE / (n - 1)
    nodes = []
    for row in range(n):
        for col in range(n):
            x = -HALF + col * step
            y = -HALF + row * step
            sid = f"ACST_{row:02d}{col:02d}"
            nodes.append((sid, np.array([x, y, 0.0])))
    return nodes

ACOUSTIC_NODES: List[Tuple[str, np.ndarray]] = _build_acoustic_grid()

# ─────────────────────────────────────────────────────────────────────────────
# Observation dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RadarObservation:
    """RADAR polar measurement at the centroid sensor."""
    t               : float
    sensor_id       : str
    drone_id        : str            # simulator truth label (debugging only)
    range           : float          # m
    azimuth         : float          # deg CW from North
    radial_velocity : float          # m/s  (positive = receding)
    snr             : float          # dB
    sensor_pos      : Tuple[float, float, float] = field(default=(0., 0., 0.))
    modality        : str = field(default="RADAR", init=False)


@dataclass
class RF1Observation:
    """Passive SDR – 2-D direction of arrival (bearing only)."""
    t               : float
    sensor_id       : str
    drone_id        : str
    doa_angle       : float          # deg CW from North
    signal_strength : float          # dBm
    sensor_pos      : Tuple[float, float, float] = field(default=(0., 0., 0.))
    modality        : str = field(default="RF1_DOA", init=False)


@dataclass
class RF2Observation:
    """Remote-ID fingerprinting + telemetry decoding (may be spoofed/ghost)."""
    t               : float
    sensor_id       : str
    drone_id        : str            # ground-truth (may differ from claimed_id)
    claimed_id      : str            # broadcast RemoteID  (attacker-controlled)
    telemetry_pos   : Tuple[float, float, float]   # (x,y,z) m, noisy GPS
    telemetry_vel   : Tuple[float, float, float]   # (vx,vy,vz) m/s
    telemetry_acc   : Tuple[float, float, float]   # (ax,ay,az) m/s²
    snr             : float
    fing_similarity : float          # [0,1]  RF fingerprint match vs known fleet
    is_ghost        : bool           # True if this is an injected phantom packet
    is_spoofed      : bool           # True if claimed_id != real id
    sensor_pos      : Tuple[float, float, float] = field(default=(0., 0., 0.))
    modality        : str = field(default="RF2_FINGERPRINT", init=False)


@dataclass
class AcousticObservation:
    """Single acoustic node raw output – activation + SPL only."""
    t               : float
    sensor_id       : str
    activation      : bool           # SPL > ACOUSTIC_THRESHOLD
    intensity       : float          # dB SPL measured at this node
    sensor_pos      : Tuple[float, float, float]  # known node coordinates
    modality        : str = field(default="ACOUSTIC_NODE_RAW", init=False)


AnyObservation = (RadarObservation | RF1Observation
                  | RF2Observation  | AcousticObservation)


# ─────────────────────────────────────────────────────────────────────────────
# Drone kinematic model
# ─────────────────────────────────────────────────────────────────────────────

class DroneState:
    """
    Multirotor-like kinematics with smooth heading, bounded turn-rate and
    mission waypoints. The goal is not perfect flight dynamics, but realistic
    motion for visualisation: steady cruise, smooth turns, limited climb rate,
    and occasional hover-like slowdowns near waypoints.
    """
    PHYSICS_DT: float = 0.02   # 50 Hz

    def __init__(self, drone_id: str, rng: np.random.Generator) -> None:
        self.drone_id = drone_id
        self._rng = rng
        spawn_xy = rng.uniform(-HALF * 0.80, HALF * 0.80, size=2)
        self.pos = np.array([
            float(spawn_xy[0]),
            float(spawn_xy[1]),
            float(rng.uniform(50.0, min(400.0, AREA_Z_MAX))),
        ], dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.acc = np.zeros(3, dtype=float)
        self.heading = float(rng.uniform(0.0, 2.0 * math.pi))
        self._speed = float(rng.uniform(11.0, 17.0))
        self._waypoints: List[np.ndarray] = []
        self._target_wp = np.zeros(3, dtype=float)
        self._hover_until = 0.0
        self.t: float = 0.0
        self._refill_waypoints()
        self._advance_waypoint()

    def _new_waypoint(self) -> np.ndarray:
        return np.array([
            self._rng.uniform(-HALF * 0.85, HALF * 0.85),
            self._rng.uniform(-HALF * 0.85, HALF * 0.85),
            self._rng.uniform(50.0, min(400.0, AREA_Z_MAX)),
        ], dtype=float)

    def _refill_waypoints(self, n: int = 5) -> None:
        for _ in range(n):
            self._waypoints.append(self._new_waypoint())

    def _advance_waypoint(self) -> None:
        if not self._waypoints:
            self._refill_waypoints()
        self._target_wp = self._waypoints.pop(0)
        if len(self._waypoints) < 2:
            self._refill_waypoints()
        if self._rng.random() < 0.22:
            self._hover_until = self.t + float(self._rng.uniform(0.8, 2.0))
        else:
            self._hover_until = self.t

    def step(self, dt: float) -> None:
        MAX_SPEED = 28.0
        MIN_SPEED = 7.5
        MAX_H_ACC = 3.2
        MAX_TURN_RATE = math.radians(18.0)
        MAX_VS = 3.0
        WP_RADIUS = 65.0

        n = max(1, int(round(dt / self.PHYSICS_DT)))
        adt = dt / n

        # Distance between actual position and target waypoint

        for _ in range(n):
            err = self._target_wp - self.pos
            horiz = err[:2]
            horiz_dist = float(np.linalg.norm(horiz))

            # Moving with the waypoint based mechanism 

            if horiz_dist < WP_RADIUS and abs(err[2]) < 18.0:
                self._advance_waypoint()
                err = self._target_wp - self.pos
                horiz = err[:2]
                horiz_dist = float(np.linalg.norm(horiz))

            # Realistic heading and turning control mechanism 

            desired_heading = self.heading if horiz_dist < 1e-6 else math.atan2(horiz[0], horiz[1])
            heading_err = (desired_heading - self.heading + math.pi) % (2.0 * math.pi) - math.pi
            turn = float(np.clip(heading_err, -MAX_TURN_RATE * adt, MAX_TURN_RATE * adt))
            self.heading = (self.heading + turn) % (2.0 * math.pi)


            # Velocity control mechanism

            hovering = self.t < self._hover_until
            desired_speed = 0.0 if hovering else float(np.clip(0.020 * horiz_dist + 9.0, MIN_SPEED, MAX_SPEED))
            speed_err = desired_speed - self._speed
            speed_step = float(np.clip(speed_err, -MAX_H_ACC * adt, MAX_H_ACC * adt))
            self._speed = float(np.clip(self._speed + speed_step, 0.0, MAX_SPEED))
            

            # Horizontal and vertical velocity control (proportional to altitude error)

            vx = self._speed * math.sin(self.heading)
            vy = self._speed * math.cos(self.heading)
            target_vz = 0.0 if hovering else float(np.clip(err[2] * 0.12, -MAX_VS, MAX_VS))
            new_vel = np.array([vx, vy, target_vz], dtype=float)
            self.acc = (new_vel - self.vel) / max(adt, 1e-6)
            self.vel = new_vel
            self.pos += self.vel * adt

            # Boundary control

            for i in range(2):
                if abs(self.pos[i]) > HALF * 0.48:
                    self.pos[i] = math.copysign(HALF * 0.48, self.pos[i])
                    self.heading = (self.heading + math.pi * 0.7) % (2.0 * math.pi)
            self.pos[2] = float(np.clip(self.pos[2], AREA_Z_MIN, AREA_Z_MAX))
            if self.pos[2] in (AREA_Z_MIN, AREA_Z_MAX):
                self.vel[2] = 0.0
            self.t += adt

    def _delta(self, ref: np.ndarray) -> np.ndarray:
        return self.pos - ref

    def range_from(self, ref: np.ndarray) -> float:
        return float(np.linalg.norm(self._delta(ref)))

    def azimuth_from(self, ref: np.ndarray) -> float:
        d = self._delta(ref)
        return math.degrees(math.atan2(d[0], d[1])) % 360.0

    def radial_velocity_from(self, ref: np.ndarray) -> float:
        d = self._delta(ref)
        r = float(np.linalg.norm(d))
        if r < 1e-6:
            return 0.0
        return float(np.dot(self.vel, d / r))


_ORIGIN = np.zeros(3)   # sensors at origin


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality observation factories
# ─────────────────────────────────────────────────────────────────────────────

# Simple SNR model

def _snr_db(base: float, range_m: float, ref_range: float = 1000.0) -> float:
    """SNR decreases 20 dB per decade of range (one-way free space, simplified)."""
    return base - 20.0 * math.log10(max(range_m, 1.0) / ref_range)


def make_radar_obs(drone: DroneState, t: float,
                   sensor_id: str, rng: np.random.Generator
                   ) -> Optional[RadarObservation]:
    if rng.random() < RADAR_DROP_PROB:
        return None
    R  = drone.range_from(_ORIGIN)
    az = drone.azimuth_from(_ORIGIN)
    vr = drone.radial_velocity_from(_ORIGIN)
    snr = _snr_db(RADAR_SNR_BASE, R) + float(rng.normal(0, 1.2))
    return RadarObservation(
        t=t, sensor_id=sensor_id, drone_id=drone.drone_id,
        range=max(0.0, R + float(rng.normal(0, RADAR_RANGE_NOISE))),
        azimuth=(az + float(rng.normal(0, RADAR_AZ_NOISE))) % 360.0,
        radial_velocity=vr + float(rng.normal(0, RADAR_VEL_NOISE)),
        snr=snr,
    )


def make_rf1_obs(drone: DroneState, t: float,
                 sensor_id: str, rng: np.random.Generator
                 ) -> Optional[RF1Observation]:
    if rng.random() < RF1_DROP_PROB:
        return None
    R  = drone.range_from(_ORIGIN)
    az = drone.azimuth_from(_ORIGIN)

    # Free-space path loss (one-way, 2.4 GHz simplified)
    ss = -20.0 - 20.0 * math.log10(max(R, 1.0) / 100.0) + float(rng.normal(0, RF1_SS_NOISE))
    return RF1Observation(
        t=t, sensor_id=sensor_id, drone_id=drone.drone_id,
        doa_angle=(az + float(rng.normal(0, RF1_DOA_NOISE))) % 360.0,
        signal_strength=ss,
    )


def make_rf2_obs(drone: DroneState, t: float,
                 sensor_id: str, rng: np.random.Generator
                 ) -> Optional[RF2Observation]:
    if rng.random() < RF2_DROP_PROB:
        return None
    R   = drone.range_from(_ORIGIN)
    snr = _snr_db(RF2_SNR_BASE, R) + float(rng.normal(0, 1.5))

    spoofed = rng.random() < RF2_SPOOF_PROB
    if spoofed:
        claimed_id  = f"SPOOF_{rng.integers(1000, 9999)}"
        fing_sim    = float(rng.uniform(0.0, 0.35))
    else:
        claimed_id  = drone.drone_id
        fing_sim    = float(rng.uniform(0.72, 1.0))

    return RF2Observation(
        t=t, sensor_id=sensor_id, drone_id=drone.drone_id,
        claimed_id=claimed_id,
        telemetry_pos=tuple((drone.pos + rng.normal(0, RF2_POS_NOISE, 3)).tolist()),
        telemetry_vel=tuple((drone.vel + rng.normal(0, RF2_VEL_NOISE, 3)).tolist()),
        telemetry_acc=tuple((drone.acc + rng.normal(0, RF2_ACC_NOISE, 3)).tolist()),
        snr=snr, fing_similarity=fing_sim,
        is_ghost=False, is_spoofed=spoofed,
    )


def make_ghost_obs(t: float, sensor_id: str,
                   rng: np.random.Generator, ghost_idx: int
                   ) -> RF2Observation:
    """Craft a phantom packet injected by an attacker (no real drone behind it)."""
    fake_pos = (
        float(rng.uniform(-HALF * 0.7, HALF * 0.7)),
        float(rng.uniform(-HALF * 0.7, HALF * 0.7)),
        float(rng.uniform(AREA_Z_MIN, AREA_Z_MAX)),
    )
    fake_vel = tuple(float(rng.uniform(-10, 10)) for _ in range(3))
    fake_acc = tuple(float(rng.uniform(-2, 2))   for _ in range(3))
    return RF2Observation(
        t=t, sensor_id=sensor_id, drone_id=f"GHOST_{ghost_idx:02d}",
        claimed_id=f"GHOST_{rng.integers(1000, 9999)}",
        telemetry_pos=fake_pos, telemetry_vel=fake_vel, telemetry_acc=fake_acc,
        snr=float(rng.uniform(5, 12)),
        fing_similarity=float(rng.uniform(0.0, 0.30)),
        is_ghost=True, is_spoofed=True,
    )


def make_acoustic_obs(active: bool, intensity: float, t: float,
                      node_id: str, node_pos: np.ndarray) -> AcousticObservation:
    return AcousticObservation(
        t=t, sensor_id=node_id,
        activation=active,
        intensity=float(intensity),
        sensor_pos=tuple(node_pos.tolist()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared thread-safe queue
# ─────────────────────────────────────────────────────────────────────────────

class ObservationQueue:
    """
    FIFO pipeline handover point.

    Downstream modules call q.get() (blocking) or q.drain() (batch).
    The queue is typed but not ordered by timestamp – alignment is the
    responsibility of the preprocessing stage.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, obs: AnyObservation) -> None:
        self._q.put(obs)

    def get(self, timeout: Optional[float] = None) -> AnyObservation:
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> Optional[AnyObservation]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def drain(self, max_items: int = 512) -> List[AnyObservation]:
        items = []
        for _ in range(max_items):
            o = self.get_nowait()
            if o is None:
                break
            items.append(o)
        return items

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────────────────────

class TimeReference:
    def __init__(self, speed: float = 1.0) -> None:
        self._t0   = time.monotonic()
        self._spd  = speed

    def now(self) -> float:
        return (time.monotonic() - self._t0) * self._spd


class _PhysicsWorker(threading.Thread):
    DT = DroneState.PHYSICS_DT

    def __init__(self, drones: List[DroneState]) -> None:
        super().__init__(daemon=True, name="physics")
        self.drones    = drones
        self._stop     = threading.Event()
        self.lock      = threading.Lock()   # shared with sensor workers

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(timeout=self.DT):
            with self.lock:
                for d in self.drones:
                    d.step(self.DT)


class _SensorWorker(threading.Thread):
    """Generic sensor worker – one thread per sensor node."""

    def __init__(self, name: str, dt: float, fn,
                 drones: List[DroneState],
                 out_q: ObservationQueue,
                 rng: np.random.Generator,
                 time_ref: TimeReference,
                 physics_lock: threading.Lock,
                 **fn_kwargs) -> None:
        super().__init__(daemon=True, name=name)
        self._dt   = dt
        self._fn   = fn
        self._drones = drones
        self._q    = out_q
        self._rng  = rng
        self._tref = time_ref
        self._lock = physics_lock
        self._kw   = fn_kwargs
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(
            timeout=self._dt * (1.0 + self._rng.uniform(-0.08, 0.08))
        ):
            t = self._tref.now()
            with self._lock:
                snapshot = [(d.pos.copy(), d.vel.copy(),
                             d.acc.copy(), d.drone_id) for d in self._drones]
            for pos, vel, acc, did in snapshot:
                # Reconstruct a lightweight state proxy for the factory functions
                proxy = _DroneProxy(did, pos, vel, acc)
                obs = self._fn(proxy, t, self.name, self._rng, **self._kw)
                if obs is not None:
                    self._q.put(obs)


class _AcousticWorker(threading.Thread):
    """One thread for raw acoustic nodes + coarse array-level bearing event."""

    def __init__(self, nodes: List[Tuple[str, np.ndarray]],
                 drones: List[DroneState],
                 out_q: ObservationQueue,
                 rng: np.random.Generator,
                 time_ref: TimeReference,
                 physics_lock: threading.Lock) -> None:
        super().__init__(daemon=True, name="acoustic_array")
        self._nodes  = nodes
        self._drones = drones
        self._q      = out_q
        self._rng    = rng
        self._tref   = time_ref
        self._lock   = physics_lock
        self._stop   = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(timeout=ACOUSTIC_DT * (1.0 + self._rng.uniform(-0.05, 0.05))):
            t = self._tref.now()
            with self._lock:
                snapshot = [(d.pos.copy(), d.vel.copy(), d.acc.copy(), d.drone_id) for d in self._drones]

            active_nodes: List[Tuple[str, np.ndarray, float]] = []
            for node_id, node_pos in self._nodes:
                max_spl = -1e9
                for pos, vel, acc, did in snapshot:
                    proxy = _DroneProxy(did, pos, vel, acc)
                    horiz_R = float(np.linalg.norm(proxy.pos[:2] - node_pos[:2]))
                    # Coarse outdoor acoustic model: the array is mainly sensitive to
                    # horizontal proximity and provides only rough bearing support.
                    # Multiple neighboring nodes can activate.
                    spl = ACOUSTIC_SRC_LEVEL - 20.0 * math.log10(max(horiz_R, 1.0)) + float(self._rng.normal(0, ACOUSTIC_INT_NOISE))
                    if horiz_R > ACOUSTIC_ARRAY_MAX_RANGE:
                        spl -= 12.0 + 0.01 * (horiz_R - ACOUSTIC_ARRAY_MAX_RANGE)
                    max_spl = max(max_spl, spl)
                if self._rng.random() < ACOUSTIC_DROP_PROB:
                    continue
                active = max_spl > ACOUSTIC_THRESHOLD
                self._q.put(make_acoustic_obs(active, max_spl, t, node_id, node_pos))
                if active:
                    active_nodes.append((node_id, node_pos, max_spl))


class _RF2GhostWorker(threading.Thread):
    """Periodically injects ghost (phantom) packets to simulate an attacker."""

    def __init__(self, sensor_id: str,
                 out_q: ObservationQueue,
                 rng: np.random.Generator,
                 time_ref: TimeReference) -> None:
        super().__init__(daemon=True, name="rf2_ghost_injector")
        self._sid   = sensor_id
        self._q     = out_q
        self._rng   = rng
        self._tref  = time_ref
        self._stop  = threading.Event()
        self._idx   = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(timeout=RF2_DT):
            if self._rng.random() < RF2_GHOST_PROB:
                t   = self._tref.now()
                obs = make_ghost_obs(t, self._sid, self._rng, self._idx)
                self._q.put(obs)
                self._idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight drone proxy (avoids locking the full DroneState for each factory)
# ─────────────────────────────────────────────────────────────────────────────

class _DroneProxy:
    """Immutable snapshot – implements the same geometry interface as DroneState."""
    __slots__ = ("drone_id", "pos", "vel", "acc")

    def __init__(self, drone_id: str,
                 pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> None:
        self.drone_id = drone_id
        self.pos = pos
        self.vel = vel
        self.acc = acc

    def range_from(self, ref: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - ref))

    def azimuth_from(self, ref: np.ndarray) -> float:
        d = self.pos - ref
        return math.degrees(math.atan2(d[0], d[1])) % 360.0

    def elevation_from(self, ref: np.ndarray) -> float:
        d = self.pos - ref
        rho = math.sqrt(d[0]**2 + d[1]**2)
        return math.degrees(math.atan2(d[2], rho))

    def radial_velocity_from(self, ref: np.ndarray) -> float:
        d = self.pos - ref
        r = float(np.linalg.norm(d))
        if r < 1e-6:
            return 0.0
        return float(np.dot(self.vel, d / r))


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SensorDataGenerator:
    """
    Orchestrates drone physics + all sensor workers.

    Parameters
    ----------
    n_drones    : int   – number of real drones
    seed        : int   – master RNG seed
    sim_speed   : float – wall-time multiplier (>1 → faster than real-time)
    enable_ghosts : bool – enable adversarial ghost injection on RF2
    """

    def __init__(self,
                 n_drones     : int   = 2,
                 seed         : int   = 42,
                 sim_speed    : float = 1.0,
                 enable_ghosts: bool  = True) -> None:

        self._rng       = np.random.default_rng(seed)
        self._sim_speed = sim_speed
        self._ghosts    = enable_ghosts

        self.drones: List[DroneState] = [
            DroneState(f"DRONE_{i:02d}",
                       np.random.default_rng(int(self._rng.integers(0, 2**32))))
            for i in range(n_drones)
        ]

        self._workers : List[threading.Thread] = []
        self._physics : Optional[_PhysicsWorker] = None
        self._tref    : Optional[TimeReference]   = None

    # ── public API ────────────────────────────────────────────────────────

    def start(self, out_q: ObservationQueue) -> None:
        self._tref   = TimeReference(speed=self._sim_speed)
        self._physics = _PhysicsWorker(self.drones)
        self._physics.start()
        lock = self._physics.lock

        def _rng() -> np.random.Generator:
            return np.random.default_rng(int(self._rng.integers(0, 2**32)))

        # RADAR (one node at origin)
        self._workers.append(_SensorWorker(
            "RADAR_01", RADAR_DT, make_radar_obs,
            self.drones, out_q, _rng(), self._tref, lock))

        # RF1 DoA (one node at origin)
        self._workers.append(_SensorWorker(
            "RF1_01", RF1_DT, make_rf1_obs,
            self.drones, out_q, _rng(), self._tref, lock))

        # RF2 fingerprint (one node at origin)
        self._workers.append(_SensorWorker(
            "RF2_01", RF2_DT, make_rf2_obs,
            self.drones, out_q, _rng(), self._tref, lock))

        # Acoustic array (all 100 nodes, one thread)
        self._workers.append(_AcousticWorker(
            ACOUSTIC_NODES, self.drones, out_q, _rng(), self._tref, lock))

        # Adversarial ghost injector
        if self._ghosts:
            self._workers.append(_RF2GhostWorker(
                "RF2_01", out_q, _rng(), self._tref))

        for w in self._workers:
            w.start()

    def stop(self) -> None:
        all_threads = []
        if self._physics:
            self._physics.stop()
            all_threads.append(self._physics)
        for w in self._workers:
            if hasattr(w, "stop"):
                w.stop()
            all_threads.append(w)
        for th in all_threads:
            try:
                th.join(timeout=2.0)
            except Exception:
                pass
        self._workers.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.stop()

    @property
    def acoustic_node_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Return {sensor_id: (x,y,z)} for all 100 acoustic nodes."""
        return {sid: tuple(pos.tolist()) for sid, pos in ACOUSTIC_NODES}


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_dict(obs: AnyObservation) -> Dict[str, Any]:
    d = asdict(obs)
    for k, v in d.items():
        if isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, (tuple, list)):
            d[k] = [x if isinstance(x, str) else float(x) for x in v]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, signal

    print("=" * 72)
    print("  CUAS Data Generator v2.0 — smoke test  (Ctrl-C to stop)")
    print(f"  Area: {AREA_SIDE/1000:.0f}×{AREA_SIDE/1000:.0f} km | "
          f"Acoustic nodes: {len(ACOUSTIC_NODES)}")
    print("=" * 72)

    q   = ObservationQueue(maxsize=4096)
    gen = SensorDataGenerator(n_drones=3, seed=12, sim_speed=5.0, enable_ghosts=True)
    gen.start(q)
    counts: Dict[str, int] = {}

    def _shutdown(*_):
        gen.stop()
        print("\nCounts:", counts)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        obs = q.get(timeout=3.0)
        m = obs.modality
        counts[m] = counts.get(m, 0) + 1
        d = obs_to_dict(obs)
        if m == "RADAR":
            print(f"[RADAR]       t={d['t']:7.2f} | {d['drone_id']} "
                  f"R={d['range']/1000:.2f}km Az={d['azimuth']:.1f}° "
                  f"Vr={d['radial_velocity']:.1f}m/s SNR={d['snr']:.1f}dB")
        elif m == "RF2_FINGERPRINT":
            tag = " ⚠ GHOST" if d['is_ghost'] else (" ⚠ SPOOF" if d['is_spoofed'] else "")
            print(f"[RF2]         t={d['t']:7.2f} | claimed={d['claimed_id']:<18s} "
                  f"sim={d['fing_similarity']:.2f}{tag}")
        elif m == "ACOUSTIC_NODE_RAW" and d['activation']:
            print(f"[ACST {d['sensor_id']}]  t={d['t']:7.2f} | ACTIVE {d['intensity']:.1f}dBSPL")
