# C-UAS Synthetic Sensor Data Simulator & 3D Live Viewer

This section contains a Counter-Unmanned Aerial System (C-UAS) multi-sensor data fusion engine simulator and a 3D live viewer. The project generates highly realistic, multi-modal synthetic sensor data to simulate drone flights, adversarial attacks, and sensor array readings, which are then broadcasted and visualized in real-time.

## System Architecture Overview

The system is split into two primary components: a physics and sensor generation engine, and a web-based visualization server.

### 1. The Sensor Data Generator (data_generator.py)
This module is the core simulation engine. It orchestrates multi-threaded workers to simulate realistic drone physics and generate time-aligned sensor observations.

* **RADAR**: Positioned at the coordinate origin, providing polar measurements (range, azimuth, radial velocity) with calculated Signal-to-Noise Ratio (SNR).
* **RF1 (Passive SDR)**: Positioned at the origin, providing 2-D Direction of Arrival (DoA) bearing data and signal strength.
* **RF2 (Telemetry & Fingerprinting)**: Decodes Remote-ID telemetry and assesses RF fingerprint similarity against known fleets.
* **Acoustic Network**: A 10×10 regular grid of 100 nodes distributed over a 15 km × 15 km area at ground level, measuring Sound Pressure Level (SPL).

### 2. The Live Stream Server & UI (cuas_UI_server.py)
A lightweight Python HTTP server that acts as a pipeline between the generator and the user interface.

* **Server-Sent Events (SSE)**: Streams real-time JSON observation data to connected web clients asynchronously.
* **3D WebGL Viewer**: A single-page application built with Three.js that visualizes the airspace.
* **Visual Modalities**: Displays RADAR ranges as cyan dots, RF1 bearings as orange rays, RF2 telemetry as colored trails, and acoustic node activations via a live 2D heat-map.

---

## Highlighted Feature: Realistic Waypoint-Based Kinematics

One of the most powerful features of this simulator is the high accuracy of the simulated drone flight paths. Rather than relying on simple linear paths or randomized jumps, the engine uses a sophisticated waypoint-based kinematic model tailored for multirotor dynamics.

### How the Waypoint Mechanism Works
The `DroneState` class dictates physical movement through:

* **Continuous Waypoint Generation**: Drones are assigned random 3D target coordinates (waypoints) within the 15 km × 15 km bounding area.
* **Smooth Heading Correction**: The drone calculates the necessary heading to reach its target but is constrained by a maximum turn rate. This ensures the drone curves naturally towards its destination instead of snapping instantly to a new angle.
* **Dynamic Speed Adjustment**: As a drone approaches its target waypoint, it calculates its horizontal distance and adjusts its speed. The engine enforces maximum acceleration bounds, meaning the drone gradually speeds up during cruise and slows down as it reaches the waypoint.
* **Hovering and Vertical Dynamics**: To simulate realistic multirotor behavior, drones have a 22% chance to enter a temporary "hovering" state upon reaching a waypoint. Additionally, climb rates (vertical speed) are independently limited to simulate the mechanical constraints of drone ascents and descents.

**Why this ensures accuracy:** This physics-motivated approach prevents the generation of impossible flight telemetry. Trackers and fusion algorithms consuming this data are tested against realistic inertia, smooth trajectory curves, and plausible velocity changes, making the simulator an excellent testbed for real-world C-UAS software.

---

## Adversarial Threat Modeling

To test the robustness of downstream data fusion algorithms, the generator actively injects two types of adversarial electronic warfare attacks:

* **Identity Spoofing**: The RF2 sensor occasionally modifies the claimed Remote-ID string and drastically lowers the fingerprint similarity score, simulating an attacker attempting to impersonate a friendly drone.
* **Ghost Injection**: An independent thread injects "phantom" RF2 observations. These packets contain fake telemetry coordinates without any corresponding physical drone, designed to saturate trackers and force algorithms to cross-verify RF data against the un-spoofable acoustic grid.

---

## How to Run

1. Ensure you have Python 3 installed.
2. Run the server directly from your terminal. You can customize the simulation parameters using command-line arguments:

```bash
python cuas_UI_server.py --drones 4 --speed 5.0 --port 8765
