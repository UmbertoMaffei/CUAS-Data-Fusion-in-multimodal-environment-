#!/usr/bin/env python3
"""
CUAS Live Stream Server  
============================
Run:  python cuas_server.py
      python cuas_server.py --drones 4 --speed 5 --port 8765

Opens http://localhost:8765 automatically in the browser.
The 3D viewer receives observations via Server-Sent Events (SSE).

Pipeline integration (for other Python modules):
    from cuas_data_generator import SensorDataGenerator, ObservationQueue
    gen = SensorDataGenerator(n_drones=3, seed=42, sim_speed=1.0)
    q   = ObservationQueue()
    gen.start(q)
    while True:
        obs = q.get()   # blocking; use q.drain() for batches
        process(obs)
"""

from __future__ import annotations
import argparse, json, os, queue, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer
from typing import Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import (
    SensorDataGenerator, ObservationQueue, obs_to_dict,
    ACOUSTIC_NODES, AREA_SIDE, AREA_Z_MAX,
)

# ── Broadcast state ──────────────────────────────────────────────────────────
_clients: Set[queue.Queue] = set()
_lock = threading.Lock()
_obs_q: ObservationQueue | None = None
_gen:   SensorDataGenerator | None = None


def _broadcast_loop() -> None:
    while True:
        try:
            obs = _obs_q.get(timeout=0.5)
            msg = "data: " + json.dumps(obs_to_dict(obs), separators=(',', ':')) + "\n\n"
            with _lock:
                dead = []
                for q in _clients:
                    try:    q.put_nowait(msg)
                    except: dead.append(q)
                for q in dead:
                    _clients.discard(q)
        except Exception:
            pass


# ── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # ← CRITICAL: SSE requires HTTP/1.1

    def log_message(self, *_): pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._html()
        elif self.path == '/stream':
            self._sse()
        elif self.path == '/cod_uav.png':
            self._image('cod_uav.png')
        else:
            self._404()

    # ── serve the single-page app ────────────────────────────────────────────
    def _html(self):
        nodes_js = json.dumps([
            {"id": sid, "x": float(p[0]), "y": float(p[1])}
            for sid, p in ACOUSTIC_NODES
        ])
        body = (_PAGE
                .replace("%%AREA%%",      str(int(AREA_SIDE)))
                .replace("%%HALFKM%%",    str(AREA_SIDE / 2000))
                .replace("%%ZMAXKM%%",    str(AREA_Z_MAX / 1000))
                .replace("%%NODES%%",     nodes_js)
                ).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection",     "close")
        self.end_headers()
        self.wfile.write(body)

    def _image(self, filename):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        try:
            with open(filepath, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._404()

    # ── SSE stream ───────────────────────────────────────────────────────────
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type",                "text/event-stream")
        self.send_header("Cache-Control",               "no-cache")
        self.send_header("X-Accel-Buffering",           "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding",           "chunked")
        self.end_headers()

        cq: queue.Queue = queue.Queue(maxsize=4096)
        with _lock:
            _clients.add(cq)
        try:
            while True:
                try:
                    msg = cq.get(timeout=15)
                    self._send_chunk(msg.encode())
                except queue.Empty:
                    self._send_chunk(b": heartbeat\n\n")
        except Exception:
            pass
        finally:
            with _lock:
                _clients.discard(cq)

    def _send_chunk(self, data: bytes):
        """Write one HTTP/1.1 chunked-encoding chunk."""
        header = f"{len(data):X}\r\n".encode()
        self.wfile.write(header + data + b"\r\n")
        self.wfile.flush()

    def _404(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _Server(ThreadingMixIn, TCPServer):
    daemon_threads    = True
    allow_reuse_address = True


# ── HTML / JS page ───────────────────────────────────────────────────────────
# Everything is in one string so no separate files are needed.
# Server replaces %%TOKENS%% before sending.
_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>C-UAS Sensor Signals Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080c12;color:#c8d8f0;font-family:'Courier New',monospace;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#hdr{padding:7px 14px;background:#0b1622;border-bottom:1px solid #1a3050;
     display:flex;align-items:center;gap:18px;flex-shrink:0;z-index:10}
#hdr h1{font-size:12px;color:#4fc3f7;letter-spacing:2px;text-transform:uppercase}
.pill{padding:2px 7px;border-radius:3px;font-size:10px;font-weight:bold;
      background:#0a2810;color:#4caf50;border:1px solid #1e6030;transition:all .3s}
.pill.err{background:#2a0808;color:#f44336;border-color:#601010}
#body{display:flex;flex:1;min-height:0}
#cvwrap{flex:1;min-width:0;position:relative}
canvas#c3d{display:block;width:100%;height:100%}
#side{width:290px;background:#09111c;border-left:1px solid #182a40;
      display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.sec{padding:9px 10px;border-bottom:1px solid #132030}
.sec h3{font-size:9px;color:#4fc3f7;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase}
#hmc{display:block;width:100%;aspect-ratio:1;background:#080c12}
.cr{display:flex;align-items:center;padding:2px 0;font-size:10px;gap:4px}
.cr span:first-child{width:90px;flex-shrink:0;color:#7a9ab8}
.bw{flex:1;height:6px;background:#0d1a28;border-radius:3px;overflow:hidden}
.bv{height:100%;border-radius:3px;transition:width .4s}
.cn{width:32px;text-align:right;color:#c8d8f0}
#log{flex:1;overflow-y:auto;font-size:9.5px;line-height:1.65;padding:6px;
     font-family:'Courier New',monospace}
.lw{color:#ff5252}.lg{color:#ff1744}.lk{color:#69f0ae}.li{color:#546e7a}
#foot{font-size:9px;color:#3a5570;padding:4px 9px;border-top:1px solid #132030;
      display:flex;gap:12px;flex-shrink:0}
.leg{display:flex;align-items:center;gap:4px}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
</style>
</head>
<body>
<div id="hdr">
  <h1>⬡ C-UAS Multi-Sensor Signals — Live</h1>
  <span class="pill" id="st">● CONNECTING</span>
  <span id="ts" style="font-size:9px;color:#3a5570">t = 0.0 s</span>
  <span id="tot" style="font-size:9px;color:#3a5570;margin-left:auto">msgs: 0</span>
</div>
<div id="body">
  <div id="cvwrap"><canvas id="c3d"></canvas></div>
  <div id="side">
    <div class="sec">
      <h3>Acoustic raw nodes (SPL heatmap)</h3>
      <canvas id="hmc"></canvas>
    </div>
    <div class="sec">
      <h3>Observation counts</h3>
      <div class="cr"><span>RADAR</span>      <div class="bw"><div class="bv" id="bRADAR"       style="background:#4fc3f7;width:0"></div></div><span class="cn" id="nRADAR">0</span></div>
      <div class="cr"><span>RF1 DoA</span>    <div class="bw"><div class="bv" id="bRF1_DOA"     style="background:#ff9800;width:0"></div></div><span class="cn" id="nRF1_DOA">0</span></div>
      <div class="cr"><span>RF2 FP</span>     <div class="bw"><div class="bv" id="bRF2_FP"      style="background:#69f0ae;width:0"></div></div><span class="cn" id="nRF2_FP">0</span></div>
      <div class="cr"><span>Acoustic raw</span>   <div class="bw"><div class="bv" id="bACOUSTIC"    style="background:#b39ddb;width:0"></div></div><span class="cn" id="nACOUSTIC">0</span></div>
      <div class="cr"><span>⚠ Ghost</span>   <div class="bw"><div class="bv" id="bGHOST"       style="background:#ff1744;width:0"></div></div><span class="cn" id="nGHOST">0</span></div>
      <div class="cr"><span>⚠ Spoof</span>   <div class="bw"><div class="bv" id="bSPOOF"       style="background:#ff6d00;width:0"></div></div><span class="cn" id="nSPOOF">0</span></div>
    </div>
    <div class="sec">
      <h3>How to read</h3>
      <div style="font-size:10px;line-height:1.45;color:#9fb6d3">
        Cyan dot = RADAR range/azimuth point<br/>
        Orange dot = RF1 bearing only<br/>
        Colored track + label = RF2 telemetry track
      </div>
    </div>
    <div class="sec">
      <h3>Latest sensor signals</h3>
      <div id="sigRadar" style="font-size:10px;line-height:1.5;color:#b7cbe3">RADAR: -</div>
      <div id="sigRf1" style="font-size:10px;line-height:1.5;color:#b7cbe3;margin-top:4px">RF1-DOA: -</div>
      <div id="sigRf2" style="font-size:10px;line-height:1.5;color:#b7cbe3;margin-top:4px">RF2: -</div>
    </div>
    <div class="sec" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
      <h3>Event log</h3>
      <div id="log"></div>
    </div>
    <div id="foot">
      <div class="leg"><div class="dot" style="background:#ffe066"></div>RF2 trail</div>
      <div class="leg"><div class="dot" style="background:#4fc3f7"></div>RADAR range-az</div>
      <div class="leg"><div class="dot" style="background:#ff9800"></div>RF1 bearing</div>
      <div class="leg"><div class="dot" style="background:#69f0ae"></div>RF2 telemetry</div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
"use strict";
// ═══════════════════════════════════════════════════════════════════
// CONSTANTS  (injected by server)
// ═══════════════════════════════════════════════════════════════════
const AREA_M   = %%AREA%%;          // metres side
const HALF_KM  = %%HALFKM%%;       // km half-side
const ZMAX_KM  = %%ZMAXKM%%;       // km max altitude
const NODES    = %%NODES%%;        // [{id,x,y}] in metres

// ─── coordinate mapping ──────────────────────────────────────────
// Python ENU:   X=East(m)  Y=North(m)  Z=Alt(m)
// Three.js:     X=East(km) Y=Alt(km)   Z=-North(km)
const M2K = 1/1000;
function enu(x,y,z){ return new THREE.Vector3(x*M2K, z*M2K, -y*M2K); }

// spherical az(deg,CW from N), el(deg), range(m)  →  THREE.Vector3
function sph2v(az,el,r){
  const a=az*Math.PI/180, e=el*Math.PI/180, R=r*M2K;
  return enu(R*Math.sin(a)*Math.cos(e), R*Math.cos(a)*Math.cos(e), R*Math.sin(e));
}

// ═══════════════════════════════════════════════════════════════════
// THREE.JS SCENE
// ═══════════════════════════════════════════════════════════════════
const canvas = document.getElementById('c3d');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setClearColor(0x080c12,1);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50,1,0.01,500);

// ── orbit control ────────────────────────────────────────────────
let camT=0.5, camP=0.4, camR=20;
function camPos(){
  camera.position.set(
    camR*Math.sin(camT)*Math.cos(camP),
    camR*Math.sin(camP),
    camR*Math.cos(camT)*Math.cos(camP));
  camera.lookAt(0,ZMAX_KM*0.3,0);
}
camPos();

let drag=false,px=0,py=0;
canvas.addEventListener('mousedown', e=>{drag=true;px=e.clientX;py=e.clientY});
window.addEventListener('mouseup',   ()=>drag=false);
window.addEventListener('mousemove', e=>{
  if(!drag)return;
  camT -= (e.clientX-px)*0.006; camP = Math.max(-1.3,Math.min(1.3,camP-(e.clientY-py)*0.006));
  px=e.clientX; py=e.clientY; camPos();
});
canvas.addEventListener('wheel', e=>{
  camR=Math.max(2,Math.min(80,camR+e.deltaY*0.025)); camPos(); e.preventDefault();
},{passive:false});

// ── lights ───────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x334466,2));
const sun=new THREE.DirectionalLight(0x8899ff,1.2);
sun.position.set(5,10,3); scene.add(sun);

// ── ground + boundary ────────────────────────────────────────────
{
  const g=new THREE.PlaneGeometry(HALF_KM*2,HALF_KM*2);
  const m=new THREE.MeshBasicMaterial({color:0x07111e,transparent:true,opacity:0.7,side:THREE.DoubleSide});
  const pl=new THREE.Mesh(g,m); pl.rotation.x=-Math.PI/2; scene.add(pl);

  const pts=[
    new THREE.Vector3(-HALF_KM,0,-HALF_KM), new THREE.Vector3( HALF_KM,0,-HALF_KM),
    new THREE.Vector3( HALF_KM,0, HALF_KM), new THREE.Vector3(-HALF_KM,0, HALF_KM),
    new THREE.Vector3(-HALF_KM,0,-HALF_KM)
  ];
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({color:0x1a3a5a})));
}

// corner altitude posts
for(const [cx,cz] of [[-1,-1],[1,-1],[1,1],[-1,1]]){
  scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(cx*HALF_KM,0,cz*HALF_KM),
      new THREE.Vector3(cx*HALF_KM,ZMAX_KM,cz*HALF_KM)
    ]),
    new THREE.LineBasicMaterial({color:0x0d2030,opacity:0.5,transparent:true})));
}

// simple canvas-based label sprite
function makeLabelSprite(text){
  const c=document.createElement('canvas'); c.width=256; c.height=64;
  const ctx=c.getContext('2d');
  ctx.fillStyle='rgba(8,12,18,0.75)'; ctx.fillRect(0,6,256,44);
  ctx.strokeStyle='rgba(79,195,247,0.55)'; ctx.strokeRect(0.5,6.5,255,43);
  ctx.font='24px Courier New'; ctx.fillStyle='#d7ecff'; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(text,128,28);
  const tex=new THREE.CanvasTexture(c);
  const mat=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false});
  const spr=new THREE.Sprite(mat); spr.scale.set(1.8,0.45,1);
  return spr;
}

// sensor origin star
const originStar=new THREE.Mesh(new THREE.SphereGeometry(0.08,8,8),
  new THREE.MeshBasicMaterial({color:0xffffff}));
scene.add(originStar);
const originLbl=makeLabelSprite('CENTROID');
originLbl.position.set(0,0.35,0); scene.add(originLbl);

// ── acoustic nodes ────────────────────────────────────────────────
const nodeMesh={};
NODES.forEach(n=>{
  const dot=new THREE.Mesh(new THREE.SphereGeometry(0.04,8,8),
    new THREE.MeshBasicMaterial({color:0x28445f}));
  dot.position.copy(enu(n.x,n.y,0));
  scene.add(dot);
  const halo=new THREE.Mesh(new THREE.CircleGeometry(0.22,24),
    new THREE.MeshBasicMaterial({color:0xb39ddb,transparent:true,opacity:0.0,side:THREE.DoubleSide}));
  halo.rotation.x=-Math.PI/2;
  halo.position.copy(enu(n.x,n.y,0)); halo.position.y=0.01;
  scene.add(halo);
  nodeMesh[n.id]={dot,halo};
});

// ═══════════════════════════════════════════════════════════════════
// DRONE TRAILS  – one per drone_id, created on first observation
// ═══════════════════════════════════════════════════════════════════
const TCOLS=[0xffe066,0xff6b9d,0xc77dff,0x4cc9f0,0xf8961e,0x90be6d,0x43aa8b];
const TLEN=150;
const drones={};
const droneTex = new THREE.TextureLoader().load('/cod_uav.png');

function getDrone(id){
  if(drones[id]) return drones[id];
  const col=TCOLS[Object.keys(drones).length % TCOLS.length];
  const buf=new Float32Array(TLEN*3);
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(buf,3));
  geo.setDrawRange(0,0);
  const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color:col,linewidth:1}));
  scene.add(line);
  const sph=new THREE.Sprite(new THREE.SpriteMaterial({map: droneTex, color: 0xffffff, transparent: false, depthTest: false}));
  sph.scale.set(0.7, 0.7, 1);
  sph.position.y=-9999;
  scene.add(sph);
  const lbl=makeLabelSprite(id); lbl.position.set(0,-9999,0); scene.add(lbl);
  const d={pts:[],line,sph,lbl,col};
  drones[id]=d; return d;
}

function pushDrone(id, v3){
  const d=getDrone(id);
  d.pts.push(v3.clone());
  if(d.pts.length>TLEN) d.pts.shift();
  const buf=d.line.geometry.attributes.position.array;
  d.pts.forEach((p,i)=>{buf[i*3]=p.x;buf[i*3+1]=p.y;buf[i*3+2]=p.z;});
  d.line.geometry.attributes.position.needsUpdate=true;
  d.line.geometry.setDrawRange(0,d.pts.length);
  const last=d.pts[d.pts.length-1];
  d.sph.position.copy(last);
  d.lbl.position.copy(last.clone().add(new THREE.Vector3(0,0.32,0)));
}

// ═══════════════════════════════════════════════════════════════════
// RADAR DETECTIONS  – ground-plane range/azimuth spoke + marker
// ═══════════════════════════════════════════════════════════════════
const RPOOL=220, rpool=[];
for(let i=0;i<RPOOL;i++){
  const dot=new THREE.Mesh(new THREE.SphereGeometry(0.05,6,6), new THREE.MeshBasicMaterial({color:0x4fc3f7,transparent:true,opacity:0}));
  scene.add(dot); rpool.push({dot,st:-1e9});
}
let rhead=0;
function groundPolar(az,r){
  const a=az*Math.PI/180, R=r*M2K;
  return new THREE.Vector3(R*Math.sin(a), 0.03, -R*Math.cos(a));
}
function addRadar(az,r,st){
  const s=rpool[rhead%RPOOL];
  const p=groundPolar(az,r);
  s.dot.position.copy(p); s.dot.material.opacity=1; s.st=st; rhead++;
}

// ═══════════════════════════════════════════════════════════════════
// RF1 DoA RAYS  – one persistent ground-plane bearing ray per drone_id
// ═══════════════════════════════════════════════════════════════════
const rf1={};
function setRF1Ray(droneId, az, st){
  if(!rf1[droneId]){
    const dot=new THREE.Mesh(new THREE.SphereGeometry(0.08,6,6),
      new THREE.MeshBasicMaterial({color:0xff9800,transparent:true,opacity:0}));
    scene.add(dot);
    rf1[droneId]={dot,st:0};
  }
  const e=rf1[droneId];
  let r = AREA_M * 0.55; // fallback al bordo
  // Usa la distanza stimata del drone se abbiamo la sua traccia telemetrica
  if (drones[droneId] && drones[droneId].pts.length > 0) {
    const last = drones[droneId].pts[drones[droneId].pts.length - 1];
    r = Math.sqrt(last.x*last.x + last.z*last.z) * 1000;
  }
  const tip=groundPolar(az, r);
  e.dot.position.copy(tip);
  e.dot.material.opacity=1; e.st=st;
}

// ═══════════════════════════════════════════════════════════════════
// RF2 TELEMETRY  – cone pool, coloured by trust
// ═══════════════════════════════════════════════════════════════════
const F2POOL=400, f2pool=[];
for(let i=0;i<F2POOL;i++){
  const m=new THREE.Mesh(new THREE.ConeGeometry(0.09,0.22,6),
    new THREE.MeshBasicMaterial({color:0x00e676,transparent:true,opacity:0}));
  scene.add(m); f2pool.push({m,st:-1e9});
}
let f2head=0;
function addRF2(arr, ghost, spoof, st){
  const s=f2pool[f2head%F2POOL];
  s.m.position.copy(enu(arr[0],arr[1],arr[2]));
  s.m.material.color.setHex(ghost?0xff1744:spoof?0xff6d00:0x00e676);
  s.m.material.opacity=1; s.st=st; f2head++;
}

// ═══════════════════════════════════════════════════════════════════
// ACOUSTIC HEATMAP  (2-D canvas)
// ═══════════════════════════════════════════════════════════════════
const hmc=document.getElementById('hmc');
const hmx=hmc.getContext('2d');
hmc.width=hmc.height=180;
const spl={};  NODES.forEach(n=>spl[n.id]=20);
const act={};

function drawHM(){
  const W=hmc.width,H=hmc.height,G=45;
  const img=hmx.createImageData(G,G);
  const nl=NODES.map(n=>({nx:(n.x+AREA_M/2)/AREA_M, ny:1-(n.y+AREA_M/2)/AREA_M, v:spl[n.id]||20}));
  for(let gy=0;gy<G;gy++) for(let gx=0;gx<G;gx++){
    const px=gx/(G-1),py=gy/(G-1);
    let ws=0,vs=0;
    nl.forEach(n=>{const d2=Math.max(1e-7,(px-n.nx)**2+(py-n.ny)**2);const w=1/d2;ws+=w;vs+=w*n.v;});
    const t=Math.max(0,Math.min(1,(vs/ws-20)/55));
    const i=(gy*G+gx)*4;
    img.data[i]  =t<.5?Math.round(t*2*120):Math.round(120+(t-.5)*2*135);
    img.data[i+1]=t<.5?Math.round(t*2*20) :Math.round(20+(t-.5)*2*160);
    img.data[i+2]=t<.5?Math.round(160-t*2*150):10;
    img.data[i+3]=230;
  }
  const tmp=document.createElement('canvas');tmp.width=G;tmp.height=G;
  tmp.getContext('2d').putImageData(img,0,0);
  hmx.clearRect(0,0,W,H);
  hmx.imageSmoothingEnabled=true;
  hmx.drawImage(tmp,0,0,W,H);
  NODES.forEach(n=>{
    const nx=((n.x+AREA_M/2)/AREA_M)*W, ny=(1-(n.y+AREA_M/2)/AREA_M)*H;
    const on=act[n.id]||false;
    hmx.beginPath();hmx.arc(nx,ny,on?4.5:2,0,Math.PI*2);
    hmx.fillStyle=on?'#00ff88':'#1c3a50';hmx.fill();
    if(on){hmx.beginPath();hmx.arc(nx,ny,7,0,Math.PI*2);
      hmx.strokeStyle='rgba(0,255,136,.3)';hmx.lineWidth=2;hmx.stroke();}
  });
  // origin
  hmx.strokeStyle='#ffffff50';hmx.lineWidth=1;
  hmx.beginPath();hmx.moveTo(W/2-5,H/2);hmx.lineTo(W/2+5,H/2);
  hmx.moveTo(W/2,H/2-5);hmx.lineTo(W/2,H/2+5);hmx.stroke();
}

// ═══════════════════════════════════════════════════════════════════
// COUNTERS + LOG
// ═══════════════════════════════════════════════════════════════════
const cnt={RADAR:0,RF1_DOA:0,RF2_FP:0,ACOUSTIC:0,GHOST:0,SPOOF:0};
const logEl=document.getElementById('log');
let total=0,simT=0;

function addLog(txt,cls){
  const el=document.createElement('div');el.className=cls;el.textContent=txt;
  logEl.prepend(el);while(logEl.children.length>80)logEl.removeChild(logEl.lastChild);
}
function refreshBars(){
  const mx=Math.max(1,...Object.values(cnt));
  [['RADAR','bRADAR','nRADAR'],['RF1_DOA','bRF1_DOA','nRF1_DOA'],
   ['RF2_FP','bRF2_FP','nRF2_FP'],['ACOUSTIC','bACOUSTIC','nACOUSTIC'],
   ['GHOST','bGHOST','nGHOST'],['SPOOF','bSPOOF','nSPOOF']].forEach(([k,b,n])=>{
    document.getElementById(b).style.width=(cnt[k]/mx*100)+'%';
    document.getElementById(n).textContent=cnt[k];
  });
}

// ═══════════════════════════════════════════════════════════════════
// SSE INGESTION
// ═══════════════════════════════════════════════════════════════════
function connectSSE(){
  const es=new EventSource('/stream');
  es.onopen=()=>{
    const s=document.getElementById('st');
    s.textContent='● LIVE'; s.className='pill';
  };
  es.onerror=()=>{
    const s=document.getElementById('st');
    s.textContent='● RECONNECTING'; s.className='pill err';
    es.close(); setTimeout(connectSSE,2000);
  };
  es.onmessage=ev=>{
    let d; try{d=JSON.parse(ev.data);}catch{return;}
    total++; simT=Math.max(simT,d.t||0);
    document.getElementById('ts').textContent='t = '+simT.toFixed(1)+' s';
    document.getElementById('tot').textContent='msgs: '+total;

    const mod=d.modality;

    if(mod==='RADAR'){
      cnt.RADAR++;
      addRadar(d.azimuth, d.range, d.t);
      document.getElementById('sigRadar').textContent = `RADAR: R=${(d.range/1000).toFixed(2)} km | Az=${d.azimuth.toFixed(1)}° | Vr=${d.radial_velocity.toFixed(1)} m/s | SNR=${d.snr.toFixed(1)} dB`;
    }
    else if(mod==='RF1_DOA'){
      cnt.RF1_DOA++;
      setRF1Ray(d.drone_id, d.doa_angle, d.t);
      document.getElementById('sigRf1').textContent = `RF1-DOA: bearing=${d.doa_angle.toFixed(1)}° | RSSI=${d.signal_strength.toFixed(1)} dBm`;
    }
    else if(mod==='RF2_FINGERPRINT'){
      cnt.RF2_FP++;
      if(d.is_ghost){
        cnt.GHOST++;
        addLog(`t=${d.t.toFixed(1)} ⚠ GHOST @ (${(d.telemetry_pos[0]/1000).toFixed(1)},${(d.telemetry_pos[1]/1000).toFixed(1)}) km`,'lg');
      } else if(d.is_spoofed){
        cnt.SPOOF++;
        addLog(`t=${d.t.toFixed(1)} ⚠ SPOOF ${d.claimed_id} sim=${d.fing_similarity.toFixed(2)}`,'lw');
      }
      addRF2(d.telemetry_pos, d.is_ghost, d.is_spoofed, d.t);
      if(!d.is_ghost) pushDrone(d.drone_id, enu(d.telemetry_pos[0], d.telemetry_pos[1], d.telemetry_pos[2]));
      document.getElementById('sigRf2').textContent = `RF2: claimed=${d.claimed_id} | sim=${d.fing_similarity.toFixed(2)} | SNR=${d.snr.toFixed(1)} dB${d.is_ghost?' | GHOST':(d.is_spoofed?' | SPOOF':'')}`;
    }
    else if(mod==='ACOUSTIC_NODE_RAW'){
      cnt.ACOUSTIC++;
      spl[d.sensor_id]=d.intensity;
      act[d.sensor_id]=d.activation;
      if(nodeMesh[d.sensor_id]){
        nodeMesh[d.sensor_id].dot.material.color.setHex(d.activation?0x00ff88:0x28445f);
        nodeMesh[d.sensor_id].halo.material.opacity=d.activation?Math.max(0.08, Math.min(0.42,(d.intensity-38)/16)):0.0;
        nodeMesh[d.sensor_id].halo.scale.setScalar(d.activation?Math.max(0.8, Math.min(1.8,(d.intensity-34)/12)):1.0);
    }
      if(d.activation && Math.random()<0.012)
        addLog(`t=${d.t.toFixed(1)} ◉ ${d.sensor_id} ${d.intensity.toFixed(1)} dBSPL`,'lk');
    }
    refreshBars();
  };
}
connectSSE();

// ═══════════════════════════════════════════════════════════════════
// RENDER LOOP
// ═══════════════════════════════════════════════════════════════════
const R_FADE=5, F2_FADE=7, RF1_FADE=2;
let lastHM=0;

(function animate(){
  requestAnimationFrame(animate);

  // fade RADAR
  rpool.forEach(s=>{ const op=Math.max(0,1-(simT-s.st)/R_FADE); s.dot.material.opacity=op; });
  // fade RF2
  f2pool.forEach(s=>{ s.m.material.opacity=Math.max(0,1-(simT-s.st)/F2_FADE); });
  // fade RF1 rays
  Object.values(rf1).forEach(e=>{ e.dot.material.opacity=Math.max(0,1-(simT-e.st)/RF1_FADE); });

  // heatmap ~5fps
  const now=performance.now()/1000;
  if(now-lastHM>0.19){drawHM();lastHM=now;}

  // responsive
  const W=canvas.clientWidth,H=canvas.clientHeight;
  if(canvas.width!==W||canvas.height!==H){
    renderer.setSize(W,H,false);
    camera.aspect=W/H; camera.updateProjectionMatrix();
  }
  renderer.render(scene,camera);
})();
</script>
</body>
</html>"""


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones",    type=int,   default=3)
    ap.add_argument("--speed",     type=float, default=3.0)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--port",      type=int,   default=8765)
    ap.add_argument("--no-ghosts", action="store_true")
    args = ap.parse_args()

    # Try requested port, fall back to any free port
    port = args.port
    try:
        s = socket.socket(); s.bind(("", port)); s.close()
    except OSError:
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()

    global _obs_q, _gen
    _obs_q = ObservationQueue(maxsize=32768)
    _gen   = SensorDataGenerator(
        n_drones=args.drones, seed=args.seed,
        sim_speed=args.speed, enable_ghosts=not args.no_ghosts,
    )
    _gen.start(_obs_q)

    threading.Thread(target=_broadcast_loop, daemon=True, name="broadcast").start()

    server = _Server(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()

    url = f"http://localhost:{port}"
    print("=" * 58)
    print("  C-UAS Live Stream Server v3")
    print(f"  → Open in browser:  {url}")
    print(f"  Drones={args.drones}  Speed={args.speed}×  "
          f"Ghosts={'OFF' if args.no_ghosts else 'ON'}")
    print("  Ctrl-C to stop")
    print("=" * 58)

    try:
        import webbrowser
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        _gen.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
