//latest 3

(function () {
  const container = document.getElementById('gradflow');
  if (!container) return;

  const clamp = (v,min,max)=>Math.max(min,Math.min(max,v));
  const parseRGB = (s, fb)=> {
    const a = String(s||'').split(',').map(n=>+n.trim());
    return a.length===3 && a.every(Number.isFinite) ? a : fb;
  };
  const typeMap = { linear:0, conic:1, animated:2, wave:3, silk:4, smoke:5, stripe:6 };

  // Read config from data-* on #gradflow
  const cfg = {
    color1: parseRGB(container.dataset.color1, [226,98,75]),
    color2: parseRGB(container.dataset.color2, [255,255,255]),
    color3: parseRGB(container.dataset.color3, [30,34,159]),
    speed: +container.dataset.speed || 0.4,
    scale: +container.dataset.scale || 1,
    type: typeMap[(container.dataset.type || 'stripe').toLowerCase()] ?? 6,
    noise: clamp(+container.dataset.noise || 0.08, 0, 1),
    swirlStrength: clamp(+container.dataset.swirlStrength || 1.25, 0.0, 20.0),
    swirlFalloff:  clamp(+container.dataset.swirlFalloff  || 6.0,  0.1, 50.0),
    dither:        clamp(+container.dataset.dither        || 1.0,  0.0, 2.0),   // 1.0 ≈ 1 LSB
    midpoint:      clamp(+container.dataset.midpoint      || 0.5,  0.0, 1.0),   // where C2 sits
    centerPower:   Math.max(+container.dataset.centerPower || 1.0, 1.0)         // >1 shrinks C2 area
  };
  const norm = (rgb)=>[rgb[0]/255, rgb[1]/255, rgb[2]/255];

  // ---------- Shaders ----------
  const vertexShader = `
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
  `;

  const fragmentShader = `
precision mediump float;

uniform float u_time;
uniform vec3  u_color1;
uniform vec3  u_color2;
uniform vec3  u_color3;
uniform float u_speed;
uniform float u_scale;
uniform int   u_type;
uniform float u_noise;
uniform vec2  u_resolution;

uniform vec2  u_mouse;          // cursor (0..1)
uniform float u_swirlStrength;  // from data-swirl-strength
uniform float u_swirlFalloff;   // from data-swirl-falloff
uniform float u_dither;         // from data-dither (0..~2)
uniform float u_midpoint;       // from data-midpoint (0..1)
uniform float u_centerPower;    // from data-center-power (>=1)

varying vec2 vUv;

#define PI 3.14159265359
#define S(a,b,t) smoothstep(a,b,t)

float saturate(float x){ return clamp(x,0.0,1.0); }
vec3  saturate(vec3 v){ return clamp(v,0.0,1.0); }

// ---- sRGB helpers ----
vec3 sRGBToLinear(vec3 c){
  bvec3 cutoff = lessThanEqual(c, vec3(0.04045));
  vec3 lower = c / 12.92;
  vec3 higher = pow((c + 0.055) / 1.055, vec3(2.4));
  return mix(higher, lower, vec3(cutoff));
}
vec3 linearToSRGB(vec3 c){
  bvec3 cutoff = lessThanEqual(c, vec3(0.0031308));
  vec3 lower = 12.92 * c;
  vec3 higher = 1.055 * pow(c, vec3(1.0/2.4)) - 0.055;
  return mix(higher, lower, vec3(cutoff));
}

// ---- Interleaved gradient noise (dither) ----
float ign(vec2 p){
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// ---- Film-style grain (optional) ----
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float valueNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * valueNoise(p); p *= 2.0; a *= 0.5; }
  return s;
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float filmGrain(vec2 px, float time, float strength) {
  float frame = floor(time * 24.0);
  float ang = frame * 2.39996323;
  mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
  vec2 rp = R * (px * 0.75);
  float g = fbm(rp + frame);
  return (g - 0.5) * 2.0 * strength;
}

// ---- Misc helpers ----
float noise(vec2 st) { return fract(sin(dot(st, vec2(12.9898,78.233))) * 43758.5453); }
mat2 Rot(float a){ float s=sin(a), c=cos(a); return mat2(c,-s,s,c); }
vec2 hash(vec2 p){ p=vec2(dot(p,vec2(2127.1,81.17)),dot(p,vec2(1269.5,283.37))); return fract(sin(p)*43758.5453); }
float advancedNoise(in vec2 p){
  vec2 i=floor(p), f=fract(p);
  vec2 u=f*f*(3.0-2.0*f);
  float n = mix(mix(dot(-1.0+2.0*hash(i+vec2(0.0,0.0)), f-vec2(0.0,0.0)),
                    dot(-1.0+2.0*hash(i+vec2(1.0,0.0)), f-vec2(1.0,0.0)), u.x),
                mix(dot(-1.0+2.0*hash(i+vec2(0.0,1.0)), f-vec2(0.0,1.0)),
                    dot(-1.0+2.0*hash(i+vec2(1.0,1.0)), f-vec2(1.0,1.0)), u.x), u.y);
  return 0.5+0.5*n;
}

// Globals: linear-space colors
vec3 C1, C2, C3;

// ----- Centralized tri-color blender -----
// t: 0..1 pattern; mid: where C2 is centered; power>=1 reduces C2 area
vec3 triBlend(vec3 c1, vec3 c2, vec3 c3, float t, float mid, float power){
  t = saturate(t);
  // map [0,mid] -> [0,0.5], [mid,1] -> [0.5,1] so mid sits at 0.5
  float x = (t < mid) ? (0.5 * t / max(mid, 1e-4)) : (0.5 + 0.5 * (t - mid) / max(1.0 - mid, 1e-4));

  // bell-shaped center weight around 0.5
  float center = 1.0 - abs(x*2.0 - 1.0);
  center = pow(center, power);      // power > 1 narrows C2 band

  float edge = 1.0 - center;
  float w1 = edge * (1.0 - x);
  float w3 = edge * x;
  float w2 = center;

  float sum = w1 + w2 + w3 + 1e-6;
  w1/=sum; w2/=sum; w3/=sum;

  return c1*w1 + c2*w2 + c3*w3;
}

// ----- Gradients now return a 0..1 pattern and use triBlend -----
vec3 linearGradient(vec2 uv, float time){
  float t = (uv.y*u_scale) + sin(uv.x*PI + time)*0.1;
  t = saturate(t);
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 conicGradient(vec2 uv, float time){
  vec2 center=vec2(0.5), pos=uv-center;
  float angle = atan(pos.y, pos.x);
  float t = fract((angle + PI) / (2.0*PI) * u_scale + time*0.06);
  // little radial ripple for interest (kept small)
  float dist = length(pos);
  t = saturate(t + sin(dist*8.0 + time*1.5)*0.01);
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 animatedGradient(vec2 uv, float time){
  float ratio=u_resolution.x/u_resolution.y;
  vec2 p=uv - 0.5;
  float degree = advancedNoise(vec2(time*0.1*u_speed, p.x*p.y));
  p.y*=1.0/ratio; p*=Rot(radians((degree-0.5)*720.0*u_scale+180.0)); p.y*=ratio;
  float freq=5.0*u_scale, sp=time*2.0*u_speed;
  p.x+=sin(p.y*freq+sp)/30.0; p.y+=sin(p.x*freq*1.5+sp)/45.0;
  float t = S(-0.3, 0.2, (p*Rot(radians(-5.0))).x); // same “axis” as before
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 waveGradient(vec2 uv,float time){
  float y=uv.y;
  float w1=sin(uv.x*PI*u_scale*0.8+time*u_speed*0.5)*0.1;
  float w2=sin(uv.x*PI*u_scale*0.5+time*u_speed*0.3)*0.15;
  float w3=sin(uv.x*PI*u_scale*1.2+time*u_speed*0.8)*0.2;
  float t = saturate(y + w1 + w2 + w3);
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 silkGradient(vec2 uv,float time){
  vec2 fc=uv*u_resolution, inv=1.0/u_resolution.xy;
  vec2 cu=(fc*2.0-u_resolution.xy)*inv * u_scale;
  float d=-time*u_speed*0.5, a=0.0, damp=1.0/(1.0+u_scale*0.1);
  for(float i=0.0;i<8.0;i++){ a+=cos(i-d-a*cu.x)*damp; d+=sin(cu.y*i+a)*damp; }
  d+=time*u_speed*0.5;
  // turn the procedural pattern into a 0..1 “t”
  float t = saturate(0.5 + 0.5 * (cos(cu.x*d+a)*0.5 + cos(cu.y*a+d)*0.5));
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 smokeGradient(vec2 uv,float time){
  float mr=min(u_resolution.x,u_resolution.y);
  vec2 fc=uv*u_resolution;
  vec2 p=(2.0*fc - u_resolution)/mr * u_scale;
  float iTime=time*u_speed;
  for(int i=1;i<10;i++){
    vec2 np=p; float fi=float(i);
    np.x+=0.6/fi*sin(fi*p.y+iTime+0.3*fi)+1.0;
    np.y+=0.6/fi*sin(fi*p.x+iTime+0.3*(fi+10.0))-1.4;
    p=np;
  }
  float g=clamp(1.0 - sin(p.y), 0.0, 1.0);
  float b=sin(p.x+p.y)*0.5 + 0.5;
  float t = saturate(0.65*b + 0.35*g); // blend the two signals
  return triBlend(C1, C2, C3, t, u_midpoint, u_centerPower);
}
vec3 stripeGradient(vec2 uv,float time){
  vec2 p=((uv*u_resolution*2.0-u_resolution.xy)/(u_resolution.x+u_resolution.y)*2.0)*u_scale;
  float t = time*0.7, a = 4.0*p.y - sin(-p.x*3.0 + p.y - t);
  a = smoothstep(cos(a)*0.7, sin(a)*0.7+1.0, cos(a-4.0*p.y)-sin(a+3.0*p.x));
  vec2 warped=(cos(a)*p + sin(a)*vec2(-p.y,p.x))*0.5 + 0.5;
  float u = saturate(0.5*(warped.x + warped.y));
  return triBlend(C1, C2, C3, u, u_midpoint, u_centerPower);
}

// ---- Aspect-corrected swirl (circular in screen space) ----
vec2 swirl(vec2 uv, vec2 center, float strength, float falloff){
  vec2 p = uv - center;
  float aspect = u_resolution.x / u_resolution.y;
  vec2 ap = vec2(p.x * aspect, p.y);
  float r = length(ap);
  if (r < 1e-4) return uv;
  float ang = strength * exp(-r * falloff);
  float s = sin(ang), c = cos(ang);
  ap = mat2(c, -s, s, c) * ap;
  p = vec2(ap.x / aspect, ap.y);
  return center + p;
}

void main(){
  vec2 uv = vUv;
  float time = u_time * u_speed;

  // sRGB -> linear
  C1 = sRGBToLinear(u_color1);
  C2 = sRGBToLinear(u_color2);
  C3 = sRGBToLinear(u_color3);

  // Swirl around the mouse
  vec2 m = clamp(u_mouse, 0.0, 1.0);
  uv = swirl(uv, m, u_swirlStrength, u_swirlFalloff);

  // Choose gradient (all triBlend with mid/power)
  vec3 color;
  if (u_type == 0)      color = linearGradient(uv, time);
  else if (u_type == 1) color = conicGradient(uv, time);
  else if (u_type == 2) color = animatedGradient(uv, time);
  else if (u_type == 3) color = waveGradient(uv, time);
  else if (u_type == 4) color = silkGradient(uv, time);
  else if (u_type == 5) color = smokeGradient(uv, time);
  else if (u_type == 6) color = stripeGradient(uv, time);
  else                  color = animatedGradient(uv, time);

  // Optional film grain (linear)
  if (u_noise > 0.001) {
    float base = u_noise * 0.25;
    float strength = base * mix(1.2, 0.6, luma(color));
    float g = filmGrain(gl_FragCoord.xy, u_time, strength);
    color = max(color, 1e-3);
    color = exp(log(color) + g);
  }

  // Tiny dithering (linear)
  color += (ign(gl_FragCoord.xy + u_time * 60.0) - 0.5) * (u_dither / 255.0);

  // linear -> sRGB
  vec3 outColor = linearToSRGB(saturate(color));
  gl_FragColor = vec4(outColor, 1.0);
}
  `;

  // ---------- WebGL setup ----------
  const canvas = document.createElement('canvas');
  container.appendChild(canvas);
  const gl = canvas.getContext('webgl', { alpha:false, antialias:false, powerPreference:'high-performance' });
  if (!gl) { console.warn('WebGL not supported'); return; }

  function createShader(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  const vs = createShader(gl.VERTEX_SHADER, vertexShader);
  const fs = createShader(gl.FRAGMENT_SHADER, fragmentShader);
  if (!vs || !fs) return;

  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  const vertices = new Float32Array([ -1,-1, 1,-1, -1,1,  -1,1, 1,-1, 1,1 ]);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(prog, 'position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Uniforms
  const u_time          = gl.getUniformLocation(prog, 'u_time');
  const u_c1            = gl.getUniformLocation(prog, 'u_color1');
  const u_c2            = gl.getUniformLocation(prog, 'u_color2');
  const u_c3            = gl.getUniformLocation(prog, 'u_color3');
  const u_speed         = gl.getUniformLocation(prog, 'u_speed');
  const u_scale         = gl.getUniformLocation(prog, 'u_scale');
  const u_type          = gl.getUniformLocation(prog, 'u_type');
  const u_noise         = gl.getUniformLocation(prog, 'u_noise');
  const u_res           = gl.getUniformLocation(prog, 'u_resolution');
  const u_mouse         = gl.getUniformLocation(prog, 'u_mouse');
  const u_swirlStrength = gl.getUniformLocation(prog, 'u_swirlStrength');
  const u_swirlFalloff  = gl.getUniformLocation(prog, 'u_swirlFalloff');
  const u_dither        = gl.getUniformLocation(prog, 'u_dither');
  const u_midpoint      = gl.getUniformLocation(prog, 'u_midpoint');
  const u_centerPower   = gl.getUniformLocation(prog, 'u_centerPower');

  gl.uniform3fv(u_c1, new Float32Array(norm(cfg.color1)));
  gl.uniform3fv(u_c2, new Float32Array(norm(cfg.color2)));
  gl.uniform3fv(u_c3, new Float32Array(norm(cfg.color3)));
  gl.uniform1f(u_speed, cfg.speed);
  gl.uniform1f(u_scale, cfg.scale);
  gl.uniform1i(u_type, cfg.type|0);
  gl.uniform1f(u_noise, cfg.noise);
  gl.uniform1f(u_swirlStrength, cfg.swirlStrength);
  gl.uniform1f(u_swirlFalloff,  cfg.swirlFalloff);
  gl.uniform1f(u_dither,        cfg.dither);
  gl.uniform1f(u_midpoint,      cfg.midpoint);
  gl.uniform1f(u_centerPower,   cfg.centerPower);

  // Mouse tracking (normalized to canvas rect)
  let mouse = [0.5, 0.5];
  function updateMouse(e) {
    const r = canvas.getBoundingClientRect();
    mouse[0] = (e.clientX - r.left) / r.width;
    mouse[1] = 1 - (e.clientY - r.top) / r.height;
  }
  window.addEventListener('pointermove', updateMouse, { passive:true });

  // Resize
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = container.clientWidth | 0;
    const h = container.clientHeight | 0;
    const W = Math.max(1, Math.floor(w * dpr));
    const H = Math.max(1, Math.floor(h * dpr));
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W; canvas.height = H;
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
      gl.viewport(0, 0, W, H);
      gl.uniform2f(u_res, w, h);
    }
  }
  resize();
  window.addEventListener('resize', resize, { passive:true });

  // Animate
  const t0 = performance.now();
  function frame(t) {
    gl.uniform1f(u_time, (t - t0) / 1000);
    gl.uniform2f(u_mouse, mouse[0], mouse[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
