//latest 2

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
    dither:        clamp(+container.dataset.dither        || 1.0,  0.0, 2.0) // 1.0 ≈ 1 LSB
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

varying vec2 vUv;

#define PI 3.14159265359
#define S(a,b,t) smoothstep(a,b,t)

float saturate(float x){ return clamp(x,0.0,1.0); }
vec3  saturate(vec3 v){ return clamp(v,0.0,1.0); }

// ---- sRGB helpers (mix in linear, output sRGB) ----
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

// ---- Tiny screen-space dithering (interleaved gradient noise) ----
float ign(vec2 p){
  // stable, cheap, avoids banding without shimmering
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// ---- Film-style grain (optional, subtle) ----
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

// ---- Misc helpers from your original ----
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

// We'll mix colors in LINEAR space using these globals:
vec3 C1, C2, C3;

vec3 linearGradient(vec2 uv, float time){
  float t=(uv.y*u_scale)+sin(uv.x*PI+time)*0.1; t=clamp(t,0.0,1.0);
  return t<0.5? mix(C1,C2,t*2.0) : mix(C2,C3,(t-0.5)*2.0);
}
vec3 conicGradient(vec2 uv, float time){
  vec2 center=vec2(0.5), pos=uv-center; float angle=atan(pos.y,pos.x);
  float normalizedAngle=(angle+PI)/(2.0*PI);
  float t=fract(normalizedAngle*u_scale+time*0.3), s=t;
  vec3 color;
  if(s<0.33) color=mix(C1,C2,smoothstep(0.0,0.33,s));
  else if(s<0.66) color=mix(C2,C3,smoothstep(0.33,0.66,s));
  else color=mix(C3,C1,smoothstep(0.66,1.0,s));
  float dist=length(pos); color+=sin(dist*8.0+time*1.5)*0.03;
  return color;
}
vec3 animatedGradient(vec2 uv,float time){
  float ratio=u_resolution.x/u_resolution.y;
  vec2 tuv=uv; tuv-=0.5;
  float degree=advancedNoise(vec2(time*0.1*u_speed, tuv.x*tuv.y));
  tuv.y*=1.0/ratio; tuv*=Rot(radians((degree-0.5)*720.0*u_scale+180.0)); tuv.y*=ratio;
  float freq=5.0*u_scale, amp=30.0, sp=time*2.0*u_speed;
  tuv.x+=sin(tuv.y*freq+sp)/amp; tuv.y+=sin(tuv.x*freq*1.5+sp)/(amp*0.5);
  vec3 l1=mix(C1,C2,S(-0.3,0.2,(tuv*Rot(radians(-5.0))).x));
  vec3 l2=mix(C2,C3,S(-0.3,0.2,(tuv*Rot(radians(-5.0))).x));
  return mix(l1,l2,S(0.05,-0.2,tuv.y));
}
vec3 waveGradient(vec2 uv,float time){
  float y=uv.y;
  float w1=sin(uv.x*PI*u_scale*0.8+time*u_speed*0.5)*0.1;
  float w2=sin(uv.x*PI*u_scale*0.5+time*u_speed*0.3)*0.15;
  float w3=sin(uv.x*PI*u_scale*1.2+time*u_speed*0.8)*0.2;
  float fy=y+w1+w2+w3; float pat=smoothstep(0.0,1.0,clamp(fy,0.0,1.0));
  vec3 color;
  if(pat<0.33) color=mix(C1,C2,smoothstep(0.0,0.33,pat));
  else if(pat<0.66) color=mix(C2,C3,smoothstep(0.33,0.66,pat));
  else color=mix(C3,C1,smoothstep(0.66,1.0,pat));
  color += sin(uv.x*PI*2.0+time*u_speed)*cos(uv.y*PI*1.5+time*u_speed*0.7)*0.02;
  return clamp(color,0.0,1.0);
}
vec3 silkGradient(vec2 uv,float time){
  vec2 fragCoord=uv*u_resolution; vec2 inv=1.0/u_resolution.xy;
  vec2 cu=(fragCoord*2.0-u_resolution.xy)*inv; cu*=u_scale;
  float damp=1.0/(1.0+u_scale*0.1); float d=-time*u_speed*0.5; float a=0.0;
  for(float i=0.0;i<8.0;i++){ a+=cos(i-d-a*cu.x)*damp; d+=sin(cu.y*i+a)*damp; }
  d+=time*u_speed*0.5;
  vec3 ptn=vec3(cos(cu.x*d+a)*0.5+0.5, cos(cu.y*a+d)*0.5+0.5, cos((cu.x+cu.y)*(d+a)*0.5)*0.5+0.5);
  vec3 c1=mix(C1,C2,ptn.x), c2=mix(C2,C3,ptn.y), c3=mix(C3,C1,ptn.z);
  vec3 fin=mix(c1,c2,ptn.z); fin=mix(fin,c3,ptn.x*0.5);
  vec3 orig=vec3(cos(cu*vec2(d,a))*0.6+0.4, cos(a+d)*0.5+0.5);
  orig=cos(orig*cos(vec3(d,a,2.5))*0.5+0.5);
  return mix(fin, orig*fin, 0.3);
}
vec3 smokeGradient(vec2 uv,float time){
  float mr=min(u_resolution.x,u_resolution.y);
  vec2 fragCoord=uv*u_resolution;
  vec2 p=(2.0*fragCoord.xy-u_resolution.xy)/mr; p*=u_scale;
  float iTime=time*u_speed;
  for(int i=1;i<10;i++){
    vec2 np=p; float fi=float(i);
    np.x+=0.6/fi*sin(fi*p.y+iTime+0.3*fi)+1.0;
    np.y+=0.6/fi*sin(fi*p.x+iTime+0.3*(fi+10.0))-1.4;
    p=np;
  }
  float gPat=clamp(1.0 - sin(p.y),0.0,1.0);
  float bPat=sin(p.x+p.y)*0.5+0.5;
  vec3 c12=mix(C1,C2,gPat);
  vec3 color=mix(c12,C3,bPat);
  return clamp(color,0.0,1.0);
}
vec3 stripeGradient(vec2 uv,float time){
  vec2 p=((uv*u_resolution*2.0-u_resolution.xy)/(u_resolution.x+u_resolution.y)*2.0)*u_scale;
  float t=time*0.7, a=4.0*p.y - sin(-p.x*3.0 + p.y - t);
  a=smoothstep(cos(a)*0.7, sin(a)*0.7+1.0, cos(a-4.0*p.y)-sin(a+3.0*p.x));
  vec2 warped=(cos(a)*p + sin(a)*vec2(-p.y,p.x))*0.5 + 0.5;
  vec3 color=mix(C1,C2,warped.x);
  color=mix(color,C3,warped.y); color*=color+0.6*sqrt(color);
  return clamp(color,0.0,1.0);
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

  // Convert uniform sRGB colors -> linear for correct mixing
  C1 = sRGBToLinear(u_color1);
  C2 = sRGBToLinear(u_color2);
  C3 = sRGBToLinear(u_color3);

  // Swirl around the mouse
  vec2 m = clamp(u_mouse, 0.0, 1.0);
  uv = swirl(uv, m, u_swirlStrength, u_swirlFalloff);

  // Evaluate gradient AFTER the warp (in linear)
  vec3 color;
  if (u_type == 0)      color = linearGradient(uv, time);
  else if (u_type == 1) color = conicGradient(uv, time);
  else if (u_type == 2) color = animatedGradient(uv, time);
  else if (u_type == 3) color = waveGradient(uv, time);
  else if (u_type == 4) color = silkGradient(uv, time);
  else if (u_type == 5) color = smokeGradient(uv, time);
  else if (u_type == 6) color = stripeGradient(uv, time);
  else                  color = animatedGradient(uv, time);

  // Optional film grain (still in linear)
  if (u_noise > 0.001) {
    float base = u_noise * 0.25;
    float strength = base * mix(1.2, 0.6, luma(color));
    float g = filmGrain(gl_FragCoord.xy, u_time, strength);
    color = max(color, 1e-3);
    color = exp(log(color) + g);
  }

  // Add tiny dithering before quantization (linear space)
  color += (ign(gl_FragCoord.xy + u_time * 60.0) - 0.5) * (u_dither / 255.0);

  // Convert back to sRGB for output
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
    gl.uniform2f(u_mouse, mouse[0], mouse[1]);  // send mouse each frame
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
