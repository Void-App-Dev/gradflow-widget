//neww

<script>
(function () {
  const container = document.getElementById('gradflow');
  if (!container) return;

  const clamp = (v,min,max)=>Math.max(min,Math.min(max,v));
  const parseRGB = (s, fb)=> {
    const a = String(s||'').split(',').map(n=>+n.trim());
    return a.length===3 && a.every(Number.isFinite) ? a : fb;
  };
  const typeMap = { linear:0, conic:1, animated:2, wave:3, silk:4, smoke:5, stripe:6 };

  const cfg = {
    color1: parseRGB(container.dataset.color1, [226,98,75]),
    color2: parseRGB(container.dataset.color2, [255,255,255]),
    color3: parseRGB(container.dataset.color3, [30,34,159]),
    speed: +container.dataset.speed || 0.4,
    scale: +container.dataset.scale || 1,
    type: typeMap[(container.dataset.type || 'stripe').toLowerCase()] ?? 6,
    noise: clamp(+container.dataset.noise || 0.08, 0, 1)
  };
  const norm = (rgb)=>[rgb[0]/255, rgb[1]/255, rgb[2]/255];

  // --- Shaders ---
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
uniform vec3 u_color1;
uniform vec3 u_color2;
uniform vec3 u_color3;
uniform float u_speed;
uniform float u_scale;
uniform int u_type;
uniform float u_noise;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

varying vec2 vUv;

#define PI 3.14159265359

// --- Film-style grain helpers ---
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
  for (int i = 0; i < 4; i++) {
    s += a * valueNoise(p);
    p *= 2.0;
    a *= 0.5;
  }
  return s;
}

float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float filmGrain(vec2 px, float time, float strength) {
  float frame = floor(time * 24.0);
  float ang = frame * 2.39996323;
  mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
  vec2 rp = R * (px * 0.75);
  float g = fbm(rp + frame); // 0..1
  return (g - 0.5) * 2.0 * strength;
}

// --- Gradient utils from your original ---
float noise(vec2 st) {
  return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 linearGradient(vec2 uv, float time) {
  float t = (uv.y * u_scale) + sin(uv.x * PI + time) * 0.1;
  t = clamp(t, 0.0, 1.0);
  return t < 0.5
    ? mix(u_color1, u_color2, t * 2.0)
    : mix(u_color2, u_color3, (t - 0.5) * 2.0);
}

vec3 conicGradient(vec2 uv, float time) {
  vec2 center = vec2(0.5);
  vec2 pos = uv - center;
  float angle = atan(pos.y, pos.x);
  float normalizedAngle = (angle + PI) / (2.0 * PI);
  float t = fract(normalizedAngle * u_scale + time * 0.3);
  float smoothT = t;
  vec3 color;
  if (smoothT < 0.33) {
    color = mix(u_color1, u_color2, smoothstep(0.0, 0.33, smoothT));
  } else if (smoothT < 0.66) {
    color = mix(u_color2, u_color3, smoothstep(0.33, 0.66, smoothT));
  } else {
    color = mix(u_color3, u_color1, smoothstep(0.66, 1.0, smoothT));
  }
  float dist = length(pos);
  color += sin(dist * 8.0 + time * 1.5) * 0.03;
  return color;
}

#define S(a,b,t) smoothstep(a,b,t)

mat2 Rot(float a) {
  float s = sin(a);
  float c = cos(a);
  return mat2(c, -s, s, c);
}

vec2 hash(vec2 p) {
  p = vec2(dot(p, vec2(2127.1, 81.17)), dot(p, vec2(1269.5, 283.37)));
  return fract(sin(p) * 43758.5453);
}

float advancedNoise(in vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float n = mix(mix(dot(-1.0 + 2.0 * hash(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)), 
                    dot(-1.0 + 2.0 * hash(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
                mix(dot(-1.0 + 2.0 * hash(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)), 
                    dot(-1.0 + 2.0 * hash(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
  return 0.5 + 0.5 * n;
}

vec3 animatedGradient(vec2 uv, float time) {
  float ratio = u_resolution.x / u_resolution.y;
  vec2 tuv = uv;
  tuv -= 0.5;
  float degree = advancedNoise(vec2(time * 0.1 * u_speed, tuv.x * tuv.y));
  tuv.y *= 1.0 / ratio;
  tuv *= Rot(radians((degree - 0.5) * 720.0 * u_scale + 180.0));
  tuv.y *= ratio;
  float frequency = 5.0 * u_scale;
  float amplitude = 30.0;
  float speed = time * 2.0 * u_speed;
  tuv.x += sin(tuv.y * frequency + speed) / amplitude;
  tuv.y += sin(tuv.x * frequency * 1.5 + speed) / (amplitude * 0.5);
  vec3 layer1 = mix(u_color1, u_color2, S(-0.3, 0.2, (tuv * Rot(radians(-5.0))).x));
  vec3 layer2 = mix(u_color2, u_color3, S(-0.3, 0.2, (tuv * Rot(radians(-5.0))).x));
  vec3 finalComp = mix(layer1, layer2, S(0.05, -0.2, tuv.y));
  return finalComp;
}

vec3 waveGradient(vec2 uv, float time) {
  float y = uv.y;
  float wave1 = sin(uv.x * PI * u_scale * 0.8 + time * u_speed * 0.5) * 0.1;
  float wave2 = sin(uv.x * PI * u_scale * 0.5 + time * u_speed * 0.3) * 0.15;  
  float wave3 = sin(uv.x * PI * u_scale * 1.2 + time * u_speed * 0.8) * 0.2; 
  float flowingY = y + wave1 + wave2 + wave3;
  float pattern = smoothstep(0.0, 1.0, clamp(flowingY, 0.0, 1.0));
  vec3 color;
  if (pattern < 0.33) {
    float t = smoothstep(0.0, 0.33, pattern);
    color = mix(u_color1, u_color2, t);
  } else if (pattern < 0.66) {
    float t = smoothstep(0.33, 0.66, pattern);
    color = mix(u_color2, u_color3, t);
  } else {
    float t = smoothstep(0.66, 1.0, pattern);
    color = mix(u_color3, u_color1, t);
  }
  float variation = sin(uv.x * PI * 2.0 + time * u_speed) *
                    cos(uv.y * PI * 1.5 + time * u_speed * 0.7) * 0.02;
  color += variation;
  return clamp(color, 0.0, 1.0);
}

vec3 silkGradient(vec2 uv, float time) {
  vec2 fragCoord = uv * u_resolution;
  vec2 invResolution = 1.0 / u_resolution.xy;
  vec2 centeredUv = (fragCoord * 2.0 - u_resolution.xy) * invResolution;
  centeredUv *= u_scale;
  float dampening = 1.0 / (1.0 + u_scale * 0.1);
  float d = -time * u_speed * 0.5;
  float a = 0.0;
  for (float i = 0.0; i < 8.0; ++i) {
      a += cos(i - d - a * centeredUv.x) * dampening;
      d += sin(centeredUv.y * i + a) * dampening;
  }
  d += time * u_speed * 0.5;
  vec3 patterns = vec3(
    cos(centeredUv.x * d + a) * 0.5 + 0.5,
    cos(centeredUv.y * a + d) * 0.5 + 0.5,
    cos((centeredUv.x + centeredUv.y) * (d + a) * 0.5) * 0.5 + 0.5
  );
  vec3 color1Mix = mix(u_color1, u_color2, patterns.x);
  vec3 color2Mix = mix(u_color2, u_color3, patterns.y);
  vec3 color3Mix = mix(u_color3, u_color1, patterns.z);
  vec3 finalColor = mix(color1Mix, color2Mix, patterns.z);
  finalColor = mix(finalColor, color3Mix, patterns.x * 0.5);
  vec3 originalPattern = vec3(cos(centeredUv * vec2(d, a)) * 0.6 + 0.4, cos(a + d) * 0.5 + 0.5);
  originalPattern = cos(originalPattern * cos(vec3(d, a, 2.5)) * 0.5 + 0.5);
  return mix(finalColor, originalPattern * finalColor, 0.3);
}

vec3 smokeGradient(vec2 uv, float time) {
  float mr = min(u_resolution.x, u_resolution.y);
  vec2 fragCoord = uv * u_resolution;
  vec2 p = (2.0 * fragCoord.xy - u_resolution.xy) / mr;
  p *= u_scale;
  float iTime = time * u_speed;
  for(int i = 1; i < 10; i++) {
    vec2 newp = p;
    float fi = float(i);
    newp.x += 0.6 / fi * sin(fi * p.y + iTime + 0.3 * fi) + 1.0;
    newp.y += 0.6 / fi * sin(fi * p.x + iTime + 0.3 * (fi + 10.0)) - 1.4;
    p = newp;
  }
  float greenPattern = clamp(1.0 - sin(p.y), 0.0, 1.0);
  float bluePattern = sin(p.x + p.y) * 0.5 + 0.5;
  vec3 color12 = mix(u_color1, u_color2, greenPattern);
  vec3 color = mix(color12, u_color3, bluePattern);
  return clamp(color, 0.0, 1.0);
}

vec3 stripeGradient(vec2 uv, float time) {
  vec2 p = ((uv * u_resolution * 2.0 - u_resolution.xy) / (u_resolution.x + u_resolution.y) * 2.0) * u_scale;
  float t = time * 0.7, a = 4.0 * p.y - sin(-p.x * 3.0 + p.y - t);
  a = smoothstep(cos(a) * 0.7, sin(a) * 0.7 + 1.0, cos(a - 4.0 * p.y) - sin(a + 3.0 * p.x));
  vec2 warped = (cos(a) * p + sin(a) * vec2(-p.y, p.x)) * 0.5 + 0.5;
  vec3 color = mix(u_color1, u_color2, warped.x);
  color = mix(color, u_color3, warped.y);
  color *= color + 0.6 * sqrt(color);
  return clamp(color, 0.0, 1.0);
}


void main() {
  vec2 uv = vUv;
  float time = u_time * u_speed;

  // --- Mouse distortion ---
  float dist = distance(uv, u_mouse);
  vec2 dir = normalize(uv - u_mouse);
  uv += dir * 0.08 * exp(-dist * 12.0);

  vec3 color;
  if (u_type == 0) color = linearGradient(uv, time);
  else if (u_type == 1) color = conicGradient(uv, time);
  else if (u_type == 2) color = animatedGradient(uv, time);
  else if (u_type == 3) color = waveGradient(uv, time);
  else if (u_type == 4) color = silkGradient(uv, time);
  else if (u_type == 5) color = smokeGradient(uv, time);
  else if (u_type == 6) color = stripeGradient(uv, time);
  else color = animatedGradient(uv, time);

  gl_FragColor = vec4(color, 1.0);
}
  `;

  // --- WebGL setup ---
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

  const vertices = new Float32Array([
    -1,-1,  1,-1, -1, 1,
    -1, 1,  1,-1,  1, 1
  ]);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const posLoc = gl.getAttribLocation(prog, 'position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const u_time  = gl.getUniformLocation(prog, 'u_time');
  const u_c1    = gl.getUniformLocation(prog, 'u_color1');
  const u_c2    = gl.getUniformLocation(prog, 'u_color2');
  const u_c3    = gl.getUniformLocation(prog, 'u_color3');
  const u_speed = gl.getUniformLocation(prog, 'u_speed');
  const u_scale = gl.getUniformLocation(prog, 'u_scale');
  const u_type  = gl.getUniformLocation(prog, 'u_type');
  const u_noise = gl.getUniformLocation(prog, 'u_noise');
  const u_res   = gl.getUniformLocation(prog, 'u_resolution');
  const u_mouse = gl.getUniformLocation(prog, 'u_mouse');

  gl.uniform3fv(u_c1, new Float32Array(norm(cfg.color1)));
  gl.uniform3fv(u_c2, new Float32Array(norm(cfg.color2)));
  gl.uniform3fv(u_c3, new Float32Array(norm(cfg.color3)));
  gl.uniform1f(u_speed, cfg.speed);
  gl.uniform1f(u_scale, cfg.scale);
  gl.uniform1i(u_type, cfg.type|0);
  gl.uniform1f(u_noise, cfg.noise);

  // Track mouse position
  let mouse = [0.5, 0.5];
  window.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect();
    mouse[0] = (e.clientX - rect.left) / rect.width;
    mouse[1] = 1.0 - (e.clientY - rect.top) / rect.height;
  });

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
  const onResize = ()=>resize();
  window.addEventListener('resize', onResize, { passive:true });

  const t0 = performance.now();
  let rafId = 0;
  function frame(t) {
    gl.uniform1f(u_time, (t - t0) / 1000);
    gl.uniform2f(u_mouse, mouse[0], mouse[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    rafId = requestAnimationFrame(frame);
  }
  rafId = requestAnimationFrame(frame);

  window.addEventListener('pagehide', function () {
    cancelAnimationFrame(rafId);
    window.removeEventListener('resize', onResize);
  });
})();
</script>
