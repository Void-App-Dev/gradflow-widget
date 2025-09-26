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

// (gradient + noise functions unchanged …)

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
