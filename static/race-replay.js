/* Perspective-projected race scene. Rank spacing is illustrative, not telemetry. */
window.createRaceReplay = function (progress) {
  const canvas = document.getElementById('raceReplayCanvas');
  const ctx = canvas && canvas.getContext('2d');
  if (!ctx) return null;
  const status = document.getElementById('raceReplayStatus');
  const event = document.getElementById('raceReplayEvent');
  const cameraButton = document.getElementById('replayCamera');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const palette = ['#35d6cd', '#ffad42', '#8e91ff', '#5ebdff', '#eddf65', '#ee7dc3', '#ced7e0', '#7ddd7f'];
  const entries = [{code:'YOU', positions:progress.user, color:'#ff4336'}].concat(
    progress.rivals.map((r, i) => ({...r, color:palette[i % palette.length]}))
  );
  let lap = progress.labels.length, playing = false, speed = 1, overview = false;
  let width = 0, height = 0, phase = 0, frame = 0, lastTime = 0;
  let startTime = 0, duration = 0, visible = true;
  let positions = entries.map(e => e.positions[lap - 1]);
  let from = positions.slice(), target = positions.slice();

  function project(x, y, z) {
    const distance = z + (overview ? 40 : 21);
    const scale = Math.min(width * .9, height * 1.5) / Math.max(5, distance);
    const bend = Math.sin(z * .025 + phase * .014) * z * .055;
    return [width / 2 + (x + bend) * scale, height * (overview ? .24 : .34) + ((overview ? 21 : 8) - y) * scale, scale];
  }
  function polygon(points, fill, stroke) {
    ctx.beginPath();
    points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = .7; ctx.stroke(); }
  }
  function ground(x1, x2, z1, z2, color) {
    polygon([[x1,0,z1],[x2,0,z1],[x2,0,z2],[x1,0,z2]].map(p => project(...p)), color);
  }
  function box(x, y, z, w, h, d, color) {
    const p = [[x-w/2,y,z-d/2],[x+w/2,y,z-d/2],[x+w/2,y,z+d/2],[x-w/2,y,z+d/2],
      [x-w/2,y+h,z-d/2],[x+w/2,y+h,z-d/2],[x+w/2,y+h,z+d/2],[x-w/2,y+h,z+d/2]].map(v => project(...v));
    polygon([p[1],p[2],p[6],p[5]], color);
    polygon([p[0],p[3],p[7],p[4]], color);
    polygon([p[0],p[1],p[5],p[4]], color);
    polygon([p[4],p[5],p[6],p[7]], color, '#ffffff35');
    polygon([p[0],p[1],p[5],p[4]], '#00000035');
  }
  function car(entry, index) {
    // Positions use rank differences only; lateral lanes keep nearby cars readable.
    const rank = positions[index];
    const own = positions[0];
    const z = overview ? 14 + (progress.field_size - rank) * 3.8 : 17 + (own - rank) * 5.4;
    if (z < -9 || z > 150) return;
    const pit = index === 0 && (progress.pit_laps || []).includes(lap);
    const x = pit ? 9 : index === 0 ? -.8 : [-4.6,3.8,-2.8,1.6][index % 4];
    const p = project(x,0,z);
    ctx.fillStyle = '#00000065'; ctx.beginPath(); ctx.ellipse(p[0],p[1]+3,1.7*p[2],.6*p[2],0,0,Math.PI*2); ctx.fill();
    for (const side of [-1,1]) {
      box(x+side*1.05,.12,z-1.2,.65,.85,.95,'#10151d');
      box(x+side*1.05,.12,z+1.25,.55,.8,.85,'#10151d');
      box(x+side*1.39,.32,z-1.2,.04,.34,.6,'#ddc65a');
      box(x+side*.62,.3,z,.55,.55,1.9,entry.color);
    }
    box(x,.2,z,.8,.65,3.2,entry.color);
    box(x,.35,z+1.8,.38,.25,1.1,entry.color);
    box(x,.24,z+2.2,2.8,.13,.42,'#263345');
    box(x,.9,z-1.65,2.7,.2,.52,entry.color);
    box(x,.76,z-.2,.6,.2,.9,'#0b1420');
    box(x,1,z-.35,.35,.32,.4,index === 0 ? '#f4ffb1' : '#dceaff');
    box(x,1.17,z-.2,.8,.08,.1,'#263345');
    const label = project(x,2.9,z);
    const text = entry.code + (width < 480 && index !== 0 ? '' : ' · P' + Math.round(rank));
    ctx.font = 'bold ' + (index === 0 ? 12 : 10) + 'px monospace';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = index === 0 ? '#e9362b' : '#0b172de0';
    ctx.fillRect(label[0]-tw/2-7,label[1]-13,tw+14,22);
    ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.fillText(text,label[0],label[1]+2);
  }
  function draw() {
    const sky = ctx.createLinearGradient(0,0,0,height);
    sky.addColorStop(0,'#0d2037'); sky.addColorStop(.5,'#526b7c'); sky.addColorStop(1,'#172c36');
    ctx.fillStyle = sky; ctx.fillRect(0,0,width,height);
    // Distant grandstand, skyline and floodlights.
    for (let i=0; i<18; i++) {
      const x = i*width/17;
      ctx.fillStyle = i%2 ? '#20394b' : '#294354';
      ctx.fillRect(x,height*.24-(i%4)*7,width/16,height*.19);
      ctx.fillStyle = '#b4c8d14a';
      for(let j=0;j<4;j++) ctx.fillRect(x+3,height*.26+j*8,width/22,2);
    }
    for(let z=160;z>-15;z-=3) {
      const stripe = Math.floor((z+phase)/3)%2 === 0;
      ground(-70,70,z,z+3,stripe ? '#254940' : '#294e43');
      ground(-6.8,6.8,z,z+3,stripe ? '#303d4a' : '#33404d');
      ground(7.5,10.8,z,z+3,'#3b505c');
      ground(-7.35,-6.8,z,z+3,stripe ? '#f5efe6' : '#d83734');
      ground(6.8,7.35,z,z+3,stripe ? '#f5efe6' : '#d83734');
      ground(10.7,10.85,z,z+3,'#66dfe0');
      if (stripe) { ground(-2.3,-2.23,z,z+1.6,'#b8c9d344'); ground(2.23,2.3,z,z+1.6,'#b8c9d344'); }
      if(Math.floor(z)%18===1) {
        box(-8.2,0,z,.15,7,.15,'#a8bac4');
        box(-7.7,7,z,1.5,.18,.4,'#e2f4ff');
        box(12,0,z,1.2,1.4,2,'#17616a');
      }
    }
    // Checkered finish marking; visible only on the last lap.
    if (lap === progress.labels.length) {
      for(let i=0;i<14;i++) for(let j=0;j<2;j++) ground(-6.8+i*.97,-6.8+(i+1)*.97,25+j,26+j,(i+j)%2?'#17222d':'#f1f5eb');
    }
    const order = entries.map((e,i)=>i).sort((a,b)=>positions[a]-positions[b]);
    order.forEach(i => car(entries[i],i));
    ctx.fillStyle = '#ffffff1a'; ctx.fillRect(18,height-5,width-36,2);
    ctx.fillStyle = '#b9ff87'; ctx.fillRect(18,height-5,(width-36)*lap/progress.labels.length,2);
  }
  function sample(now) {
    const t = duration ? Math.min(1,(now-startTime)/duration) : 1;
    const eased = t*t*(3-2*t);
    positions = target.map((p,i)=>from[i]+(p-from[i])*eased);
    return t;
  }
  function loop(now) {
    frame = 0;
    const dt = lastTime ? Math.min(now-lastTime,50) : 0;
    lastTime = now;
    const t = sample(now);
    if (playing && !motion.matches) phase += dt*.018*speed;
    draw();
    if (visible && !document.hidden && !motion.matches && (playing || t<1)) frame = requestAnimationFrame(loop);
  }
  function requestDraw() {
    if (!frame && visible && !document.hidden) { lastTime=0; frame=requestAnimationFrame(loop); }
  }
  function resize() {
    const rect = canvas.getBoundingClientRect();
    width=rect.width; height=rect.height;
    const dpr=Math.min(devicePixelRatio || 1,2);
    canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0); requestDraw();
  }
  const observer = new ResizeObserver(resize); observer.observe(canvas);
  const visibility = new IntersectionObserver(records => {
    visible=records[0].isIntersecting;
    if (visible) requestDraw(); else { cancelAnimationFrame(frame); frame=0; }
  });
  visibility.observe(canvas);
  cameraButton.addEventListener('click',()=>{
    overview=!overview;
    cameraButton.textContent=overview?'มุมกล้อง: มุมสูง ↙':'มุมกล้อง: ไล่ตาม ↗';
    cameraButton.setAttribute('aria-label',overview?'เปลี่ยนเป็นมุมกล้องไล่ตาม':'เปลี่ยนเป็นมุมกล้องมุมสูง');
    requestDraw();
  });
  motion.addEventListener('change',()=> { duration=0; requestDraw(); });
  window.addEventListener('pagehide',()=>{ cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect(); },{once:true});
  return {
    setLap(next, ms=0) {
      sample(performance.now());
      lap=next; from=positions.slice(); target=entries.map(e=>e.positions[lap-1]);
      startTime=performance.now(); duration=motion.matches ? 0 : ms*.85;
      const pos=progress.user[lap-1], prev=progress.user[Math.max(0,lap-2)], delta=prev-pos;
      const pit=(progress.pit_laps || []).includes(lap);
      status.textContent='LAP '+lap+' / '+progress.labels.length+' · YOUR POSITION P'+pos;
      event.textContent=lap===progress.labels.length ? 'CHEQUERED FLAG · P'+pos : pit ? 'PIT STOP · เข้าพิต' : lap===1 ? 'LIGHTS OUT · เริ่มการแข่งขัน' : delta>0 ? '▲ ขึ้น '+delta+' อันดับ' : delta<0 ? '▼ ลดลง '+(-delta)+' อันดับ' : 'รักษาตำแหน่ง · P'+pos;
      event.style.color=pit?'#6ae4ee':delta<0?'#ffb28c':'#b9ff87';
      canvas.setAttribute('aria-label','รอบ '+lap+' คุณอยู่อันดับ '+pos+(pit?' กำลังเข้าพิต':'')+' แสดงรถคู่แข่ง '+progress.rivals.length+' คัน');
      requestDraw();
    },
    setPlaying(value, multiplier=1) {
      playing=value; speed=multiplier;
      if (!value) { duration=0; positions=target.slice(); }
      requestDraw();
    }
  };
};
