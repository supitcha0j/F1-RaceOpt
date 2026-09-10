/* Local 3D mesh, perspective projection and depth-sorted faces; no CDN needed. */
(() => {
  const canvas = document.getElementById('race-car');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const faces = [];
  const add = (points, color) => faces.push({ points, color });
  function box(x,y,z,w,h,d,color) {
    const p = [[x-w/2,y,z-d/2],[x+w/2,y,z-d/2],[x+w/2,y+h,z-d/2],[x-w/2,y+h,z-d/2],
      [x-w/2,y,z+d/2],[x+w/2,y,z+d/2],[x+w/2,y+h,z+d/2],[x-w/2,y+h,z+d/2]];
    [[0,1,2,3],[4,7,6,5],[0,4,5,1],[3,2,6,7],[1,5,6,2],[0,3,7,4]].forEach((f,i)=>add(f.map(n=>p[n]),color.map(c=>Math.round(c*[.62,.8,.45,1,.72,.88][i]))));
  }
  function body(sections,color) {
    for(let i=1;i<sections.length;i++) {
      const ring = ([x,w,y,h]) => [[x,y,-w],[x,y,w],[x,y+h,w*.65],[x,y+h,-w*.65]];
      const a=ring(sections[i-1]),b=ring(sections[i]);
      for(let j=0;j<4;j++) add([a[j],b[j],b[(j+1)%4],a[(j+1)%4]],color.map(c=>Math.round(c*[.5,.75,1,.85][j])));
      if(i===1) add(a,color);
      if(i===sections.length-1) add(b,color);
    }
  }
  function wheel(x,z) {
    const n=32,r=.46,y=.49,depth=.34;
    for(let i=0;i<n;i++) {
      const a=i*2*Math.PI/n,b=(i+1)*2*Math.PI/n;
      const point=(t,zz,rr=r)=>[x+Math.cos(t)*rr,y+Math.sin(t)*rr,zz];
      const shade=Math.round(25+14*Math.max(0,Math.sin(a)));
      add([point(a,z-depth/2),point(b,z-depth/2),point(b,z+depth/2),point(a,z+depth/2)],[shade,shade,shade]);
      for(const side of [-1,1]) {
        const zz=z+side*depth/2;
        add([[x,y,zz],point(a,zz),point(b,zz)],[18,18,20]);
        add([point(a,zz+side*.002,r*.8),point(b,zz+side*.002,r*.8),point(b,zz+side*.002,r*.74),point(a,zz+side*.002,r*.74)],[224,185,45]);
        add([[x,y,zz+side*.004],point(a,zz+side*.004,r*.43),point(b,zz+side*.004,r*.43)],i%4===0?[112,112,116]:[47,47,50]);
      }
    }
  }
  const red=[230,30,20],carbon=[35,36,39];
  box(0,.19,0,4.65,.10,1.5,carbon);
  body([[-2.75,.12,.36,.13],[-1.25,.23,.33,.35],[-.45,.4,.32,.48],[.8,.42,.32,.5],[1.8,.23,.32,.35],[2.05,.15,.32,.2]],red);
  // Sidepods, cockpit, air intake and front/rear aerodynamic wings.
  box(.55,.3,-.49,1.85,.31,.37,red); box(.55,.3,.49,1.85,.31,.37,red);
  box(-.28,.79,0,.65,.055,.42,[14,14,16]);
  body([[.12,.17,.75,.04],[.48,.17,.75,.5],[.85,.12,.66,.22],[1.5,.06,.55,.08]],red);
  box(-.35,.85,-.24,.8,.045,.045,carbon); box(-.35,.85,.24,.8,.045,.045,carbon);
  box(-.74,.77,0,.05,.13,.5,carbon);
  box(-2.44,.25,0,.46,.065,2.0,red); box(-2.22,.34,0,.2,.06,1.96,carbon);
  box(2.05,.95,0,.48,.10,1.8,red); box(2.08,.44,0,.08,.52,.09,carbon);
  for(const z of [-.91,.91]) { box(2.05,.61,z,.57,.48,.035,red); box(-2.44,.2,z,.5,.28,.035,red); }
  for(const x of [-1.62,1.48]) for(const z of [-.97,.97]) { box(x,.4,z/2,.12,.05,.86,carbon); wheel(x,z); }
  let yaw=.55,pitch=.28,width=0,height=0,drag=null;
  function transform([x,y,z]) {
    const a=x*Math.cos(yaw)+z*Math.sin(yaw), b=-x*Math.sin(yaw)+z*Math.cos(yaw);
    return [a,y*Math.cos(pitch)-b*Math.sin(pitch),y*Math.sin(pitch)+b*Math.cos(pitch)];
  }
  function draw() {
    ctx.clearRect(0,0,width,height);
    const scale=Math.min(width/6.8,height/2.8);
    const project=([x,y,z])=>[width/2+x*scale*8/(8-z),height*.66-y*scale*8/(8-z)];
    const shadow=ctx.createRadialGradient(width/2,height*.72,0,width/2,height*.72,width*.36);
    shadow.addColorStop(0,'#000b'); shadow.addColorStop(1,'#0000');
    ctx.save(); ctx.translate(0,height*.58); ctx.scale(1,.24); ctx.fillStyle=shadow; ctx.fillRect(0,-height,width,height*4); ctx.restore();
    faces.map(f=>({p:f.points.map(transform),color:f.color})).sort((a,b)=>a.p.reduce((s,p)=>s+p[2],0)/a.p.length-b.p.reduce((s,p)=>s+p[2],0)/b.p.length).forEach(f=>{
      ctx.beginPath(); f.p.map(project).forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.closePath();
      ctx.fillStyle=`rgb(${f.color.join(',')})`; ctx.fill(); ctx.strokeStyle=ctx.fillStyle; ctx.lineWidth=.45; ctx.stroke();
    });
  }
  function resize() { const r=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2); width=r.width;height=r.height;canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw(); }
  new ResizeObserver(resize).observe(canvas);
  canvas.parentElement.classList.add('ready');
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag.x)*.008;pitch=Math.max(.08,Math.min(.8,pitch+(e.clientY-drag.y)*.004));drag={x:e.clientX,y:e.clientY};clearSelection();draw();});
  ['pointerup','pointercancel','lostpointercapture'].forEach(name=>canvas.addEventListener(name,()=>drag=null));
  function clearSelection(){document.querySelectorAll('[data-car-view]').forEach(b=>b.setAttribute('aria-pressed','false'));}
  document.querySelectorAll('[data-car-view]').forEach(button=>button.addEventListener('click',()=>{yaw=Number(button.dataset.carView);pitch=.28;clearSelection();button.setAttribute('aria-pressed','true');draw();}));
  canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(e.key))return;e.preventDefault();if(e.key==='Home'){yaw=.55;pitch=.28;}else if(e.key==='ArrowLeft')yaw-=.12;else if(e.key==='ArrowRight')yaw+=.12;else pitch=Math.max(.08,Math.min(.8,pitch+(e.key==='ArrowUp'?.08:-.08)));clearSelection();draw();});
})();
