(() => {
  const nav = document.querySelector('.nav');
  const toggle = nav.querySelector('.nav-toggle');
  const menu = nav.querySelector('#primary-menu');
  const mobile = window.matchMedia('(max-width: 719px)');

  const setOpen = (open) => {
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'ปิดเมนู' : 'เปิดเมนู');
    nav.classList.toggle('is-open', open);
  };

  toggle.addEventListener('click', () => {
    setOpen(toggle.getAttribute('aria-expanded') !== 'true');
  });
  menu.addEventListener('click', (event) => {
    if (event.target.closest('a')) setOpen(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
      setOpen(false);
      toggle.focus();
    }
  });
  document.addEventListener('click', (event) => {
    if (!nav.contains(event.target)) setOpen(false);
  });
  mobile.addEventListener('change', () => {
    const focused = document.activeElement;
    setOpen(false);
    if (mobile.matches && menu.contains(focused)) toggle.focus();
    if (!mobile.matches && focused === toggle) menu.querySelector('a').focus();
  });
  nav.classList.add('nav-ready');
})();
