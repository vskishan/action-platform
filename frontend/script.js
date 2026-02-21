/* script.js */
(() => {
  'use strict';

  /* Footer year */
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* Header scroll effect */
  const header = document.querySelector('.site-header');
  if (header) {
    const onScroll = () => {
      header.classList.toggle('scrolled', window.scrollY > 32);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* Intersection-observer fade-up */
  const fadeEls = document.querySelectorAll('.fade-up');
  if (fadeEls.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            /* stagger children if parent has .stagger */
            const parent = e.target.closest('.stagger');
            if (parent) {
              const siblings = [...parent.querySelectorAll('.fade-up')];
              const idx = siblings.indexOf(e.target);
              e.target.style.transitionDelay = `${idx * 80}ms`;
            }
            e.target.classList.add('visible');
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );
    fadeEls.forEach((el) => io.observe(el));
  } else {
    /* fallback – show immediately */
    fadeEls.forEach((el) => el.classList.add('visible'));
  }

  /* Active nav link highlighting */
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if (navLinks.length && 'IntersectionObserver' in window) {
    const sections = [...navLinks].map((a) =>
      document.querySelector(a.getAttribute('href'))
    ).filter(Boolean);

    const navIO = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            navLinks.forEach((l) => l.classList.remove('active'));
            const match = [...navLinks].find(
              (l) => l.getAttribute('href') === `#${e.target.id}`
            );
            if (match) match.classList.add('active');
          }
        });
      },
      { threshold: 0.25 }
    );
    sections.forEach((s) => navIO.observe(s));
  }

  /* Smooth scroll for anchor links */
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();
