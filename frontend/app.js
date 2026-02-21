/* app.js - Shared utilities for ACTION Platform */
(() => {
  'use strict';

  const API = window.location.origin;

  /* API helper */
  window.api = {
    async get(path) {
      const res = await fetch(`${API}${path}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      return res.json();
    },

    async post(path, body = {}) {
      const res = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      if (res.status === 204) return null;
      return res.json();
    },

    async del(path) {
      const res = await fetch(`${API}${path}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      return null;
    },

    async put(path, body = {}) {
      const res = await fetch(`${API}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      if (res.status === 204) return null;
      return res.json();
    },
  };

  /* Toast notifications */
  window.showToast = function (message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(16px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  };

  /* Tab switching */
  window.switchTab = function (btn, tabId) {
    // Deactivate all tabs and content
    btn.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    // Show target tab content
    const panel = btn.closest('.panel');
    panel.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');
  };

  /* Button loading state */
  window.setLoading = function (btn, loading) {
    if (loading) {
      btn._originalText = btn.innerHTML;
      btn.classList.add('loading');
      btn.disabled = true;
    } else {
      btn.innerHTML = btn._originalText || btn.innerHTML;
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  };

  /* Format date */
  window.formatDate = function (isoStr) {
    if (!isoStr) return '—';
    const d = new Date(isoStr);
    return d.toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  /* Prettify stage name */
  window.stageName = function (key) {
    const names = {
      patient_screening: 'Patient Screening',
      cohort_formation: 'Cohort Formation',
      cohort_monitoring: 'Cohort Monitoring',
    };
    return names[key] || key;
  };

  /* Escape HTML */
  window.escapeHtml = function (str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  };

  /* Workflow context from URL params */
  window.getWorkflowContext = function () {
    const params = new URLSearchParams(window.location.search);
    return {
      workflowId: params.get('workflow_id'),
      trialName: params.get('trial_name'),
      workflowName: params.get('workflow_name'),
    };
  };

  /**
   * Require workflow context on stage pages.
   * If workflow_id is missing, redirect to /workflows.html.
   * Otherwise, populate the context banner and pre-fill trial name fields.
   */
  window.requireWorkflowContext = function () {
    const ctx = getWorkflowContext();
    if (!ctx.workflowId) {
      window.location.href = '/workflows.html';
      return null;
    }

    // Show context banner
    const banner = document.getElementById('wf-context-banner');
    if (banner) {
      banner.style.display = '';
      const nameEl = document.getElementById('wf-ctx-name');
      const trialEl = document.getElementById('wf-ctx-trial');
      if (nameEl) nameEl.textContent = ctx.workflowName || ctx.workflowId;
      if (trialEl) trialEl.textContent = ctx.trialName || '—';
    }

    // Update Back to Workflow button to link to the specific workflow
    const backBtn = document.getElementById('back-to-wf-btn');
    if (backBtn) {
      backBtn.href = '/workflows.html';
      backBtn.onclick = (e) => {
        e.preventDefault();
        window.location.href = '/workflows.html';
        // The modal will open once workflows page loads if needed
      };
    }

    // Pre-fill any trial name fields
    if (ctx.trialName) {
      ['nl-trial', 'struct-trial', 'mon-trial'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.value = ctx.trialName;
          el.readOnly = true;
          el.style.opacity = '0.7';
        }
      });
    }

    return ctx;
  };

  /**
   * Build URL to a stage page with workflow context.
   */
  window.stagePageUrl = function (page, workflowId, trialName, workflowName) {
    const params = new URLSearchParams({
      workflow_id: workflowId,
      trial_name: trialName || '',
      workflow_name: workflowName || '',
    });
    return `/${page}?${params.toString()}`;
  };

  /* Footer year */
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* Intersection-observer fade-up */
  const fadeEls = document.querySelectorAll('.fade-up:not(.visible)');
  if (fadeEls.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
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
    fadeEls.forEach((el) => el.classList.add('visible'));
  }

  /* == Background Job Helpers == */
  /**
   * Submit a background job for a stage and start polling.
   *
   * @param {object} opts
   * @param {string} opts.workflowId
   * @param {string} opts.stage        - e.g. "patient_screening"
   * @param {object} opts.payload      - request body for the worker
   * @param {string} opts.description  - human label
   * @param {function} opts.onComplete - (result) => void
   * @param {function} opts.onError    - (errorMsg) => void
   * @param {number}   [opts.pollMs=3000] - poll interval
   * @returns {Promise<string>}  job_id
   */
  window.submitJob = async function ({
    workflowId, stage, payload, description,
    onComplete, onError, pollMs = 3000,
  }) {
    let res;
    try {
      res = await api.post('/api/jobs/submit', {
        workflow_id: workflowId,
        stage,
        payload,
        description,
      });
    } catch (err) {
      // 409 = a job is already running for this stage
      if (err.message && err.message.includes('already')) {
        showToast('A job is already running for this stage. Please wait for it to finish.', 'error');
      } else {
        showToast(err.message || 'Failed to submit job', 'error');
      }
      if (onError) onError(err.message || 'Failed to submit job');
      return null;
    }

    const jobId = res.job_id;
    showJobBanner(description || 'Processing…');

    // Start polling
    const poll = setInterval(async () => {
      try {
        const job = await api.get(`/api/jobs/${jobId}`);
        if (job.status === 'completed') {
          clearInterval(poll);
          hideJobBanner();
          if (onComplete) onComplete(job.result);
        } else if (job.status === 'failed') {
          clearInterval(poll);
          hideJobBanner();
          if (onError) onError(job.error || 'Job failed');
        }
        // else still pending/running — keep polling
      } catch (err) {
        clearInterval(poll);
        hideJobBanner();
        if (onError) onError(err.message);
      }
    }, pollMs);

    // Store poll id so we can cancel on page unload
    window._activeJobPoll = poll;
    window._activeJobId = jobId;

    return jobId;
  };

  /**
   * Check if there is an active/completed job for a stage and resume
   * polling or invoke the callback immediately.
   *
   * Used on page load to recover from navigation-away.
   *
   * @param {object} opts
   * @param {string} opts.workflowId
   * @param {string} opts.stage
   * @param {function} opts.onComplete  - (result) => void
   * @param {function} opts.onError     - (errorMsg) => void
   * @param {number}   [opts.pollMs=3000]
   * @returns {Promise<boolean>}  true if a job was found and handled
   */
  window.checkForPendingJob = async function ({
    workflowId, stage, onComplete, onError, pollMs = 3000,
  }) {
    try {
      const job = await api.get(`/api/jobs/stage/${workflowId}/${stage}`);
      if (!job || !job.job_id) return false;

      if (job.status === 'completed') {
        // Job finished while user was away — deliver the result
        if (onComplete) onComplete(job.result);
        return true;
      }

      if (job.status === 'failed') {
        if (onError) onError(job.error || 'Job failed');
        return true;
      }

      // Still pending/running — show banner and resume polling
      showJobBanner(job.description || 'Processing…');

      const poll = setInterval(async () => {
        try {
          const updated = await api.get(`/api/jobs/${job.job_id}`);
          if (updated.status === 'completed') {
            clearInterval(poll);
            hideJobBanner();
            if (onComplete) onComplete(updated.result);
          } else if (updated.status === 'failed') {
            clearInterval(poll);
            hideJobBanner();
            if (onError) onError(updated.error || 'Job failed');
          }
        } catch (err) {
          clearInterval(poll);
          hideJobBanner();
          if (onError) onError(err.message);
        }
      }, pollMs);

      window._activeJobPoll = poll;
      window._activeJobId = job.job_id;
      return true;
    } catch {
      return false;
    }
  };

  /* Job banner (fixed bottom notification) */

  function _ensureBannerEl() {
    let el = document.getElementById('job-banner');
    if (!el) {
      el = document.createElement('div');
      el.id = 'job-banner';
      el.className = 'job-banner';
      el.style.display = 'none';
      el.innerHTML = `
        <div class="job-banner-inner">
          <div class="job-banner-spinner"></div>
          <div class="job-banner-text">
            <strong id="job-banner-title">Processing…</strong>
            <span id="job-banner-sub">This may take a moment. You can navigate away — we'll save the results.</span>
          </div>
        </div>
      `;
      document.body.appendChild(el);
    }
    return el;
  }

  window.showJobBanner = function (title) {
    const el = _ensureBannerEl();
    const titleEl = el.querySelector('#job-banner-title');
    if (titleEl) titleEl.textContent = title || 'Processing…';
    el.style.display = '';
  };

  window.hideJobBanner = function () {
    const el = document.getElementById('job-banner');
    if (el) el.style.display = 'none';
  };

  // Cleanup on unload
  window.addEventListener('beforeunload', () => {
    if (window._activeJobPoll) clearInterval(window._activeJobPoll);
  });
})();
