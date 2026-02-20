/* ─── workflows.js ─── Workflow Studio page logic ─── */
(() => {
  'use strict';

  const STAGES = ['patient_screening', 'cohort_formation', 'cohort_monitoring'];

  // Map stage keys to their dedicated pages
  const STAGE_PAGES = {
    patient_screening: 'screening.html',
    cohort_formation:  'cohort.html',
    cohort_monitoring: 'monitoring.html',
  };

  /* ══════════════════════════════════════════════════════
     Load & Render Workflows
     ══════════════════════════════════════════════════════ */
  window.loadWorkflows = async function () {
    const listEl = document.getElementById('wf-list');
    const emptyEl = document.getElementById('wf-empty');
    const loadingEl = document.getElementById('wf-loading');
    const countEl = document.getElementById('wf-count');

    listEl.innerHTML = '';
    emptyEl.style.display = 'none';
    loadingEl.style.display = '';

    try {
      const data = await api.get('/api/workflow');
      loadingEl.style.display = 'none';

      const workflows = data.workflows || [];
      countEl.textContent = `${workflows.length} workflow${workflows.length !== 1 ? 's' : ''}`;

      if (workflows.length === 0) {
        emptyEl.style.display = '';
        return;
      }

      workflows
        .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
        .forEach(wf => {
          listEl.appendChild(renderWorkflowCard(wf));
        });
    } catch (err) {
      loadingEl.style.display = 'none';
      countEl.textContent = 'Failed to load';
      showToast(err.message, 'error');
    }
  };

  function renderWorkflowCard(wf) {
    const card = document.createElement('div');
    card.className = 'wf-card';
    card.onclick = () => openWorkflowDetail(wf.id);

    // Stage dots — show "current" for the active stage if it's still not_started
    const cardCurrentIdx = wf.current_stage ? STAGES.indexOf(wf.current_stage) : -1;
    const dots = STAGES.map((s, idx) => {
      let dotStatus = (wf.stages_summary && wf.stages_summary[s]) || 'not_started';
      if (wf.status !== 'completed' && idx === cardCurrentIdx && dotStatus === 'not_started') {
        dotStatus = 'current';
      }
      const dotLabel = dotStatus.replace(/_/g, ' ');
      return `<div class="stage-dot ${dotStatus}" title="${stageName(s)}: ${dotLabel}"></div>`;
    }).join('');

    card.innerHTML = `
      <div class="wf-card-info">
        <div class="wf-card-name">${escapeHtml(wf.name)}</div>
        <div class="wf-card-trial">${escapeHtml(wf.trial_name)}</div>
        ${wf.description ? `<div class="wf-card-desc">${escapeHtml(wf.description)}</div>` : ''}
      </div>
      <div class="wf-card-stages" title="Stage progress">${dots}</div>
      <span class="status-badge ${wf.status}">${wf.status}</span>
    `;

    return card;
  }

  /* ══════════════════════════════════════════════════════
     Create Workflow
     ══════════════════════════════════════════════════════ */
  document.getElementById('create-workflow-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    setLoading(btn, true);

    const payload = {
      name: document.getElementById('wf-name').value.trim(),
      trial_name: document.getElementById('wf-trial').value.trim(),
      description: document.getElementById('wf-desc').value.trim() || null,
    };

    try {
      await api.post('/api/workflow', payload);
      showToast('Workflow created successfully!', 'success');
      e.target.reset();
      loadWorkflows();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoading(btn, false);
    }
  });

  /* ══════════════════════════════════════════════════════
     Workflow Detail Modal
     ══════════════════════════════════════════════════════ */
  window.openWorkflowDetail = async function (id) {
    const modal = document.getElementById('wf-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');
    const actions = document.getElementById('modal-actions');

    modal.style.display = '';
    title.textContent = 'Loading...';
    body.innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';
    actions.innerHTML = '';

    try {
      const data = await api.get(`/api/workflow/${id}`);
      const wf = data.workflow;

      // Fetch active-job status for each stage in parallel
      const stageJobStatuses = {};
      await Promise.all(STAGES.map(async (s) => {
        try {
          const job = await api.get(`/api/jobs/stage/${wf.id}/${s}`);
          if (job && job.job_id && (job.status === 'pending' || job.status === 'running')) {
            stageJobStatuses[s] = 'pending';
          }
        } catch (_) { /* no active job — ignore */ }
      }));

      title.textContent = wf.name;

      // Detail rows
      let html = `
        <div class="detail-row">
          <span class="detail-label">Trial</span>
          <span class="detail-value">${escapeHtml(wf.trial_name)}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Status</span>
          <span class="detail-value"><span class="status-badge ${wf.status}">${wf.status}</span></span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Current Stage</span>
          <span class="detail-value">${wf.current_stage ? stageName(wf.current_stage) : '—'}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Created</span>
          <span class="detail-value">${formatDate(wf.created_at)}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Updated</span>
          <span class="detail-value">${formatDate(wf.updated_at)}</span>
        </div>
      `;

      if (wf.description) {
        html += `
          <div class="detail-row">
            <span class="detail-label">Description</span>
            <span class="detail-value" style="text-align:right; max-width:60%;">${escapeHtml(wf.description)}</span>
          </div>
        `;
      }

      // Stage pipeline
      const currentStageIdx = wf.current_stage ? STAGES.indexOf(wf.current_stage) : -1;
      const workflowDone = wf.status === 'completed';

      html += '<h3 style="font-size:0.88rem;font-weight:600;color:var(--text-2);margin-top:24px;margin-bottom:12px;">WORKFLOW STAGES</h3>';
      html += '<div class="stage-pipeline">';
      STAGES.forEach((s, i) => {
        const stageData = wf.stages?.[s];
        const rawStatus = stageData?.status || 'not_started';
        const pageUrl = stagePageUrl(STAGE_PAGES[s], wf.id, wf.trial_name, wf.name);
        const hasResults = stageData?.output_data && Object.keys(stageData.output_data).length > 0;

        // Determine display status:
        //  - workflow completed → all stages show their raw status
        //  - stage is beyond current → locked
        //  - stage IS the current stage with not_started → "current"
        //  - otherwise → raw status
        const isFutureStage = !workflowDone && currentStageIdx >= 0 && i > currentStageIdx;
        const hasActiveJob = stageJobStatuses[s] === 'pending';
        let displayStatus;
        if (hasActiveJob) {
          displayStatus = 'pending';
        } else if (isFutureStage) {
          displayStatus = 'locked';
        } else if (!workflowDone && i === currentStageIdx && rawStatus === 'not_started') {
          displayStatus = 'current';
        } else {
          displayStatus = rawStatus;
        }

        const displayLabel = displayStatus.replace(/_/g, ' ');
        const canRerun = rawStatus === 'completed' || rawStatus === 'failed';
        // Can mark complete only on the current stage (or past incomplete stages), never on future/locked
        const canMarkComplete = !isFutureStage && !hasActiveJob && (rawStatus === 'not_started' || rawStatus === 'in_progress');

        html += `<div class="stage-card${isFutureStage ? ' stage-locked' : ''}">`;

        if (isFutureStage) {
          // Locked stage — greyed out, non-clickable
          html += `
            <div class="stage-item-row">
              <div class="stage-item locked stage-link" title="Complete earlier stages first" style="flex:1; opacity:0.45; pointer-events:none; cursor:not-allowed;">
                <div class="stage-num">${i + 1}</div>
                <div class="stage-info">
                  <div class="stage-name">${stageName(s)}</div>
                  <div class="stage-status-text">locked — complete earlier stages first</div>
                </div>
                <span class="status-badge locked" style="font-size:0.68rem;">locked</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              </div>
            </div>
          `;
        } else {
          // Accessible stage
          html += `
            <div class="stage-item-row">
              <a href="${pageUrl}" class="stage-item ${displayStatus} stage-link" title="Open ${stageName(s)}" style="flex:1;">
                <div class="stage-num">${i + 1}</div>
                <div class="stage-info">
                  <div class="stage-name">${stageName(s)}</div>
                  <div class="stage-status-text">${displayLabel}${stageData?.completed_at ? ' · ' + formatDate(stageData.completed_at) : ''}${stageData?.error ? ' · Error: ' + escapeHtml(stageData.error) : ''}</div>
                </div>
                <span class="status-badge ${displayStatus}" style="font-size:0.68rem;">${displayLabel}</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
              </a>
            </div>
          `;
        }

        // Stage action bar (Mark Complete / View Results / Rerun) — not for locked stages
        if (!isFutureStage && (hasResults || canRerun || canMarkComplete)) {
          html += '<div class="stage-actions-bar">';
          if (canMarkComplete) {
            html += `<button class="btn btn-text btn-xs stage-complete-btn" onclick="event.stopPropagation(); markStageComplete('${wf.id}', '${s}')" title="Mark this stage as complete" style="color:var(--green);">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              Mark Complete
            </button>`;
          }
          if (hasResults) {
            html += `<button class="btn btn-text btn-xs stage-view-results-btn" onclick="event.stopPropagation(); toggleStageResults('stage-results-${i}')" title="View stage results">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
              View Results
            </button>`;
          }
          if (canRerun) {
            html += `<button class="btn btn-text btn-xs stage-rerun-btn" onclick="event.stopPropagation(); rerunStage('${wf.id}', '${s}')" title="Re-run this stage">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
              Re-run
            </button>`;
          }
          html += '</div>';
        }

        // Collapsible results panel
        if (hasResults) {
          html += `<div id="stage-results-${i}" class="stage-results-panel" style="display:none;">`;
          html += `<div class="stage-results-header">Stage Output</div>`;
          html += `<pre class="stage-results-pre">${escapeHtml(JSON.stringify(stageData.output_data, null, 2))}</pre>`;
          html += '</div>';
        }

        html += '</div>'; // close .stage-card
      });
      html += '</div>';

      body.innerHTML = html;

      // Actions — compute button visibility based on workflow + current stage state
      const currentStageData = wf.current_stage ? wf.stages?.[wf.current_stage] : null;
      const currentStageStatus = currentStageData?.status || 'not_started';
      const canResume = wf.status === 'paused' || wf.status === 'failed';
      const canAdvance = (wf.status === 'running' || wf.status === 'paused')
                         && currentStageStatus === 'completed';
      const canPause = wf.status === 'running';

      actions.innerHTML = '';

      if (canResume) {
        actions.innerHTML += `<button class="btn btn-primary btn-sm" onclick="workflowAction('${wf.id}', 'resume')">Resume</button>`;
      }
      if (canAdvance) {
        const isLastStage = currentStageIdx === STAGES.length - 1;
        const advanceLabel = isLastStage ? 'Complete Workflow' : 'Advance Stage';
        actions.innerHTML += `<button class="btn btn-${isLastStage ? 'primary' : 'secondary'} btn-sm" onclick="workflowAction('${wf.id}', 'advance')">${advanceLabel}</button>`;
      }
      if (canPause) {
        actions.innerHTML += `<button class="btn btn-secondary btn-sm" onclick="workflowAction('${wf.id}', 'pause')">Pause</button>`;
      }

      actions.innerHTML += `<button class="btn btn-text btn-sm" style="color:var(--red);" onclick="deleteWorkflow('${wf.id}')">Delete</button>`;
      actions.innerHTML += `<button class="btn btn-secondary btn-sm" onclick="closeModal()" style="margin-left:auto;">Close</button>`;

    } catch (err) {
      body.innerHTML = `<p style="color:var(--red);">${escapeHtml(err.message)}</p>`;
      actions.innerHTML = '<button class="btn btn-secondary btn-sm" onclick="closeModal()">Close</button>';
    }
  };

  window.closeModal = function () {
    document.getElementById('wf-modal').style.display = 'none';
  };

  /* ══════════════════════════════════════════════════════
     Workflow Actions
     ══════════════════════════════════════════════════════ */
  window.workflowAction = async function (id, action) {
    try {
      const data = await api.post(`/api/workflow/${id}/${action}`);
      showToast(data.message || `${action} successful`, 'success');
      // Refresh both modal and list, awaiting modal so user sees update
      await openWorkflowDetail(id);
      loadWorkflows();
    } catch (err) {
      showToast(err.message, 'error');
      // Still refresh modal on error so state is accurate
      try { await openWorkflowDetail(id); } catch (_) {}
    }
  };

  /* ── Toggle stage results visibility ── */
  window.toggleStageResults = function (panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : '';
    // Toggle button text
    const btn = panel.closest('.stage-card')?.querySelector('.stage-view-results-btn');
    if (btn) {
      const label = btn.querySelector('.stage-view-results-label') || btn.lastChild;
      if (isVisible) {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> View Results`;
      } else {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg> Hide Results`;
      }
    }
  };

  /* ── Mark a stage as complete ── */
  window.markStageComplete = async function (workflowId, stage) {
    if (!confirm(`Mark "${stageName(stage)}" as complete?`)) return;
    try {
      await api.put(`/api/workflow/${workflowId}/stage/${stage}`, {
        status: 'completed',
      });
      showToast(`${stageName(stage)} marked as complete!`, 'success');
      await openWorkflowDetail(workflowId);
      loadWorkflows();
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  /* ── Re-run a specific stage ── */
  window.rerunStage = async function (workflowId, stage) {
    if (!confirm(`Reset the "${stageName(stage)}" stage and navigate to re-run it?`)) return;
    try {
      // Reset the stage status via the update endpoint
      await api.put(`/api/workflow/${workflowId}/stage/${stage}`, {
        status: 'not_started',
      });
      showToast(`${stageName(stage)} reset. Navigating to stage page...`, 'info');
      // Navigate to the dedicated stage page
      const data = await api.get(`/api/workflow/${workflowId}`);
      const wf = data.workflow;
      const pageUrl = stagePageUrl(STAGE_PAGES[stage], wf.id, wf.trial_name, wf.name);
      window.location.href = pageUrl;
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  window.deleteWorkflow = async function (id) {
    if (!confirm('Delete this workflow? This cannot be undone.')) return;
    try {
      await api.del(`/api/workflow/${id}`);
      showToast('Workflow deleted.', 'success');
      closeModal();
      loadWorkflows();
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  /* ══════════════════════════════════════════════════════
     Init
     ══════════════════════════════════════════════════════ */
  loadWorkflows();
})();
