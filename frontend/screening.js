/* screening.js - Patient Screening page logic */
(() => {
  'use strict';

  // Require workflow context – redirects to /workflows.html if missing
  const wfCtx = requireWorkflowContext();
  if (!wfCtx) return;

  // In-memory conversation history for persistence
  let conversationHistory = [];

  /* Save conversation to backend */
  async function saveConversation() {
    if (!wfCtx.workflowId) return;
    try {
      await api.put(
        `/api/workflow/${wfCtx.workflowId}/stage/patient_screening/conversation`,
        { messages: conversationHistory }
      );
    } catch (err) {
      console.warn('Failed to save screening conversation:', err);
    }
  }

  /* Load conversation from backend */
  async function loadConversation() {
    if (!wfCtx.workflowId) return;
    try {
      const data = await api.get(
        `/api/workflow/${wfCtx.workflowId}/stage/patient_screening/conversation`
      );
      if (data.messages && data.messages.length > 0) {
        conversationHistory = data.messages;
        // Re-render the last screening result if present
        const lastResult = conversationHistory.filter(m => m.role === 'result').pop();
        if (lastResult && lastResult.data) {
          renderScreeningResults(lastResult.data);
        }
      }
    } catch (err) {
      console.warn('Failed to load screening conversation:', err);
    }
  }

  /* Natural Language Screening */
  document.getElementById('nl-screening-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('nl-submit-btn');
    setLoading(btn, true);

    const payload = {
      trial_name: document.getElementById('nl-trial').value.trim(),
      inclusion: [],
      exclusion: [],
      natural_language_criteria: document.getElementById('nl-criteria').value.trim(),
    };

    const userText = payload.natural_language_criteria || JSON.stringify(payload);

    // Save the query immediately so it's preserved even if user navigates away
    conversationHistory.push({ role: 'user', text: userText, timestamp: new Date().toISOString() });
    conversationHistory.push({ role: 'pending', text: 'Screening in progress…', timestamp: new Date().toISOString() });
    saveConversation();

    try {
      await submitJob({
        workflowId: wfCtx.workflowId,
        stage: 'patient_screening',
        payload,
        description: 'Running federated patient screening…',
        onComplete: (result) => {
          renderScreeningResults(result);
          showToast('Screening completed!', 'success');
          // Replace the pending entry with actual result
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'result', text: 'Screening completed', data: result, timestamp: new Date().toISOString() });
          saveConversation();
          setLoading(btn, false);
        },
        onError: (errMsg) => {
          showToast(errMsg, 'error');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'error', text: errMsg, timestamp: new Date().toISOString() });
          saveConversation();
          setLoading(btn, false);
        },
      });
    } catch (err) {
      showToast(err.message, 'error');
      setLoading(btn, false);
    }
  });

  /* Structured Screening */
  document.getElementById('structured-screening-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('struct-submit-btn');
    setLoading(btn, true);

    const inclusion = parseCriteria('inclusion-criteria');
    const exclusion = parseCriteria('exclusion-criteria');

    const payload = {
      trial_name: document.getElementById('struct-trial').value.trim(),
      inclusion,
      exclusion,
    };

    const userText = `Structured screening: ${inclusion.length} inclusion, ${exclusion.length} exclusion criteria`;

    conversationHistory.push({ role: 'user', text: userText, timestamp: new Date().toISOString() });
    conversationHistory.push({ role: 'pending', text: 'Screening in progress…', timestamp: new Date().toISOString() });
    saveConversation();

    try {
      await submitJob({
        workflowId: wfCtx.workflowId,
        stage: 'patient_screening',
        payload,
        description: 'Running structured patient screening…',
        onComplete: (result) => {
          renderScreeningResults(result);
          showToast('Screening completed!', 'success');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'result', text: 'Screening completed', data: result, timestamp: new Date().toISOString() });
          saveConversation();
          setLoading(btn, false);
        },
        onError: (errMsg) => {
          showToast(errMsg, 'error');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'error', text: errMsg, timestamp: new Date().toISOString() });
          saveConversation();
          setLoading(btn, false);
        },
      });
    } catch (err) {
      showToast(err.message, 'error');
      setLoading(btn, false);
    }
  });

  function parseCriteria(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return [];
    const rows = container.querySelectorAll('.criterion-row');
    const criteria = [];
    rows.forEach(row => {
      const category = row.querySelector('.criterion-category')?.value;
      const field = row.querySelector('.criterion-field')?.value?.trim();
      const operator = row.querySelector('.criterion-operator')?.value;
      const rawValue = row.querySelector('.criterion-value')?.value?.trim();
      if (!field || !rawValue) return;

      // Try to parse as number, else keep string, else try as list
      let value = rawValue;
      if (!isNaN(rawValue) && rawValue !== '') {
        value = Number(rawValue);
      } else if (rawValue.includes(',')) {
        value = rawValue.split(',').map(v => v.trim());
      }

      criteria.push({ category, field, operator, value });
    });
    return criteria;
  }

  /* Add / Remove Criteria Rows */
  window.addCriterion = function (type) {
    const container = document.getElementById(`${type}-criteria`);
    if (!container) return;
    const row = document.createElement('div');
    row.className = 'criterion-row';
    row.innerHTML = `
      <select class="criterion-category">
        <option value="demographic">Demographic</option>
        <option value="condition">Condition</option>
        <option value="lab">Lab</option>
        <option value="medication">Medication</option>
      </select>
      <input type="text" class="criterion-field" placeholder="Field (e.g. age)" />
      <select class="criterion-operator">
        <option value="gte">≥</option>
        <option value="lte">≤</option>
        <option value="eq">=</option>
        <option value="gt">></option>
        <option value="lt"><</option>
        <option value="neq">≠</option>
        <option value="in">in</option>
      </select>
      <input type="text" class="criterion-value" placeholder="Value" />
      <button type="button" class="btn-icon-sm" onclick="removeCriterion(this)" title="Remove">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;
    container.appendChild(row);
  };

  window.removeCriterion = function (btn) {
    const row = btn.closest('.criterion-row');
    if (row) row.remove();
  };

  /* Render Screening Results */
  function renderScreeningResults(data) {
    const panel = document.getElementById('results-panel');
    const statsEl = document.getElementById('result-stats');
    const sitesEl = document.getElementById('site-results');
    const subtitle = document.getElementById('results-subtitle');

    panel.style.display = '';
    subtitle.textContent = `Trial: ${data.trial_name} · Status: ${data.status}`;

    // Aggregate stats
    const eligRate = data.aggregate_total_patients > 0
      ? ((data.aggregate_eligible_patients / data.aggregate_total_patients) * 100).toFixed(1)
      : 0;

    let statsHtml = `
      <div class="result-stat-card accent-blue">
        <div class="stat-number">${data.site_results?.length || 0}</div>
        <div class="stat-label">Sites Queried</div>
      </div>
      <div class="result-stat-card accent-green">
        <div class="stat-number">${data.aggregate_total_patients}</div>
        <div class="stat-label">Total Patients</div>
      </div>
      <div class="result-stat-card accent-green">
        <div class="stat-number">${data.aggregate_eligible_patients}</div>
        <div class="stat-label">Eligible Patients</div>
      </div>
      <div class="result-stat-card accent-yellow">
        <div class="stat-number">${eligRate}%</div>
        <div class="stat-label">Eligibility Rate</div>
      </div>
    `;

    // Self-correcting screening audit stats (if available)
    const hasCorrected = (data.aggregate_corrected_count || 0) > 0;
    const hasFlagged   = (data.aggregate_flagged_for_review || 0) > 0;
    const hasHighConf  = (data.aggregate_high_confidence || 0) > 0;
    const hasLowConf   = (data.aggregate_low_confidence || 0) > 0;
    const hasAuditData = hasCorrected || hasFlagged || hasHighConf || hasLowConf;

    if (hasAuditData) {
      statsHtml += `
        <div class="result-stat-card accent-green">
          <div class="stat-number">${data.aggregate_high_confidence || 0}</div>
          <div class="stat-label">High Confidence</div>
        </div>
        <div class="result-stat-card accent-yellow">
          <div class="stat-number">${data.aggregate_low_confidence || 0}</div>
          <div class="stat-label">Low Confidence</div>
        </div>
        <div class="result-stat-card accent-blue">
          <div class="stat-number">${data.aggregate_corrected_count || 0}</div>
          <div class="stat-label">Self-corrected</div>
        </div>
        <div class="result-stat-card accent-red">
          <div class="stat-number">${data.aggregate_flagged_for_review || 0}</div>
          <div class="stat-label">Flagged for Review</div>
        </div>
      `;
    }

    statsEl.innerHTML = statsHtml;

    // Per-site cards
    const colors = ['blue', 'green', 'red', 'yellow'];
    sitesEl.innerHTML = '';

    (data.site_results || []).forEach((site, i) => {
      const color = colors[i % colors.length];
      const siteEligRate = site.total_patients > 0
        ? ((site.eligible_patients / site.total_patients) * 100).toFixed(1)
        : 0;

      let metricsHtml = `
        <div class="site-metric-row">
          <span class="site-metric-label">Total Patients</span>
          <span class="site-metric-value">${site.total_patients}</span>
        </div>
        <div class="site-metric-row">
          <span class="site-metric-label">Eligible</span>
          <span class="site-metric-value">${site.eligible_patients}</span>
        </div>
        <div class="site-metric-row">
          <span class="site-metric-label">Eligibility Rate</span>
          <span class="site-metric-value">${siteEligRate}%</span>
        </div>
      `;

      // Self-correcting screening audit metrics for this site
      const siteHasAudit = (site.high_confidence_count || 0) + (site.medium_confidence_count || 0) + (site.low_confidence_count || 0) > 0;
      if (siteHasAudit) {
        metricsHtml += '<div style="margin-top:12px;"><strong style="font-size:0.78rem;color:var(--text-2);text-transform:uppercase;letter-spacing:0.08em;">Audit & Confidence</strong></div>';
        metricsHtml += `
          <div class="site-metric-row">
            <span class="site-metric-label">High Confidence</span>
            <span class="site-metric-value" style="color:var(--green);">${site.high_confidence_count || 0}</span>
          </div>
          <div class="site-metric-row">
            <span class="site-metric-label">Medium Confidence</span>
            <span class="site-metric-value">${site.medium_confidence_count || 0}</span>
          </div>
          <div class="site-metric-row">
            <span class="site-metric-label">Low Confidence</span>
            <span class="site-metric-value" style="color:var(--yellow);">${site.low_confidence_count || 0}</span>
          </div>
          <div class="site-metric-row">
            <span class="site-metric-label">Self-corrected</span>
            <span class="site-metric-value" style="color:var(--blue);">${site.corrected_count || 0}</span>
          </div>
          <div class="site-metric-row">
            <span class="site-metric-label">Flagged for Review</span>
            <span class="site-metric-value" style="color:var(--red);">${site.flagged_for_review_count || 0}</span>
          </div>
        `;
      }

      // Per-patient audit details (collapsible)
      if (site.patient_audit_details?.length) {
        metricsHtml += `<div style="margin-top:12px;">
          <button class="btn btn-text btn-xs" onclick="togglePatientAudit(this, 'audit-${i}')" style="padding:0;font-size:0.78rem;color:var(--text-2);">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            Patient Audit Details (${site.patient_audit_details.length})
          </button>
        </div>
        <div id="audit-${i}" style="display:none; margin-top:8px;">`;

        site.patient_audit_details.forEach(pd => {
          const confColor = pd.confidence === 'high' ? 'var(--green)' : pd.confidence === 'low' ? 'var(--yellow)' : 'var(--text-2)';
          const correctedTag = pd.was_corrected ? ' <span style="color:var(--blue);font-weight:600;font-size:0.72rem;">CORRECTED</span>' : '';
          const flaggedTag = pd.flagged_for_review ? ' <span style="color:var(--red);font-weight:600;font-size:0.72rem;">FLAGGED</span>' : '';

          metricsHtml += `
            <div style="padding:6px 8px; border-radius:6px; background:var(--bg-2); margin-bottom:4px; font-size:0.8rem;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>${escapeHtml(pd.patient_id)}</strong>
                <span style="color:${confColor}; font-weight:600; font-size:0.72rem; text-transform:uppercase;">${pd.confidence}</span>
              </div>
              <div style="color:var(--text-3); font-size:0.75rem; margin-top:2px;">
                ${escapeHtml(pd.final_decision)}: ${escapeHtml(pd.final_reason)}
                ${pd.screening_passes > 1 ? `<span style="color:var(--text-3);"> (${pd.screening_passes} passes)</span>` : ''}
                ${correctedTag}${flaggedTag}
              </div>
              ${pd.audit_issues?.length ? `<div style="color:var(--yellow); font-size:0.72rem; margin-top:2px;">Issues: ${pd.audit_issues.map(i => escapeHtml(i)).join(', ')}</div>` : ''}
            </div>
          `;
        });

        metricsHtml += '</div>';
      }

      // Inclusion pass counts
      if (site.inclusion_pass_counts && Object.keys(site.inclusion_pass_counts).length) {
        metricsHtml += '<div style="margin-top:12px;"><strong style="font-size:0.78rem;color:var(--text-2);text-transform:uppercase;letter-spacing:0.08em;">Inclusion Pass Counts</strong></div>';
        for (const [key, val] of Object.entries(site.inclusion_pass_counts)) {
          metricsHtml += `
            <div class="site-metric-row">
              <span class="site-metric-label">${escapeHtml(key)}</span>
              <span class="site-metric-value">${val}</span>
            </div>
          `;
        }
      }

      // Errors
      if (site.errors?.length) {
        metricsHtml += '<div style="margin-top:12px;color:var(--red);font-size:0.82rem;">';
        site.errors.forEach(e => { metricsHtml += `<div>⚠ ${escapeHtml(e)}</div>`; });
        metricsHtml += '</div>';
      }

      const card = document.createElement('div');
      card.className = 'site-card';
      card.innerHTML = `
        <div class="site-card-header">
          <div class="site-badge ${color}">${site.site_id.slice(-2).toUpperCase()}</div>
          <div>
            <h3>${escapeHtml(site.site_id)}</h3>
          </div>
        </div>
        ${metricsHtml}
      `;
      sitesEl.appendChild(card);
    });

    // Scroll into view
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

    if (data.message) {
      showToast(data.message, 'info');
    }
  }

  /* Toggle patient audit details */
  window.togglePatientAudit = function (btn, panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : '';
    const svg = btn.querySelector('svg');
    if (svg) {
      svg.style.transform = isVisible ? '' : 'rotate(180deg)';
    }
  };

  /* Init - load conversation & check for pending jobs */
  async function init() {
    await loadConversation();

    // Check if there's a background job still running from a previous visit
    if (wfCtx.workflowId) {
      const handled = await checkForPendingJob({
        workflowId: wfCtx.workflowId,
        stage: 'patient_screening',
        onComplete: (result) => {
          renderScreeningResults(result);
          showToast('Screening completed!', 'success');
          // Remove any stale pending entry and save result
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          // Only add result if we don't already have it (avoid duplicates)
          const lastResult = conversationHistory.filter(m => m.role === 'result').pop();
          if (!lastResult || JSON.stringify(lastResult.data) !== JSON.stringify(result)) {
            conversationHistory.push({ role: 'result', text: 'Screening completed', data: result, timestamp: new Date().toISOString() });
            saveConversation();
          }
        },
        onError: (errMsg) => {
          showToast(errMsg, 'error');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) {
            conversationHistory.splice(pendingIdx, 1);
            conversationHistory.push({ role: 'error', text: errMsg, timestamp: new Date().toISOString() });
            saveConversation();
          }
        },
      });
    }
  }

  init();
})();
