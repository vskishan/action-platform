/* monitoring.js - Monitoring page logic */
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
        `/api/workflow/${wfCtx.workflowId}/stage/cohort_monitoring/conversation`,
        { messages: conversationHistory }
      );
    } catch (err) {
      console.warn('Failed to save monitoring conversation:', err);
    }
  }

  /* Load conversation from backend */
  async function loadConversation() {
    if (!wfCtx.workflowId) return;
    try {
      const data = await api.get(
        `/api/workflow/${wfCtx.workflowId}/stage/cohort_monitoring/conversation`
      );
      if (data.messages && data.messages.length > 0) {
        conversationHistory = data.messages;
        // Re-render the last monitoring result if present
        const lastResult = conversationHistory.filter(m => m.role === 'result').pop();
        if (lastResult && lastResult.data) {
          renderMonitoringResults(lastResult.data);
        }
      }
    } catch (err) {
      console.warn('Failed to load monitoring conversation:', err);
    }
  }

  /* Set query from quick pills */
  window.setQuery = function (text) {
    document.getElementById('mon-query').value = text;
  };

  /* Submit Monitoring Query */
  document.getElementById('monitoring-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('mon-submit-btn');

    const payload = {
      trial_name: document.getElementById('mon-trial').value.trim(),
      query: document.getElementById('mon-query').value.trim(),
      use_extraction: false,
    };

    // Save a pending entry so we know a query is in-flight
    conversationHistory.push({ role: 'user', text: payload.query, timestamp: new Date().toISOString() });
    conversationHistory.push({ role: 'pending', text: 'Monitoring query running…', timestamp: new Date().toISOString() });
    saveConversation();

    setLoading(btn, true);

    submitJob({
      workflowId: wfCtx.workflowId,
      stage: 'cohort_monitoring',
      payload,
      description: `Monitoring query: ${payload.query}`,
      onComplete: (result) => {
        setLoading(btn, false);
        // Remove pending entry
        const idx = conversationHistory.findIndex(m => m.role === 'pending');
        if (idx >= 0) conversationHistory.splice(idx, 1);
        renderMonitoringResults(result);
        showToast('Monitoring query completed!', 'success');
        conversationHistory.push({ role: 'result', text: 'Monitoring completed', data: result, timestamp: new Date().toISOString() });
        saveConversation();
      },
      onError: (errMsg) => {
        setLoading(btn, false);
        const idx = conversationHistory.findIndex(m => m.role === 'pending');
        if (idx >= 0) conversationHistory.splice(idx, 1);
        showToast(errMsg, 'error');
        saveConversation();
      },
    });
  });

  /* Render Monitoring Results */
  function renderMonitoringResults(data) {
    const panel = document.getElementById('mon-results-panel');
    const subtitle = document.getElementById('mon-results-subtitle');
    const aiEl = document.getElementById('mon-ai-response');
    const statsEl = document.getElementById('mon-stats');
    const sitesEl = document.getElementById('mon-site-results');

    panel.style.display = '';
    subtitle.textContent = `Trial: ${data.trial_name} · Query Type: ${(data.query_type || '').replace(/_/g, ' ')} · Status: ${data.status}`;

    // AI response
    if (data.response) {
      aiEl.textContent = data.response;
    } else {
      aiEl.textContent = '';
    }

    // Global aggregate stats
    const gd = data.global_result || {};
    let statsHtml = '';

    // Try to pull out interesting top-level numbers
    const totalPatients = sumSiteField(data.site_results, 'total_patients_monitored');
    statsHtml += `
      <div class="result-stat-card accent-blue">
        <div class="stat-number">${data.site_results?.length || 0}</div>
        <div class="stat-label">Sites Reporting</div>
      </div>
      <div class="result-stat-card accent-green">
        <div class="stat-number">${totalPatients}</div>
        <div class="stat-label">Patients Monitored</div>
      </div>
    `;

    // Display up to 3 aggregate numbers from global_result
    const displayKeys = Object.keys(gd).slice(0, 3);
    const accentCycle = ['accent-yellow', 'accent-red', 'accent-blue'];
    displayKeys.forEach((key, i) => {
      const val = gd[key];
      if (typeof val === 'number' || typeof val === 'string') {
        statsHtml += `
          <div class="result-stat-card ${accentCycle[i % accentCycle.length]}">
            <div class="stat-number">${typeof val === 'number' ? val.toLocaleString() : escapeHtml(String(val))}</div>
            <div class="stat-label">${escapeHtml(key.replace(/_/g, ' '))}</div>
          </div>
        `;
      }
    });

    statsEl.innerHTML = statsHtml;

    // Per-site cards
    const colors = ['blue', 'green', 'red', 'yellow'];
    sitesEl.innerHTML = '';

    (data.site_results || []).forEach((site, i) => {
      const color = colors[i % colors.length];
      let metricsHtml = `
        <div class="site-metric-row">
          <span class="site-metric-label">Patients Monitored</span>
          <span class="site-metric-value">${site.total_patients_monitored}</span>
        </div>
        <div class="site-metric-row">
          <span class="site-metric-label">Query Type</span>
          <span class="site-metric-value">${(site.query_type || '').replace(/_/g, ' ')}</span>
        </div>
      `;

      // Result data entries
      if (site.result_data && typeof site.result_data === 'object') {
        for (const [key, val] of Object.entries(site.result_data)) {
          if (typeof val === 'object' && val !== null) {
            // Sub-object: render as nested rows
            metricsHtml += `<div style="margin-top:12px;"><strong style="font-size:0.78rem;color:var(--text-2);text-transform:uppercase;letter-spacing:0.08em;">${escapeHtml(key.replace(/_/g, ' '))}</strong></div>`;
            for (const [sk, sv] of Object.entries(val)) {
              metricsHtml += `
                <div class="site-metric-row">
                  <span class="site-metric-label">${escapeHtml(sk)}</span>
                  <span class="site-metric-value">${escapeHtml(String(sv))}</span>
                </div>
              `;
            }
          } else {
            metricsHtml += `
              <div class="site-metric-row">
                <span class="site-metric-label">${escapeHtml(key.replace(/_/g, ' '))}</span>
                <span class="site-metric-value">${escapeHtml(String(val))}</span>
              </div>
            `;
          }
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
            ${site.data_as_of ? `<p>Data as of: ${escapeHtml(site.data_as_of)}</p>` : ''}
          </div>
        </div>
        ${metricsHtml}
      `;
      sitesEl.appendChild(card);
    });

    // Scroll results into view
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function sumSiteField(sites, field) {
    if (!sites) return 0;
    return sites.reduce((sum, s) => sum + (s[field] || 0), 0);
  }

  /* Init - load conversation & check for pending jobs */
  async function init() {
    await loadConversation();

    if (wfCtx.workflowId) {
      await checkForPendingJob({
        workflowId: wfCtx.workflowId,
        stage: 'cohort_monitoring',
        onComplete: (result) => {
          const idx = conversationHistory.findIndex(m => m.role === 'pending');
          if (idx >= 0) conversationHistory.splice(idx, 1);
          renderMonitoringResults(result);
          conversationHistory.push({ role: 'result', text: 'Monitoring completed', data: result, timestamp: new Date().toISOString() });
          saveConversation();
          showToast('Monitoring query completed!', 'success');
        },
        onError: (errMsg) => {
          const idx = conversationHistory.findIndex(m => m.role === 'pending');
          if (idx >= 0) conversationHistory.splice(idx, 1);
          showToast(errMsg, 'error');
          saveConversation();
        },
      });
    }
  }

  init();
})();
