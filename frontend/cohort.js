/* cohort.js - Cohort Analytics page logic */
(() => {
  'use strict';

  // Require workflow context – redirects to /workflows.html if missing
  const wfCtx = requireWorkflowContext();
  if (!wfCtx) return;

  const conversationEl = document.getElementById('cohort-conversation');

  // In-memory conversation history for persistence
  let conversationHistory = [];

  // Session ID for multi-turn agentic conversation (ReAct agent memory)
  let agentSessionId = null;

  /* Save conversation to backend */
  async function saveConversation() {
    if (!wfCtx.workflowId) return;
    try {
      await api.put(
        `/api/workflow/${wfCtx.workflowId}/stage/cohort_formation/conversation`,
        { messages: conversationHistory }
      );
    } catch (err) {
      console.warn('Failed to save conversation:', err);
    }
  }

  /* Load conversation from backend */
  async function loadConversation() {
    if (!wfCtx.workflowId) return;
    try {
      const data = await api.get(
        `/api/workflow/${wfCtx.workflowId}/stage/cohort_formation/conversation`
      );
      if (data.messages && data.messages.length > 0) {
        conversationHistory = data.messages;
        // Show the results panel and render saved messages
        const panel = document.getElementById('cohort-results-panel');
        panel.style.display = '';
        document.getElementById('cohort-results-subtitle').textContent = 'Conversation with MedGemma';
        conversationHistory.forEach(msg => addBubble(msg.text, msg.role, false));
        conversationEl.scrollTo({ top: conversationEl.scrollHeight, behavior: 'auto' });
      }
    } catch (err) {
      console.warn('Failed to load conversation:', err);
    }
  }

  /* Set query from quick pills */
  window.setCohortQuery = function (text) {
    document.getElementById('cohort-query').value = text;
  };

  /* Submit Cohort Query */
  document.getElementById('cohort-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('cohort-submit-btn');
    const queryText = document.getElementById('cohort-query').value.trim();
    if (!queryText) return;

    setLoading(btn, true);

    // Show user bubble immediately
    const panel = document.getElementById('cohort-results-panel');
    panel.style.display = '';
    document.getElementById('cohort-results-subtitle').textContent = 'Conversation with MedGemma';

    addBubble(queryText, 'user', true);

    // Show typing indicator
    const typingId = addTypingIndicator();

    // Save pending state so it persists across navigation
    conversationHistory.push({ role: 'pending', text: 'Waiting for response…', timestamp: new Date().toISOString() });
    saveConversation();

    // Clear the input
    document.getElementById('cohort-query').value = '';

    try {
      await submitJob({
        workflowId: wfCtx.workflowId,
        stage: 'cohort_formation',
        payload: { query: queryText, session_id: agentSessionId },
        description: 'Running cohort analytics query…',
        onComplete: (result) => {
          removeTypingIndicator(typingId);
          // Track session_id for multi-turn conversations
          if (result.session_id) agentSessionId = result.session_id;
          const responseText = result.response || JSON.stringify(result);
          addBubble(responseText, 'ai', false);
          // Show agentic info (tools used, steps)
          if (result.tools_used && result.tools_used.length > 0) {
            const agentInfo = `🔧 Agent used ${result.steps || 1} step(s) and called: ${result.tools_used.join(', ')}`;
            addAgentInfoBubble(agentInfo);
          }
          // Replace pending entry with actual result
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'ai', text: responseText, timestamp: new Date().toISOString() });
          saveConversation();
          conversationEl.scrollTo({ top: conversationEl.scrollHeight, behavior: 'smooth' });
          setLoading(btn, false);
        },
        onError: (errMsg) => {
          removeTypingIndicator(typingId);
          addBubble(`Error: ${errMsg}`, 'ai', false);
          showToast(errMsg, 'error');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          conversationHistory.push({ role: 'ai', text: `Error: ${errMsg}`, timestamp: new Date().toISOString() });
          saveConversation();
          setLoading(btn, false);
        },
      });
    } catch (err) {
      removeTypingIndicator(typingId);
      addBubble(`Error: ${err.message}`, 'ai', true);
      showToast(err.message, 'error');
      saveConversation();
      setLoading(btn, false);
    }
  });

  /* Chat Helpers */
  function addBubble(text, role, persist = false) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    conversationEl.appendChild(bubble);
    // Scroll into view
    bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
    // Track in history
    if (persist) {
      conversationHistory.push({
        role,
        text,
        timestamp: new Date().toISOString(),
      });
    }
  }

  function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const el = document.createElement('div');
    el.className = 'chat-bubble ai';
    el.id = id;
    el.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;margin:0;"></div>';
    conversationEl.appendChild(el);
    el.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return id;
  }

  function addAgentInfoBubble(text) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble agent-info';
    bubble.textContent = text;
    bubble.style.cssText = 'font-size:0.8em;opacity:0.7;font-style:italic;padding:4px 12px;';
    conversationEl.appendChild(bubble);
    bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  /* Init - load conversation & check for pending jobs */
  async function init() {
    await loadConversation();

    if (wfCtx.workflowId) {
      await checkForPendingJob({
        workflowId: wfCtx.workflowId,
        stage: 'cohort_formation',
        onComplete: (result) => {
          const responseText = result.response || JSON.stringify(result);
          // Remove pending entries and add result
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) conversationHistory.splice(pendingIdx, 1);
          const lastAi = conversationHistory.filter(m => m.role === 'ai').pop();
          if (!lastAi || lastAi.text !== responseText) {
            addBubble(responseText, 'ai', false);
            conversationHistory.push({ role: 'ai', text: responseText, timestamp: new Date().toISOString() });
            saveConversation();
          }
          showToast('Cohort query completed!', 'success');
        },
        onError: (errMsg) => {
          showToast(errMsg, 'error');
          const pendingIdx = conversationHistory.findIndex(m => m.role === 'pending');
          if (pendingIdx >= 0) {
            conversationHistory.splice(pendingIdx, 1);
            addBubble(`Error: ${errMsg}`, 'ai', false);
            conversationHistory.push({ role: 'ai', text: `Error: ${errMsg}`, timestamp: new Date().toISOString() });
            saveConversation();
          }
        },
      });
    }
  }

  init();
})();
