const socket = io("http://localhost:8002");

// --- GLOBAL STATE for Drawers ---
window.lastDirectorState = null;
window.graphLog = [];
window.scoreLabels = [];
window.scoreData = [];
window.interestChart = null;

// DOM Elements (Static ones only)
const moodPill = document.getElementById('mood-pill');
const moodText = document.getElementById('mood-text');
const statePill = document.getElementById('state-pill');
const stateText = document.getElementById('state-text');
const summaryEl = document.getElementById('summary-text');
const summaryContextEl = document.getElementById('summary-raw-context');
const predictionEl = document.getElementById('prediction-text');

// Adaptive Metrics Elements
const adaptiveStateLabel = document.getElementById('adaptive-state-label');
const thresholdBar = document.getElementById('threshold-bar');
const thresholdVal = document.getElementById('threshold-value');
const velocityBar = document.getElementById('velocity-bar');
const velocityVal = document.getElementById('velocity-value');
const energyBar = document.getElementById('energy-bar');
const energyVal = document.getElementById('energy-value');
const batteryBar = document.getElementById('battery-bar');
const batteryVal = document.getElementById('battery-value');

// Dynamics Elements
const flowText = document.getElementById('flow-text');
const intentText = document.getElementById('intent-text');

// Directive Elements
const dirObjectiveEl = document.getElementById('dir-objective');
const dirToneEl = document.getElementById('dir-tone');
const dirActionEl = document.getElementById('dir-action');
const dirConstraintsBox = document.getElementById('dir-constraints-box');
const dirConstraintsEl = document.getElementById('dir-constraints');

// Context Control Elements
const streamerSelect = document.getElementById('streamer-select');
const contextInput = document.getElementById('context-input');
const contextCharCount = document.getElementById('context-char-count');
const streamerLockBtn = document.getElementById('streamer-lock-btn');
const contextLockBtn = document.getElementById('context-lock-btn');
const aiSuggestionIndicator = document.getElementById('ai-suggestion-indicator');
const aiSuggestionText = document.getElementById('ai-suggestion-text');
const acceptAiSuggestionBtn = document.getElementById('accept-ai-suggestion');

// Lock States
let streamerLocked = false;
let contextLocked = false;
let pendingAiContext = null;

// Context Logs
const visionLog = [];
const spokenLog = [];
const audioLog = [];

// Chart.js Setup
const CHART_HISTORY_SIZE = 50;

// Exposed function for the Interest Graph drawer
window.initializeChart = function() {
    const ctxEl = document.getElementById('interest-chart');
    if (!ctxEl) return;
    if (typeof Chart === 'undefined') return;
    
    const ctx = ctxEl.getContext('2d');
    
    // Destroy existing to prevent memory leaks or double-rendering
    if (window.interestChart) {
        window.interestChart.destroy();
    }
    
    window.interestChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: window.scoreLabels,
            datasets: [{
                label: 'Interest Score', 
                data: window.scoreData, 
                borderColor: '#63e2b7', 
                backgroundColor: 'rgba(99, 226, 183, 0.2)', 
                tension: 0.3, 
                pointRadius: 2, 
                fill: true
            }]
        },
        options: {
            responsive: true, 
            maintainAspectRatio: false, 
            animation: { duration: 0 },
            scales: { 
                y: { min: 0.0, max: 1.0, ticks: { color: '#e0e0e0' }, grid: { color: '#444' } }, 
                x: { display: false } 
            },
            plugins: { legend: { display: false } }
        }
    });
};

// --- Lock Button Functions ---
window.toggleStreamerLock = function() {
    streamerLocked = !streamerLocked;
    updateLockButtonUI(streamerLockBtn, streamerLocked);
    socket.emit('set_streamer_lock', { locked: streamerLocked });
    console.log('[Director] Streamer lock:', streamerLocked);
};

window.toggleContextLock = function() {
    contextLocked = !contextLocked;
    updateLockButtonUI(contextLockBtn, contextLocked);
    socket.emit('set_context_lock', { locked: contextLocked });
    console.log('[Director] Context lock:', contextLocked);
};

function updateLockButtonUI(btn, isLocked) {
    if (!btn) return;
    btn.textContent = isLocked ? '🔒' : '🔓';
    btn.classList.toggle('locked', isLocked);
}

// --- Context Input Character Counter ---
if (contextInput) {
    contextInput.addEventListener('input', () => {
        const len = contextInput.value.length;
        if (contextCharCount) {
            contextCharCount.textContent = `${len}/120`;
            contextCharCount.classList.toggle('text-red-400', len >= 120);
        }
    });
}

// --- AI Context Suggestion Handling ---
socket.on('ai_context_suggestion', (data) => {
    console.log('[AI Context] Received suggestion:', data);
    
    if (data.context && !contextLocked) {
        // Auto-apply if not locked
        if (contextInput) {
            contextInput.value = data.context;
            if (contextCharCount) {
                contextCharCount.textContent = `${data.context.length}/120`;
            }
            contextInput.classList.add('ai-updated');
            setTimeout(() => contextInput.classList.remove('ai-updated'), 2000);
        }
    } else if (data.context && contextLocked) {
        // Show suggestion but don't apply
        pendingAiContext = data.context;
        if (aiSuggestionText) aiSuggestionText.textContent = data.context;
        if (aiSuggestionIndicator) {
            aiSuggestionIndicator.classList.remove('hidden');
            aiSuggestionIndicator.classList.add('flex');
        }
    }
});

// Accept AI Suggestion button
if (acceptAiSuggestionBtn) {
    acceptAiSuggestionBtn.addEventListener('click', () => {
        if (pendingAiContext) {
            if (contextInput) {
                contextInput.value = pendingAiContext;
                if (contextCharCount) {
                    contextCharCount.textContent = `${pendingAiContext.length}/120`;
                }
            }
            updateManualContext(pendingAiContext);
            if (aiSuggestionIndicator) {
                aiSuggestionIndicator.classList.add('hidden');
                aiSuggestionIndicator.classList.remove('flex');
            }
            pendingAiContext = null;
        }
    });
}

// --- Director Controls Logic ---
async function loadStreamers() {
    try {
        const response = await fetch('/static/streamers.json');
        const data = await response.json();
        const select = document.getElementById('streamer-select');
        
        if (!select) return;
        
        select.innerHTML = '';
        data.streamers.forEach(streamer => {
            const option = document.createElement('option');
            option.value = streamer.id;
            option.textContent = streamer.display_name;
            select.appendChild(option);
        });
        
        select.value = 'peepingotter';
        updateStreamerContext(select.value);
    } catch (error) {
        console.error('Failed to load streamers:', error);
    }
}

function updateStreamerContext(streamerId) {
    socket.emit('set_streamer', { streamer_id: streamerId });
    console.log('[Director] Set streamer to:', streamerId);
}

function updateManualContext(contextText) {
    socket.emit('set_manual_context', { context: contextText });
    console.log('[Director] Set manual context:', contextText);
}

const streamerSelectEl = document.getElementById('streamer-select');
if (streamerSelectEl) {
    streamerSelectEl.addEventListener('change', (e) => {
        updateStreamerContext(e.target.value);
    });
}

const contextSubmitEl = document.getElementById('context-submit');
if (contextSubmitEl) {
    contextSubmitEl.addEventListener('click', () => {
        const input = document.getElementById('context-input');
        if (input) updateManualContext(input.value);
    });
}

const contextInputEl = document.getElementById('context-input');
if (contextInputEl) {
    contextInputEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            updateManualContext(contextInputEl.value);
        }
    });
}

window.onload = () => {
    // Note: initializeChart is now called by the drawer script
    loadStreamers();
};

// --- UI UPDATE HELPERS (Exposed for Drawers) ---
window.updateActiveUserUI = function(data) {
    const userContentEl = document.getElementById('user-content');
    if (!userContentEl) return;

    if (data.active_user) {
        const u = data.active_user;
        userContentEl.innerHTML = `
            <div class="user-card">
                <div class="flex justify-between items-center mb-2">
                    <h3>${u.username}</h3>
                    <span class="user-badge">${u.relationship.tier}</span>
                </div>
                <div class="text-xs text-gray-400 mb-2">Nickname: <span class="text-white">${u.nickname}</span></div>
                <div class="text-xs text-gray-400 mb-2">Affinity: <div class="w-full bg-gray-700 h-1 rounded mt-1"><div class="bg-purple-500 h-1 rounded" style="width: ${u.relationship.affinity}%"></div></div></div>
                <div class="text-xs text-gray-300 mt-2 border-t border-[#444] pt-2">
                    <strong>Facts:</strong>
                    <ul class="list-disc pl-4 mt-1 text-gray-400">
                        ${u.facts.slice(-3).map(f => `<li>${f.content}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    } else {
        userContentEl.innerHTML = `<p class="text-gray-500 italic text-sm text-center mt-4">No active user context.</p>`;
    }
};

window.updateMemoriesUI = function(data) {
    const memoryListEl = document.getElementById('memory-list');
    if (!memoryListEl) return;

    if (data.memories && data.memories.length > 0) {
        memoryListEl.innerHTML = data.memories.map(mem => `
            <li class="border-l-2 border-purple-500 pl-3 bg-[#2a2a2a] p-2 rounded text-sm">
                <div class="flex justify-between"><span class="text-purple-400 text-xs font-bold">${mem.source}</span> <span class="text-gray-500 text-xs">${mem.score}</span></div>
                <div class="text-gray-300 leading-snug">${mem.text}</div>
            </li>
        `).join('');
    } else {
        memoryListEl.innerHTML = '<li class="text-gray-500 italic text-sm text-center mt-4">No high-impact memories yet.</li>';
    }
};

// --- State Handlers ---
socket.on('director_state', (data) => {
    window.lastDirectorState = data;

    // 1. Mood
    const mood = data.mood || 'Neutral';
    if (moodText) moodText.textContent = mood;
    if (moodPill) moodPill.className = `px-4 py-2 border-2 rounded-full font-bold text-lg flex items-center gap-2 mood-${mood}`;
    
    // 2. Conversation State
    const state = data.conversation_state || 'IDLE';
    if (stateText) stateText.textContent = state;
    
    if (statePill) {
        if (state === 'FRUSTRATED') statePill.className = "px-4 py-2 border-2 border-red-600 bg-red-900 rounded-full font-bold text-lg flex items-center gap-2 text-red-100";
        else if (state === 'CELEBRATORY') statePill.className = "px-4 py-2 border-2 border-yellow-500 bg-yellow-900 rounded-full font-bold text-lg flex items-center gap-2 text-yellow-100";
        else if (state === 'IDLE') statePill.className = "px-4 py-2 border-2 border-gray-600 bg-gray-800 rounded-full font-bold text-lg flex items-center gap-2 text-gray-400";
        else statePill.className = "px-4 py-2 border-2 border-blue-500 bg-blue-900 rounded-full font-bold text-lg flex items-center gap-2 text-blue-100";
    }

    // 3. Dynamics
    if(flowText) flowText.textContent = data.flow || 'Unknown';
    if(intentText) intentText.textContent = data.intent || 'Unknown';

    // 4. Summary & Prediction
    if(summaryEl) summaryEl.textContent = data.summary || 'No summary.';
    if(summaryContextEl) summaryContextEl.textContent = data.raw_context || '';
    if(predictionEl) predictionEl.textContent = data.prediction || 'Observing flow...';

    // 5. Update Directive Panel
    if (data.directive) {
        const d = data.directive;
        if(dirObjectiveEl) dirObjectiveEl.textContent = d.objective || "Waiting...";
        if(dirToneEl) dirToneEl.textContent = d.tone || "Waiting...";
        if(dirActionEl) dirActionEl.textContent = d.suggested_action || "Waiting...";

        if (d.constraints && d.constraints.length > 0) {
            if(dirConstraintsBox) dirConstraintsBox.classList.remove('hidden');
            if(dirConstraintsEl) dirConstraintsEl.textContent = d.constraints.join(", ");
        } else {
            if(dirConstraintsBox) dirConstraintsBox.classList.add('hidden');
        }
    }

    // 6. Adaptive Metrics
    if (data.adaptive) {
        const a = data.adaptive;
        if (adaptiveStateLabel) {
            adaptiveStateLabel.textContent = a.state || "Normal";
            if (a.state.includes("Chaos")) adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-red-900 text-red-200";
            else if (a.state.includes("Dead")) adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-blue-900 text-blue-200";
            else adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-gray-700 text-gray-300";
        }
        if (thresholdBar) thresholdBar.style.width = `${(a.threshold || 0.9) * 100}%`;
        if (thresholdVal) thresholdVal.textContent = (a.threshold || 0.9).toFixed(2);
        if (velocityBar) velocityBar.style.width = `${Math.min((a.chat_velocity || 0) / 40 * 100, 100)}%`;
        if (velocityVal) velocityVal.textContent = `${(a.chat_velocity || 0).toFixed(1)} /m`;
        if (energyBar) energyBar.style.width = `${(a.energy || 0) * 100}%`;
        if (energyVal) energyVal.textContent = (a.energy || 0).toFixed(2);
        if (a.social_battery && batteryBar) {
            batteryBar.style.width = `${a.social_battery.percent}%`;
            if(batteryVal) batteryVal.textContent = `${a.social_battery.percent}%`;
            if (a.social_battery.percent < 20) batteryBar.className = "h-full bg-red-500 transition-all duration-500";
            else if (a.social_battery.percent < 50) batteryBar.className = "h-full bg-yellow-500 transition-all duration-500";
            else batteryBar.className = "h-full bg-green-500 transition-all duration-500";
        }
    }
    
    // 7. Lock states
    if (typeof data.streamer_locked !== 'undefined') {
        streamerLocked = data.streamer_locked;
        updateLockButtonUI(streamerLockBtn, streamerLocked);
    }
    if (typeof data.context_locked !== 'undefined') {
        contextLocked = data.context_locked;
        updateLockButtonUI(contextLockBtn, contextLocked);
    }
    
    // 8. Dynamic Drawer Updates (Check if elements exist first)
    updateActiveUserUI(data);
    updateMemoriesUI(data);
});

// --- Log Helpers ---
const MAX_LOG_LINES = 100;
const MAX_PANEL_LINES = 20;

function updateAmbientLog(elementId, logArray, newText, itemClass = '', isUpdate = false) {
    const el = document.getElementById(elementId);
    if(!el) return;
    
    if (isUpdate && logArray.length > 0) {
        logArray[logArray.length - 1] = newText;
    } else {
        logArray.push(newText);
        if (logArray.length > MAX_PANEL_LINES) {
            logArray.shift();
        }
    }
    
    el.innerHTML = '';
    logArray.forEach(text => {
        const div = document.createElement('div');
        div.textContent = text;
        div.style.marginBottom = '0.5rem';
        if (itemClass) div.className = itemClass;
        el.appendChild(div);
    });
    el.parentElement.scrollTop = el.parentElement.scrollHeight;
}

function appendLog(element, html) {
    if(!element) return;
    element.insertAdjacentHTML('beforeend', html);
    while (element.children.length > MAX_LOG_LINES) {
        element.removeChild(element.firstChild);
    }
    element.scrollTop = element.scrollHeight;
}

function sanitizeHTML(str) {
    const temp = document.createElement('div'); 
    temp.textContent = str; 
    return temp.innerHTML;
}

function highlightMentions(message) {
    const sanitizedMessage = sanitizeHTML(message);
    const regex = /(nami|peepingnami)/gi;
    return sanitizedMessage.replace(regex, '<span class="highlight-word">$&</span>');
}

// --- Data Listeners ---
socket.on('vision_context', d => updateAmbientLog('vision-context', visionLog, d.context));
socket.on('spoken_word_context', d => updateAmbientLog('spoken-word-context', spokenLog, d.context));
socket.on('audio_context', d => {
    const logContainer = document.getElementById('audio-context');
    if (!logContainer) return;

    let existingItem = document.getElementById(`session-${d.session_id}`);
    
    if (d.is_partial && existingItem) {
        existingItem.textContent = d.context;
        existingItem.classList.add('live-pulse');
    } else {
        const div = document.createElement('div');
        if (d.session_id) div.id = `session-${d.session_id}`;
        div.className = 'audio-highlight';
        div.textContent = d.context;
        logContainer.appendChild(div);
        logContainer.parentElement.scrollTop = logContainer.parentElement.scrollHeight;
    }
});

socket.on('event_scored', (data) => {
    // Push to global data
    window.scoreLabels.push(""); 
    window.scoreData.push(data.score.toFixed(2));
    if (window.scoreData.length > CHART_HISTORY_SIZE) { 
        window.scoreLabels.shift(); 
        window.scoreData.shift(); 
    }
    
    // Update chart if it exists
    if (window.interestChart) {
        window.interestChart.update('none');
    }
    
    const logLine = `${data.score.toFixed(2)} - ${data.source}: ${data.text.substring(0,30)}...`;
    window.graphLog.push(logLine);
    if(window.graphLog.length > 10) window.graphLog.shift();
    
    // Update log if it exists
    const graphDataLogEl = document.getElementById('graph-data-log');
    if(graphDataLogEl) {
        graphDataLogEl.textContent = window.graphLog.join('\n');
        graphDataLogEl.scrollTop = graphDataLogEl.scrollHeight;
    }
});

socket.on('twitch_message', (data) => {
    const isMention = /(nami|peepingnami)/gi.test(data.message);
    const bgClass = isMention ? 'mention-bg' : '';
    const formattedMessage = highlightMentions(data.message);
    
    const html = `
        <div class="log-line ${bgClass}">
            <span class="twitch-user">${sanitizeHTML(data.username)}:</span> 
            <span>${formattedMessage}</span>
        </div>`;
    
    appendLog(document.getElementById('twitch-messages'), html);
});

socket.on('bot_reply', (data) => {
    const sanitizedReply = sanitizeHTML(data.reply);
    const isCensored = data.is_censored || false;
    const reason = data.censorship_reason || "Unknown Policy"; 
    const filteredArea = data.filtered_area || "";
    
    const censorshipClass = isCensored ? 'censored-reply' : '';
    const censorshipIndicator = isCensored ? 
        `<span class="censored-indicator">🚨 FILTERED (${reason})</span>` : '';

    const namiHTML = `<div class="log-line nami-reply cursor-pointer ${censorshipClass}" onclick="openDrawer(this)" 
        data-reply="${encodeURIComponent(data.reply || '')}" 
        data-sent="${encodeURIComponent(data.prompt || '')}" 
        data-censored="${isCensored}"
        data-reason="${encodeURIComponent(reason)}"
        data-filtered-area="${encodeURIComponent(filteredArea)}">
        <strong>Nami:</strong> ${sanitizedReply}${censorshipIndicator}
        <span class="ml-2 opacity-0 group-hover:opacity-100 text-xs text-gray-400 align-middle">📄 Context</span></div>`;
    
    appendLog(document.getElementById('nami-replies'), namiHTML);

    const chatHTML = `
        <div class="log-line" style="background-color: rgba(99, 226, 183, 0.05);">
            <span class="twitch-user" style="color: #63e2b7;">Nami:</span> 
            <span>${sanitizedReply}</span>
        </div>`;
    
    appendLog(document.getElementById('twitch-messages'), chatHTML);
});

// Reuse drawer functions
window.openDrawer = function(el) {
    // ... same content as before ...
    const sent = decodeURIComponent(el.getAttribute('data-sent') || '');
    const replyRaw = decodeURIComponent(el.getAttribute('data-reply') || '');
    const isCensored = el.getAttribute('data-censored') === 'true';
    const reason = decodeURIComponent(el.getAttribute('data-reason') || 'Unknown');
    const filteredArea = decodeURIComponent(el.getAttribute('data-filtered-area') || '');
    
    document.getElementById('drawer-sent').textContent = sent || '(No prompt data available)';
    const replyEl = document.getElementById('drawer-reply');
    replyEl.innerHTML = '';
    replyEl.style.borderColor = '#3a3a3a';
    replyEl.style.backgroundColor = '#1f1f1f';
    
    if (isCensored) {
        replyEl.innerHTML = `
            <div class="mb-1 p-1 bg-red-900/30 border border-red-500 rounded text-sm">
                <div class="text-red-400 text-xs uppercase font-bold mb-1">⚠️ Safety Filter Triggered</div>
                <div class="flex items-center gap-1 mb-1">
                    <span class="text-gray-400 text-xs">Filtered Word:</span>
                    <span class="text-red-200 font-mono bg-red-900/50 px-1.5 py-0.5 rounded text-xs">${sanitizeHTML(reason)}</span>
                </div>
                ${filteredArea ? `<div><span class="text-gray-400 text-xs">Filtered Area:</span><span class="text-red-200 font-mono text-xs ml-1">${sanitizeHTML(filteredArea)}</span></div>` : ''}
            </div>
            <div class="text-gray-500 text-xs uppercase font-bold mb-1">Original Response:</div>
            <div class="text-gray-300 text-sm">${sanitizeHTML(replyRaw)}</div>`;
        replyEl.style.borderColor = '#ef4444';
        replyEl.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
    } else {
        replyEl.textContent = replyRaw || '(No reply data available)';
    }
    
    document.getElementById('context-drawer').classList.remove('translate-x-full');
    document.getElementById('context-drawer-overlay').classList.remove('hidden');
};

window.closeDrawer = function() {
    document.getElementById('context-drawer').classList.add('translate-x-full');
    document.getElementById('context-drawer-overlay').classList.add('hidden');
};

document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
