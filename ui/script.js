const socket = io("http://localhost:8002");

// DOM Elements
const moodPill = document.getElementById('mood-pill');
const moodText = document.getElementById('mood-text');
const statePill = document.getElementById('state-pill');
const stateText = document.getElementById('state-text');
const summaryEl = document.getElementById('summary-text');
const summaryContextEl = document.getElementById('summary-raw-context');
const predictionEl = document.getElementById('prediction-text');
const userContentEl = document.getElementById('user-content');
const memoryListEl = document.getElementById('memory-list');
const graphDataLogEl = document.getElementById('graph-data-log');

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

// Directive Elements (The missing piece!)
const dirObjectiveEl = document.getElementById('dir-objective');
const dirToneEl = document.getElementById('dir-tone');
const dirActionEl = document.getElementById('dir-action');
const dirConstraintsBox = document.getElementById('dir-constraints-box');
const dirConstraintsEl = document.getElementById('dir-constraints');

let lastAudioWasPartial = false;


// Context Logs
const visionLog = [];
const spokenLog = [];
const audioLog = [];
const graphLog = []; 

// Chart.js Setup
const CHART_HISTORY_SIZE = 50;
let interestChart;
const scoreLabels = []; 
const scoreData = [];

function initializeChart() {
    if (typeof Chart === 'undefined') return;
    const ctx = document.getElementById('interest-chart').getContext('2d');
    interestChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: scoreLabels,
            datasets: [{
                label: 'Interest Score', 
                data: scoreData, 
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
}
window.onload = initializeChart;

// --- State Handlers ---
socket.on('director_state', (data) => {
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

    // 3. Dynamics (Flow & Intent)
    if(flowText) flowText.textContent = data.flow || 'Unknown';
    if(intentText) intentText.textContent = data.intent || 'Unknown';

    // 4. Summary & Prediction
    if(summaryEl) summaryEl.textContent = data.summary || 'No summary.';
    if(summaryContextEl) summaryContextEl.textContent = data.raw_context || '';
    if(predictionEl) predictionEl.textContent = data.prediction || 'Observing flow...';

    // 5. --- NEW: Update Directive Panel ---
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

    // 6. Adaptive Metrics Update
    if (data.adaptive) {
        const a = data.adaptive;
        
        // State Label
        if (adaptiveStateLabel) {
            adaptiveStateLabel.textContent = a.state || "Normal";
            if (a.state.includes("Chaos")) adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-red-900 text-red-200";
            else if (a.state.includes("Dead")) adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-blue-900 text-blue-200";
            else adaptiveStateLabel.className = "text-xs px-2 py-1 rounded bg-gray-700 text-gray-300";
        }

        // Threshold
        if (thresholdBar) {
            const tVal = a.threshold || 0.9;
            thresholdBar.style.width = `${tVal * 100}%`;
            if(thresholdVal) thresholdVal.textContent = tVal.toFixed(2);
        }

        // Chat Velocity
        if (velocityBar) {
            const vVal = a.chat_velocity || 0;
            const vPct = Math.min((vVal / 40) * 100, 100); 
            velocityBar.style.width = `${vPct}%`;
            if(velocityVal) velocityVal.textContent = `${vVal.toFixed(1)} /m`;
        }

        // Energy
        if (energyBar) {
            const eVal = a.energy || 0;
            energyBar.style.width = `${eVal * 100}%`;
            if(energyVal) energyVal.textContent = eVal.toFixed(2);
        }

        // Social Battery
        if (a.social_battery && batteryBar) {
            const bat = a.social_battery;
            batteryBar.style.width = `${bat.percent}%`;
            if(batteryVal) batteryVal.textContent = `${bat.percent}%`;
            
            if (bat.percent < 20) batteryBar.className = "h-full bg-red-500 transition-all duration-500";
            else if (bat.percent < 50) batteryBar.className = "h-full bg-yellow-500 transition-all duration-500";
            else batteryBar.className = "h-full bg-green-500 transition-all duration-500";
        }
    }
    
    // 7. User
    if (data.active_user && userContentEl) {
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
    } else if (userContentEl) {
        userContentEl.innerHTML = `<p class="text-gray-500 italic text-sm text-center mt-4">No active user context.</p>`;
    }
    
    // 8. Memories
    if (memoryListEl) {
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
    }
});

// --- Log Helpers ---
const MAX_LOG_LINES = 100;
const MAX_PANEL_LINES = 20;

function updateAmbientLog(elementId, logArray, newText, itemClass = '', isUpdate = false) {
    const el = document.getElementById(elementId);
    if(!el) return;
    
    if (isUpdate && logArray.length > 0) {
        // Replace the last item instead of pushing
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

    // Check if we are updating a specific session item
    let existingItem = document.getElementById(`session-${d.session_id}`);
    
    if (d.is_partial && existingItem) {
        // Update the singular long item
        existingItem.textContent = d.context;
        existingItem.classList.add('live-pulse'); // Visual feedback it's still "typing"
    } else {
        // Create a new item (either a new session or a final transcript)
        const div = document.createElement('div');
        if (d.session_id) div.id = `session-${d.session_id}`;
        div.className = 'audio-highlight';
        div.textContent = d.context;
        
        logContainer.appendChild(div);
        
        // Auto-scroll
        logContainer.parentElement.scrollTop = logContainer.parentElement.scrollHeight;
    }
});
socket.on('event_scored', (data) => {
    // 1. Update Graph
    if (interestChart) {
        scoreLabels.push(""); 
        scoreData.push(data.score.toFixed(2));
        if (scoreData.length > CHART_HISTORY_SIZE) { 
            scoreLabels.shift(); 
            scoreData.shift(); 
        }
        interestChart.update('none');
    }
    
    // 2. Update Score Log
    const logLine = `${data.score.toFixed(2)} - ${data.source}: ${data.text.substring(0,30)}...`;
    graphLog.push(logLine);
    if(graphLog.length > 10) graphLog.shift();
    if(graphDataLogEl) graphDataLogEl.textContent = graphLog.join('\n');
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
    
    // --- FIX: Define 'reason' so the script doesn't crash ---
    const reason = data.censorship_reason || "Unknown Policy"; 
    
    const censorshipClass = isCensored ? 'censored-reply' : '';
    const censorshipIndicator = isCensored ? 
        `<span class="censored-indicator">🚨 FILTERED (${reason})</span>` : '';

    const namiHTML = `<div class="log-line nami-reply cursor-pointer ${censorshipClass}" onclick="openDrawer(this)" 
        data-reply="${encodeURIComponent(data.reply)}" 
        data-sent="${encodeURIComponent(data.prompt)}" 
        data-censored="${isCensored}"
        data-reason="${encodeURIComponent(reason)}">
        <strong>Nami:</strong> ${sanitizedReply}${censorshipIndicator}
        <span class="ml-2 opacity-0 group-hover:opacity-100 text-xs text-gray-400 align-middle">📄 Context</span></div>`;
    
    // This line will now run correctly
    appendLog(document.getElementById('nami-replies'), namiHTML);

    const chatHTML = `
        <div class="log-line" style="background-color: rgba(99, 226, 183, 0.05);">
            <span class="twitch-user" style="color: #63e2b7;">Nami:</span> 
            <span>${sanitizedReply}</span>
        </div>`;
    
    appendLog(document.getElementById('twitch-messages'), chatHTML);
});

window.openDrawer = function(el) {
    const sent = decodeURIComponent(el.getAttribute('data-sent'));
    const replyRaw = decodeURIComponent(el.getAttribute('data-reply'));
    const isCensored = el.getAttribute('data-censored') === 'true';
    const reason = decodeURIComponent(el.getAttribute('data-reason') || 'Unknown');
    
    document.getElementById('drawer-sent').textContent = sent;
    const replyEl = document.getElementById('drawer-reply');
    
    if (isCensored) {
        replyEl.innerHTML = `
            <div class="mb-2 p-2 bg-red-900/30 border border-red-500 rounded text-red-200 text-xs">
                <strong>Safety Trigger:</strong> Banned content detected ("${reason}")
            </div>
            <div style="color: #ef4444; font-weight: 600; margin-bottom: 0.5rem;">🚨 ORIGINAL FILTERED RESPONSE:</div>
        `;
        const textEl = document.createElement('div');
        textEl.textContent = replyRaw;
        replyEl.appendChild(textEl);
        replyEl.style.borderColor = '#ef4444';
        replyEl.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
    } else {
        replyEl.textContent = replyRaw;
        replyEl.style.borderColor = '#3a3a3a';
        replyEl.style.backgroundColor = '#1f1f1f';
    }
    
    document.getElementById('context-drawer').classList.remove('translate-x-full');
    document.getElementById('context-drawer-overlay').classList.remove('hidden');
};

window.closeDrawer = function() {
    document.getElementById('context-drawer').classList.add('translate-x-full');
    document.getElementById('context-drawer-overlay').classList.add('hidden');
};

document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });