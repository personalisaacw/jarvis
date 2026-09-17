/**
 * JARVIS Code Review — Mobile PWA Client
 * 
 * Connects to the JARVIS review gateway via WebSocket for real-time
 * hunk synchronization. Supports swipe gestures for accept/reject,
 * tap buttons for all actions, and automatic state transitions.
 */

// ============================================================
// State
// ============================================================

let ws = null;
let currentHunk = null;
let sessionData = null;
let reconnectTimer = null;
const WS_URL = `ws://${window.location.host}/api/review/ws/live`;
const API_BASE = `${window.location.origin}/api/review`;

// ============================================================
// WebSocket Connection
// ============================================================

function connect() {
    if (ws && ws.readyState <= 1) return;
    
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('[WS] Connected');
        const status = document.getElementById('conn-status');
        status.textContent = 'Connected';
        status.classList.add('connected');
        clearTimeout(reconnectTimer);
    };
    
    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleEvent(msg);
        } catch (e) {
            console.error('[WS] Parse error:', e);
        }
    };
    
    ws.onclose = () => {
        console.log('[WS] Disconnected');
        const status = document.getElementById('conn-status');
        status.textContent = 'Reconnecting...';
        status.classList.remove('connected');
        reconnectTimer = setTimeout(connect, 3000);
    };
    
    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };
}

function sendCommand(action, hunkId) {
    const payload = { action };
    if (hunkId) payload.hunk_id = hunkId;
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    } else {
        // Fallback to REST
        let url, method = 'POST';
        switch (action) {
            case 'accept': url = `${API_BASE}/accept/${hunkId}`; break;
            case 'reject': url = `${API_BASE}/reject/${hunkId}`; break;
            case 'accept_all': url = `${API_BASE}/accept-all`; break;
            case 'reject_all': url = `${API_BASE}/reject-all`; break;
            case 'skip': url = `${API_BASE}/skip`; break;
            case 'explain': url = `${API_BASE}/explain/${hunkId}`; break;
            default: return;
        }
        fetch(url, { method }).catch(console.error);
    }
}

// ============================================================
// Event Handling
// ============================================================

function handleEvent(msg) {
    switch (msg.event) {
        case 'sync':
        case 'session_started':
            sessionData = msg.session;
            if (msg.current_hunk || msg.hunk) {
                currentHunk = msg.current_hunk || msg.hunk;
            }
            showScreen('review');
            updateReviewUI();
            break;
            
        case 'hunk_displayed':
            sessionData = msg.session;
            currentHunk = msg.hunk;
            updateReviewUI();
            break;
            
        case 'hunk_processed':
            sessionData = msg.session;
            flashProcessed(msg.hunk.status);
            break;
            
        case 'session_completed':
            sessionData = msg.session;
            showCompleted();
            break;
            
        case 'explanation_ready':
            showExplanation(msg.explanation);
            break;
    }
}

// ============================================================
// UI Updates
// ============================================================

function showScreen(name) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(`${name}-screen`).classList.add('active');
}

function updateReviewUI() {
    if (!sessionData || !currentHunk) return;
    
    showScreen('review');
    
    // Progress
    const total = sessionData.total_hunks;
    const idx = sessionData.current_hunk_index + 1;
    const progress = ((total - sessionData.pending) / total) * 100;
    
    document.getElementById('hunk-counter').textContent = `Block ${idx} of ${total}`;
    document.getElementById('file-badge').textContent = currentHunk.file_path.split('/').pop();
    document.getElementById('progress-fill').style.width = `${progress}%`;
    
    // Stats
    document.getElementById('stat-accepted').textContent = `✓ ${sessionData.accepted}`;
    document.getElementById('stat-rejected').textContent = `✗ ${sessionData.rejected}`;
    document.getElementById('stat-pending').textContent = `⏳ ${sessionData.pending}`;
    
    // Render diff
    renderDiff(currentHunk.diff_text);
}

function renderDiff(diffText) {
    const codeEl = document.getElementById('diff-code');
    codeEl.innerHTML = '';
    
    const lines = diffText.split('\n');
    for (const line of lines) {
        const span = document.createElement('span');
        span.textContent = line + '\n';
        
        if (line.startsWith('@@')) {
            span.className = 'diff-line-header';
        } else if (line.startsWith('+')) {
            span.className = 'diff-line-add';
        } else if (line.startsWith('-')) {
            span.className = 'diff-line-del';
        } else {
            span.className = 'diff-line-ctx';
        }
        
        codeEl.appendChild(span);
    }
    
    // Scroll to top
    document.getElementById('diff-card').scrollTop = 0;
}

function flashProcessed(status) {
    const card = document.getElementById('diff-card');
    card.classList.add(status === 'accepted' ? 'swiping-right' : 'swiping-left');
    setTimeout(() => {
        card.classList.remove('swiping-right', 'swiping-left');
    }, 300);
}

function showCompleted() {
    if (sessionData) {
        document.getElementById('complete-summary').textContent =
            `${sessionData.accepted} accepted, ${sessionData.rejected} rejected`;
    }
    showScreen('complete');
    
    // Return to idle after delay
    setTimeout(() => showScreen('idle'), 5000);
}

function showExplanation(text) {
    document.getElementById('explanation-text').textContent = text;
    document.getElementById('explanation-modal').classList.add('active');
}

function closeExplanation() {
    document.getElementById('explanation-modal').classList.remove('active');
}

// ============================================================
// Actions
// ============================================================

function acceptHunk() {
    if (currentHunk) sendCommand('accept', currentHunk.hunk_id);
}

function rejectHunk() {
    if (currentHunk) sendCommand('reject', currentHunk.hunk_id);
}

function explainHunk() {
    if (currentHunk) sendCommand('explain', currentHunk.hunk_id);
}

function acceptAll() {
    sendCommand('accept_all');
}

function rejectAll() {
    if (confirm('Reject all remaining changes?')) {
        sendCommand('reject_all');
    }
}

// ============================================================
// Swipe Gesture Support
// ============================================================

(function initSwipe() {
    const card = document.getElementById('diff-card');
    if (!card) return;
    
    let startX = 0;
    let currentX = 0;
    let isDragging = false;
    const THRESHOLD = 80;
    
    card.addEventListener('touchstart', (e) => {
        // Only track horizontal swipes starting from edges
        startX = e.touches[0].clientX;
        currentX = startX;
        isDragging = true;
    }, { passive: true });
    
    card.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        currentX = e.touches[0].clientX;
        const dx = currentX - startX;
        
        // Visual feedback
        card.style.transform = `translateX(${dx * 0.3}px)`;
        
        if (dx > 30) {
            card.classList.add('swiping-right');
            card.classList.remove('swiping-left');
        } else if (dx < -30) {
            card.classList.add('swiping-left');
            card.classList.remove('swiping-right');
        } else {
            card.classList.remove('swiping-right', 'swiping-left');
        }
    }, { passive: true });
    
    card.addEventListener('touchend', () => {
        if (!isDragging) return;
        isDragging = false;
        
        const dx = currentX - startX;
        card.style.transform = '';
        card.classList.remove('swiping-right', 'swiping-left');
        
        if (dx > THRESHOLD) {
            acceptHunk();
        } else if (dx < -THRESHOLD) {
            rejectHunk();
        }
    });
})();

// ============================================================
// Polling Fallback (if WebSocket is unavailable)
// ============================================================

async function pollStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        
        if (data.session) {
            sessionData = data.session;
            currentHunk = data.current_hunk;
            
            if (data.session.state === 'review_active' && currentHunk) {
                showScreen('review');
                updateReviewUI();
            } else if (data.session.state === 'completed') {
                showCompleted();
            }
        }
    } catch (e) {
        // Silently fail; WebSocket is primary channel
    }
}

// ============================================================
// Init
// ============================================================

connect();
// Fallback poll every 5s in case WS drops
setInterval(pollStatus, 5000);
