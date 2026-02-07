// ===================================
// DEBUG DRAWER SYSTEM - ADD TO script.js
// ===================================

let currentDebugDrawer = null;
let drawerRefreshIntervals = {};

// Open a debug drawer
async function openDebugDrawer(drawerName) {
    const drawer = document.getElementById('debug-drawer');
    const overlay = document.getElementById('debug-drawer-overlay');
    const content = document.getElementById('debug-drawer-content');
    
    if (!drawer || !overlay || !content) {
        console.error('Debug drawer elements not found');
        return;
    }
    
    // If same drawer, close it
    if (currentDebugDrawer === drawerName) {
        closeDebugDrawer();
        return;
    }
    
    // Stop any existing refresh intervals
    stopAllDrawerRefreshes();
    
    // Load drawer content
    try {
        const response = await fetch(`/static/drawers/${drawerName}.html`);
        if (!response.ok) throw new Error(`Failed to load drawer: ${response.status}`);
        
        const html = await response.text();
        content.innerHTML = html;
        
        // Update nav button states
        document.querySelectorAll('.debug-nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const activeBtn = document.querySelector(`.debug-nav-btn[data-drawer="${drawerName}"]`);
        if (activeBtn) activeBtn.classList.add('active');
        
        // Open drawer
        drawer.classList.add('open');
        overlay.classList.add('visible');
        currentDebugDrawer = drawerName;
        
        // Start refresh for this drawer (if it has a refresh function)
        const refreshFnName = `start${drawerName.charAt(0).toUpperCase() + drawerName.slice(1).replace(/_/g, '')}Refresh`;
        if (typeof window[refreshFnName] === 'function') {
            window[refreshFnName]();
        }
        
    } catch (error) {
        console.error('Error loading drawer:', error);
        content.innerHTML = `<div class="p-8 text-center text-red-400">Error loading drawer: ${error.message}</div>`;
    }
}

// Close debug drawer
function closeDebugDrawer() {
    const drawer = document.getElementById('debug-drawer');
    const overlay = document.getElementById('debug-drawer-overlay');
    
    if (drawer) drawer.classList.remove('open');
    if (overlay) overlay.classList.remove('visible');
    
    // Stop all refresh intervals
    stopAllDrawerRefreshes();
    
    // Clear active button
    document.querySelectorAll('.debug-nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    currentDebugDrawer = null;
}

// Stop all drawer refresh intervals
function stopAllDrawerRefreshes() {
    // Call stop functions for each drawer type
    const stopFunctions = [
        'stopThreadStatsRefresh',
        'stopPromptDebugRefresh', 
        'stopRunTestsRefresh'
    ];
    
    stopFunctions.forEach(fnName => {
        if (typeof window[fnName] === 'function') {
            window[fnName]();
        }
    });
}

// Close on overlay click
window.addEventListener('click', (e) => {
    if (e.target.id === 'debug-drawer-overlay') {
        closeDebugDrawer();
    }
});

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && currentDebugDrawer) {
        closeDebugDrawer();
    }
});