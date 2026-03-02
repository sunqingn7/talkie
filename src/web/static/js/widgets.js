/**
 * Widget System - Extensible widget management for Talkie
 */

class WidgetManager {
    constructor(app) {
        this.app = app;
        this.widgets = new Map();
        this.panel = document.getElementById('widget-panel');
        this.container = document.getElementById('widget-container');
        this.toggleBtn = document.getElementById('toggle-widget-panel');
        
        if (!this.panel || !this.container) {
            console.warn('[WidgetManager] Widget panel not found in DOM');
            return;
        }
        
        this.init();
    }
    
    init() {
        // Setup toggle button
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.togglePanel());
        }
        
        // Load saved panel state
        this.loadPanelState();
        
        // Load configured widgets
        this.loadWidgets();
        
        console.log('[WidgetManager] Initialized');
    }
    
    /**
     * Register a widget class
     * @param {string} name - Widget identifier
     * @param {Class} widgetClass - Widget constructor
     */
    registerWidget(name, widgetClass) {
        this.widgets.set(name, widgetClass);
        console.log(`[WidgetManager] Registered widget: ${name}`);
    }
    
    /**
     * Load widgets based on configuration
     */
    loadWidgets() {
        // Default widgets to load
        const defaultWidgets = ['music'];
        
        defaultWidgets.forEach(widgetName => {
            const WidgetClass = this.widgets.get(widgetName);
            if (WidgetClass) {
                try {
                    new WidgetClass(this);
                    console.log(`[WidgetManager] Loaded widget: ${widgetName}`);
                } catch (error) {
                    console.error(`[WidgetManager] Failed to load widget ${widgetName}:`, error);
                }
            } else {
                console.warn(`[WidgetManager] Widget not registered: ${widgetName}`);
            }
        });
    }
    
    /**
     * Toggle widget panel visibility
     */
    togglePanel() {
        this.panel.classList.toggle('collapsed');
        this.updateToggleIcon();
        this.savePanelState();
    }
    
    /**
     * Update toggle button icon based on panel state
     */
    updateToggleIcon() {
        if (!this.toggleBtn) return;
        
        const icon = this.toggleBtn.querySelector('i');
        if (icon) {
            if (this.panel.classList.contains('collapsed')) {
                icon.className = 'fas fa-chevron-right';
            } else {
                icon.className = 'fas fa-chevron-left';
            }
        }
    }
    
    /**
     * Save panel state to localStorage
     */
    savePanelState() {
        try {
            const state = {
                collapsed: this.panel.classList.contains('collapsed'),
                timestamp: Date.now()
            };
            localStorage.setItem('talkie-widget-panel', JSON.stringify(state));
        } catch (e) {
            console.warn('[WidgetManager] Could not save panel state:', e);
        }
    }
    
    /**
     * Load panel state from localStorage
     */
    loadPanelState() {
        try {
            const saved = localStorage.getItem('talkie-widget-panel');
            if (saved) {
                const state = JSON.parse(saved);
                if (state.collapsed) {
                    this.panel.classList.add('collapsed');
                }
                this.updateToggleIcon();
            }
        } catch (e) {
            console.warn('[WidgetManager] Could not load panel state:', e);
        }
    }
    
    /**
     * Add widget element to panel
     * @param {HTMLElement} widgetElement 
     */
    addWidgetToPanel(widgetElement) {
        if (widgetElement) {
            this.container.appendChild(widgetElement);
        }
    }
    
    /**
     * Remove widget element from panel
     * @param {HTMLElement} widgetElement 
     */
    removeWidgetFromPanel(widgetElement) {
        if (widgetElement && widgetElement.parentNode) {
            this.container.removeChild(widgetElement);
        }
    }
    
    /**
     * Get widget by name
     * @param {string} name 
     * @returns {Class|undefined}
     */
    getWidget(name) {
        return this.widgets.get(name);
    }
    
    /**
     * Check if widget is registered
     * @param {string} name 
     * @returns {boolean}
     */
    hasWidget(name) {
        return this.widgets.has(name);
    }
}

// Make available globally
window.WidgetManager = WidgetManager;
