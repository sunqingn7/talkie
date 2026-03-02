/**
 * Music Player Widget - Controls music playback in browser
 */

class MusicPlayerWidget {
    constructor(widgetManager) {
        this.widgetManager = widgetManager;
        this.app = widgetManager.app;
        this.audio = new Audio();
        this.isPlaying = false;
        this.currentUrl = null;
        this.streamToken = null;
        this.mode = null; // 'direct' or 'stream'
        
        this.init();
    }
    
    init() {
        console.log('[MusicPlayerWidget] Initializing...');
        
        // Setup audio event listeners
        this.setupAudioListeners();
        
        // Setup button handlers
        this.setupButtonHandlers();
        
        // Override app's WebSocket message handler to intercept music messages
        this.setupWebSocketHandler();
        
        console.log('[MusicPlayerWidget] Initialized');
    }
    
    /**
     * Setup audio element event listeners
     */
    setupAudioListeners() {
        this.audio.onplay = () => {
            this.isPlaying = true;
            this.updateUI();
        };
        
        this.audio.onpause = () => {
            this.isPlaying = false;
            this.updateUI();
        };
        
        this.audio.onended = () => {
            this.stop();
            this.updateStatus('Not Playing');
        };
        
        this.audio.onerror = (e) => {
            console.error('[MusicPlayerWidget] Audio error:', e);
            this.updateStatus('Playback Error');
            this.updateUI();
            
            // Try to re-extract URL if streaming mode
            if (this.mode === 'stream' && this.streamToken) {
                console.log('[MusicPlayerWidget] Attempting to re-extract URL');
                this.sendControlMessage('music_reextract', { token: this.streamToken });
            }
        };
        
        this.audio.ontimeupdate = () => {
            // Could update progress bar here if we add one
        };
    }
    
    /**
     * Setup button click handlers
     */
    setupButtonHandlers() {
        const playPauseBtn = document.getElementById('music-play-pause');
        const stopBtn = document.getElementById('music-stop');
        const volumeSlider = document.getElementById('music-volume');
        
        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', () => this.toggle());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stop());
        }
        
        if (volumeSlider) {
            volumeSlider.addEventListener('input', (e) => {
                this.audio.volume = parseFloat(e.target.value);
            });
        }
    }
    
    /**
     * Setup WebSocket message handler for music control
     */
    setupWebSocketHandler() {
        // Store original handler if it exists
        const originalOnMessage = this.app.ws.onmessage;
        
        this.app.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Check if this is a music control message
                if (data.type === 'music_control') {
                    this.handleMusicControl(data);
                    return;
                }
                
                // Pass through to original handler if it exists
                if (originalOnMessage) {
                    originalOnMessage(event);
                }
            } catch (e) {
                console.error('[MusicPlayerWidget] Parse error:', e);
                if (originalOnMessage) {
                    originalOnMessage(event);
                }
            }
        };
    }
    
    /**
     * Handle music control messages from backend
     * @param {Object} data 
     */
    handleMusicControl(data) {
        console.log('[MusicPlayerWidget] Received control message:', data);
        
        switch (data.action) {
            case 'play':
                this.play(data.url, data.mode || 'direct');
                break;
            case 'stop':
                this.stop();
                break;
            case 'pause':
                this.audio.pause();
                break;
            case 'resume':
                this.audio.play();
                break;
        }
    }
    
    /**
     * Play audio
     * @param {string} url - Audio URL
     * @param {string} mode - 'direct' or 'stream'
     */
    play(url, mode = 'direct') {
        console.log('[MusicPlayerWidget] Playing:', url, 'mode:', mode);
        
        this.currentUrl = url;
        this.mode = mode;
        
        // Extract stream token if streaming mode
        if (mode === 'stream') {
            try {
                const urlObj = new URL(url);
                this.streamToken = urlObj.searchParams.get('token');
            } catch (e) {
                console.warn('[MusicPlayerWidget] Could not parse stream URL:', e);
            }
        }
        
        this.audio.src = url;
        this.audio.volume = document.getElementById('music-volume')?.value || 1;
        
        this.updateStatus('Loading...');
        
        this.audio.play().catch(e => {
            console.error('[MusicPlayerWidget] Play error:', e);
            this.updateStatus('Error');
        });
        
        this.updateUI();
    }
    
    /**
     * Stop playback
     */
    stop() {
        console.log('[MusicPlayerWidget] Stopping');
        
        // Notify backend to cleanup stream if applicable
        if (this.streamToken && this.app.ws && this.app.ws.readyState === WebSocket.OPEN) {
            this.sendControlMessage('music_stop_stream', { token: this.streamToken });
        }
        
        this.audio.pause();
        this.audio.src = "";
        this.audio.currentTime = 0;
        this.isPlaying = false;
        this.currentUrl = null;
        this.streamToken = null;
        this.mode = null;
        
        this.updateStatus('Not Playing');
        this.updateUI();
    }
    
    /**
     * Toggle play/pause
     */
    toggle() {
        if (!this.currentUrl) {
            console.log('[MusicPlayerWidget] No audio to play');
            return;
        }
        
        if (this.audio.paused) {
            this.audio.play().catch(e => {
                console.error('[MusicPlayerWidget] Play error:', e);
            });
        } else {
            this.audio.pause();
        }
    }
    
    /**
     * Update status text
     * @param {string} status 
     */
    updateStatus(status) {
        const statusEl = document.getElementById('music-status');
        if (statusEl) {
            statusEl.textContent = status;
            
            // Update status class for styling
            statusEl.classList.remove('playing', 'paused', 'error');
            if (status === 'Playing') {
                statusEl.classList.add('playing');
            } else if (status === 'Paused') {
                statusEl.classList.add('paused');
            } else if (status === 'Playback Error' || status === 'Error') {
                statusEl.classList.add('error');
            }
        }
    }
    
    /**
     * Update track display
     * @param {string} url 
     */
    updateTrack(url) {
        const trackEl = document.getElementById('music-track');
        if (trackEl && url) {
            // Extract friendly name from URL
            let displayName = url;
            
            try {
                const urlObj = new URL(url);
                
                if (urlObj.hostname.includes('youtube.com') || 
                    urlObj.hostname.includes('youtu.be')) {
                    const videoId = urlObj.searchParams.get('v') || 
                                   urlObj.pathname.split('/').pop();
                    displayName = 'YouTube: ' + (videoId || 'Unknown');
                } else if (urlObj.pathname) {
                    // For direct URLs, show filename
                    const filename = urlObj.pathname.split('/').pop();
                    displayName = filename.includes('?') ? 
                                  filename.split('?')[0] : filename;
                    if (displayName.length > 40) {
                        displayName = '...' + displayName.slice(-37);
                    }
                }
            } catch (e) {
                // Invalid URL, just use as-is
                console.warn('[MusicPlayerWidget] Could not parse URL:', e);
            }
            
            trackEl.textContent = displayName;
        }
    }
    
    /**
     * Update UI elements
     */
    updateUI() {
        // Update play/pause button
        const playPauseBtn = document.getElementById('music-play-pause');
        if (playPauseBtn) {
            const icon = playPauseBtn.querySelector('i');
            if (icon) {
                if (this.isPlaying && !this.audio.paused) {
                    icon.className = 'fas fa-pause';
                    this.updateStatus('Playing');
                } else if (this.currentUrl && this.audio.paused) {
                    icon.className = 'fas fa-play';
                    this.updateStatus('Paused');
                } else {
                    icon.className = 'fas fa-play';
                }
            }
        }
        
        // Update track display
        if (this.currentUrl && document.getElementById('music-track').textContent === '') {
            this.updateTrack(this.currentUrl);
        }
    }
    
    /**
     * Send control message to backend
     * @param {string} type 
     * @param {Object} data 
     */
    sendControlMessage(type, data = {}) {
        if (this.app.ws && this.app.ws.readyState === WebSocket.OPEN) {
            this.app.ws.send(JSON.stringify({
                type: type,
                ...data
            }));
        }
    }
}

// Make available globally
window.MusicPlayerWidget = MusicPlayerWidget;
