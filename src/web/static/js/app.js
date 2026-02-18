/**
 * Talkie Web Control Panel - Frontend Application
 */

class TalkieApp {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.voiceOutputEnabled = true;
        this.autoscrollEnabled = true;
        this.showToolDetails = false;
        this.currentModels = null;
        this.attachments = []; // Store uploaded file info
        this.isMusicPlaying = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.adjustTextareaHeight();
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => this.switchView(e));
        });
        
        // Chat input
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        messageInput.addEventListener('input', () => {
            this.adjustTextareaHeight();
        });
        
        sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.currentTarget.dataset.message;
                document.getElementById('message-input').value = message;
                this.sendMessage();
            });
        });
        
        // Voice input
        document.getElementById('voice-input-btn').addEventListener('click', () => this.startVoiceInput());
        document.getElementById('stop-recording-btn').addEventListener('click', () => this.stopVoiceInput());
        
        // Chat actions
        document.getElementById('clear-chat').addEventListener('click', () => this.clearChat());
        document.getElementById('voice-toggle').addEventListener('click', (e) => this.toggleVoiceOutput(e));
        
        // Settings
        document.getElementById('reconnect-btn').addEventListener('click', () => this.connectWebSocket());
        document.getElementById('voice-output-toggle').addEventListener('change', (e) => {
            this.voiceOutputEnabled = e.target.checked;
        });
        document.getElementById('voice-output-mode').addEventListener('change', (e) => {
            const mode = e.target.value;
            this.sendSystemMessage('set_voice_output', { mode: mode });
        });
        document.getElementById('autoscroll-toggle').addEventListener('change', (e) => {
            this.autoscrollEnabled = e.target.checked;
        });
        document.getElementById('show-tools-toggle').addEventListener('change', (e) => {
            this.showToolDetails = e.target.checked;
        });
        
        // Control panel buttons
        document.getElementById('test-tts-btn')?.addEventListener('click', () => {
            this.sendSystemMessage('test_tts');
        });
        document.getElementById('test-stt-btn')?.addEventListener('click', () => {
            this.startVoiceInput();
        });
        document.getElementById('check-weather-btn')?.addEventListener('click', () => {
            document.getElementById('message-input').value = "What's the weather today?";
            this.sendMessage();
        });
        
        // LLM Model management
        document.getElementById('restart-llm-btn')?.addEventListener('click', () => {
            this.restartLLMServer();
        });
        
        // TTS Speaker test button
        document.getElementById('test-speaker-btn')?.addEventListener('click', () => {
            this.testTTSSpeaker();
        });
        
        // File attachment handlers
        this.setupFileUploadHandlers();
    }
    
    setupFileUploadHandlers() {
        // Use setTimeout to ensure DOM is fully rendered
        setTimeout(() => {
            const fileInput = document.getElementById('file-input');
            const attachmentBtn = document.getElementById('attachment-btn');
            const chatView = document.getElementById('chat-view');
            
            console.log('Setting up file upload handlers:', {
                fileInput: !!fileInput,
                attachmentBtn: !!attachmentBtn,
                chatView: !!chatView
            });
            
            // Button click to select file
            if (attachmentBtn && fileInput) {
                attachmentBtn.onclick = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Attachment button clicked');
                    fileInput.click();
                    return false;
                };
                console.log('Attachment button onclick set successfully');
            } else {
                console.error('Attachment button or file input not found:', {
                    attachmentBtn: !!attachmentBtn,
                    fileInput: !!fileInput
                });
            }
            
            // File input change
            if (fileInput) {
                fileInput.onchange = (e) => {
                    this.handleFiles(e.target.files);
                    fileInput.value = ''; // Reset input
                };
            }
            
            // Drag and drop support
            if (chatView) {
                chatView.ondragover = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    chatView.classList.add('drag-active');
                };
                
                chatView.ondragleave = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    chatView.classList.remove('drag-active');
                };
                
                chatView.ondrop = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    chatView.classList.remove('drag-active');
                    this.handleFiles(e.dataTransfer.files);
                };
            }
        }, 100);
    }
    
    async handleFiles(files) {
        for (const file of files) {
            await this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        console.log('📤 Starting upload:', file.name, file.size, 'bytes');
        
        // Add pending attachment UI
        const attachmentId = this.attachments.length;
        const attachment = {
            id: attachmentId,
            filename: file.name,
            fileType: this.getFileType(file.name),
            status: 'uploading',
            content: null
        };
        this.attachments.push(attachment);
        this.renderAttachmentChip(attachment);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('transcribe', 'true');
            
            console.log('📤 Sending upload request...');
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            console.log('📤 Response status:', response.status);
            
            const result = await response.json();
            console.log('📤 Upload result:', result);
            
            if (result.success) {
                attachment.status = 'ready';
                attachment.attachmentId = result.attachment_id;
                attachment.content = result.content_preview;
                this.updateAttachmentChip(attachment);
                this.showNotification(`File uploaded: ${file.name}`, 'success');
            } else {
                attachment.status = 'error';
                attachment.error = result.error;
                this.updateAttachmentChip(attachment);
                this.showNotification(`Upload failed: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('📤 Upload error:', error);
            attachment.status = 'error';
            attachment.error = error.message;
            this.updateAttachmentChip(attachment);
            this.showNotification(`Upload error: ${error.message}`, 'error');
        }
    }
    
    getFileType(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const textExts = ['txt', 'md', 'markdown', 'csv', 'json', 'xml', 'yaml', 'yml', 
                         'py', 'js', 'ts', 'html', 'htm', 'css', 'sh', 'bash',
                         'c', 'cpp', 'h', 'hpp', 'java', 'kt', 'go', 'rs',
                         'rb', 'php', 'swift', 'sql', 'log', 'ini', 'conf'];
        const audioExts = ['mp3', 'wav', 'ogg', 'oga', 'm4a', 'aac', 'flac', 'wma'];
        const videoExts = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'flv', 'wmv', 'm4v'];
        
        if (ext === 'pdf') return 'pdf';
        if (ext === 'epub') return 'epub';
        if (textExts.includes(ext)) return 'text';
        if (audioExts.includes(ext)) return 'audio';
        if (videoExts.includes(ext)) return 'video';
        return 'file';
    }
    
    getFileIcon(fileType) {
        const icons = {
            'pdf': 'fa-file-pdf',
            'epub': 'fa-book',
            'text': 'fa-file-lines',
            'audio': 'fa-file-audio',
            'video': 'fa-file-video',
            'file': 'fa-file'
        };
        return icons[fileType] || 'fa-file';
    }
    
    renderAttachmentChip(attachment) {
        const previewContainer = document.getElementById('attachments-preview');
        if (!previewContainer) return;
        
        previewContainer.style.display = 'flex';
        
        const chip = document.createElement('div');
        chip.className = `attachment-chip ${attachment.status}`;
        chip.id = `attachment-${attachment.id}`;
        chip.innerHTML = `
            <i class="fas ${this.getFileIcon(attachment.fileType)}"></i>
            <span class="filename">${attachment.filename}</span>
            <button class="remove-btn" data-attachment-id="${attachment.id}" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Attach event listener to remove button
        const removeBtn = chip.querySelector('.remove-btn');
        removeBtn.addEventListener('click', () => this.removeAttachment(attachment.id));
        
        previewContainer.appendChild(chip);
        
        // Update attachment button state
        const btn = document.getElementById('attachment-btn');
        if (btn && this.attachments.some(a => a && a.status === 'ready')) {
            btn.classList.add('has-attachments');
        }
    }
    
    updateAttachmentChip(attachment) {
        const chip = document.getElementById(`attachment-${attachment.id}`);
        if (!chip) return;
        
        chip.className = `attachment-chip ${attachment.status}`;
        
        if (attachment.status === 'error') {
            chip.title = `Error: ${attachment.error}`;
        }
        
        // Update button state
        const btn = document.getElementById('attachment-btn');
        if (btn) {
            if (this.attachments.some(a => a && a.status === 'ready')) {
                btn.classList.add('has-attachments');
            } else {
                btn.classList.remove('has-attachments');
            }
        }
    }
    
    removeAttachment(id) {
        const attachment = this.attachments[id];
        if (!attachment) return;
        
        // Remove from array
        this.attachments[id] = null;
        
        // Remove from UI
        const chip = document.getElementById(`attachment-${id}`);
        if (chip) {
            chip.remove();
        }
        
        // Hide container if no attachments left
        const previewContainer = document.getElementById('attachments-preview');
        if (previewContainer && !this.attachments.some(a => a && a.status === 'ready')) {
            previewContainer.style.display = 'none';
            previewContainer.innerHTML = '';
        }
        
        // Update button state
        const btn = document.getElementById('attachment-btn');
        if (btn) {
            btn.classList.remove('has-attachments');
        }
    }
    
    clearAttachments() {
        this.attachments = [];
        const previewContainer = document.getElementById('attachments-preview');
        if (previewContainer) {
            previewContainer.innerHTML = '';
            previewContainer.style.display = 'none';
        }
        const btn = document.getElementById('attachment-btn');
        if (btn) {
            btn.classList.remove('has-attachments');
        }
    }
    
    updateEngineUI(engine) {
        // Update engine description
        const descriptions = {
            'edge_tts': '<i class="fas fa-info-circle" style="margin-right: 6px;"></i>Uses Microsoft Edge online TTS service. Requires internet connection.',
            'coqui': '<i class="fas fa-info-circle" style="margin-right: 6px;"></i>Uses local XTTS models. Works offline but requires ~1.5GB download.',
            'pyttsx3': '<i class="fas fa-info-circle" style="margin-right: 6px;"></i>Basic offline TTS. Limited quality and language support.'
        };
        
        const descEl = document.getElementById('engine-description');
        if (descEl) {
            descEl.innerHTML = descriptions[engine] || '';
        }
        
        // Update current engine badge
        const engineNames = {
            'edge_tts': 'Edge TTS',
            'coqui': 'Coqui',
            'pyttsx3': 'pyttsx3'
        };
        const badge = document.getElementById('current-engine-badge');
        if (badge) {
            badge.textContent = engineNames[engine] || engine;
        }
        
        // Show/hide sections based on engine
        const xttsSection = document.getElementById('xtts-model-section');
        const speakerSection = document.querySelector('.speaker-section');
        
        if (xttsSection) {
            xttsSection.style.display = engine === 'coqui' ? 'block' : 'none';
        }
        // Speaker section is always visible - it shows voices for Edge TTS or personas for Coqui
        if (speakerSection) {
            speakerSection.style.display = 'block';
        }
        
        // Update speaker section label based on engine
        const speakerLabel = document.querySelector('.speaker-section h4');
        if (speakerLabel) {
            if (engine === 'edge_tts') {
                speakerLabel.innerHTML = '<i class="fas fa-globe" style="margin-right: 8px;"></i>Voice';
            } else {
                speakerLabel.innerHTML = '<i class="fas fa-user-voice" style="margin-right: 8px;"></i>Voice Persona';
            }
        }
        
        // Update speaker description
        const speakerDesc = document.querySelector('.speaker-section .card-description');
        if (speakerDesc) {
            if (engine === 'edge_tts') {
                speakerDesc.textContent = 'Select a voice for speech synthesis';
            } else {
                speakerDesc.textContent = 'Select a voice persona for the assistant';
            }
        }
    }
    
    switchTTSEngine(engineId) {
        this.sendSystemMessage('switch_tts_engine', { engine_id: engineId });
    }
    
    switchEdgeVoice(voiceId) {
        console.log('Switching Edge TTS voice to:', voiceId);
        this.sendSystemMessage('switch_edge_voice', { voice_id: voiceId });
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(), this.reconnectDelay);
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'system_status':
                this.updateSystemStatus(data);
                break;
                
            case 'available_models':
                this.updateModelsList(data);
                break;
                
            case 'assistant_message':
                console.log('[Assistant Message] skip_voice:', data.skip_voice);
                this.addAssistantMessage(data.content, data.tool_calls, data.skip_voice);
                this.hideThinking();
                break;
                
            case 'thinking':
                this.showThinking();
                break;
                
            case 'error':
                this.addErrorMessage(data.content);
                this.hideThinking();
                break;
                
            case 'model_switched':
                this.showNotification(data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    document.getElementById('current-tts-model').textContent = data.new_model;
                    // Speakers already updated in the response
                }
                break;
                
            case 'tts_speakers':
                this.updateTTSSpeakers({
                    tts_speakers: data.speakers,
                    current_tts_speaker: data.current_speaker,
                    current_tts_engine: data.current_tts_engine,
                    edge_voices: data.edge_voices,
                    current_edge_voice: data.current_edge_voice
                });
                break;
                
            case 'speaker_switched':
                this.showNotification(data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    // Update the dropdown to show new selection
                    const speakerSelect = document.getElementById('tts-speaker-select');
                    if (speakerSelect && data.new_speaker) {
                        speakerSelect.value = data.new_speaker;
                    }
                    // Update current speaker display
                    const currentSpeakerName = document.getElementById('current-speaker-name');
                    if (currentSpeakerName) {
                        currentSpeakerName.textContent = data.new_speaker;
                    }
                    // No need to refresh - already have the data
                }
                break;

            case 'tts_engine_switched':
                // Show notification - use warning if it's a fallback
                const notifType = data.message && data.message.includes('not available') ? 'warning' : (data.success ? 'success' : 'error');
                this.showNotification(data.message, notifType);

                if (data.success) {
                    // Update UI with the ACTUAL engine (might be different from requested)
                    this.updateEngineUI(data.new_engine);

                    // Update sidebar engine display
                    const engineNames = {
                        'edge_tts': 'Edge TTS',
                        'coqui': 'Coqui TTS',
                        'pyttsx3': 'pyttsx3'
                    };
                    const engineDisplay = document.getElementById('current-tts-engine');
                    if (engineDisplay) {
                        engineDisplay.textContent = engineNames[data.new_engine] || data.new_engine;
                    }

                    // Update the dropdown to match the actual engine
                    const engineSelect = document.getElementById('tts-engine-select');
                    if (engineSelect && data.new_engine) {
                        engineSelect.value = data.new_engine;
                    }

                    // Note: get_models already sent by server, no need to call again
                }
                break;

            case 'edge_voice_switched':
                this.showNotification(data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    // Update speaker dropdown (shared between Edge and Coqui)
                    const speakerSelect = document.getElementById('tts-speaker-select');
                    if (speakerSelect && data.new_voice) {
                        speakerSelect.value = data.new_voice;
                    }
                    // Update current speaker display
                    const currentSpeakerName = document.getElementById('current-speaker-name');
                    if (currentSpeakerName) {
                        // Extract name from the voice ID or use the value from speakers list
                        const voiceOption = speakerSelect?.options[speakerSelect.selectedIndex];
                        if (voiceOption) {
                            currentSpeakerName.textContent = voiceOption.text.split(' (')[0];
                        }
                    }
                }
                break;

            case 'tts_test_result':
                if (data.success) {
                    this.showNotification('Voice test completed!', 'success');
                } else {
                    this.showNotification(data.message || 'Voice test failed', 'error');
                }
                break;
                
            case 'tts_demo_result':
                // Demo is playing, no need to show notification to avoid interruption
                if (!data.success) {
                    this.showNotification(data.message || 'Voice demo failed', 'error');
                }
                break;
                
            case 'tts_spoken':
                // Assistant response spoken - debug logging only
                if (!data.success) {
                    console.error('TTS failed:', data.message);
                } else {
                    console.log(`TTS spoken: ${data.characters} characters using speaker: ${data.speaker}, output: ${data.voice_output}`);
                }
                break;
            
            case 'audio':
                // Receive audio from server and play in browser
                if (data.audio_data) {
                    console.log('[Web Audio] Received audio, type:', data.audio_type);
                    try {
                        const audioBytes = atob(data.audio_data);
                        const audioBuffer = new Uint8Array(audioBytes.length);
                        for (let i = 0; i < audioBytes.length; i++) {
                            audioBuffer[i] = audioBytes.charCodeAt(i);
                        }
                        
                        const blob = new Blob([audioBuffer], { type: data.format === 'mp3' ? 'audio/mpeg' : 'audio/wav' });
                        const audioUrl = URL.createObjectURL(blob);
                        const audio = new Audio(audioUrl);
                        audio.play();
                        console.log('[Web Audio] Playing audio in browser');
                    } catch (e) {
                        console.error('[Web Audio] Error playing audio:', e);
                    }
                }
                break;
                
            case 'llm_status':
                // Update LLM status display
                const llmStatusEl = document.getElementById('llm-server-status');
                if (llmStatusEl) {
                    if (data.running) {
                        llmStatusEl.innerHTML = '<i class="fas fa-circle" style="color: var(--success-color);"></i> Server Running';
                        llmStatusEl.style.background = 'rgba(16, 185, 129, 0.1)';
                        llmStatusEl.style.color = 'var(--success-color)';
                    } else {
                        llmStatusEl.innerHTML = '<i class="fas fa-circle" style="color: var(--error-color);"></i> Server Offline';
                        llmStatusEl.style.background = 'rgba(239, 68, 68, 0.1)';
                        llmStatusEl.style.color = 'var(--error-color)';
                    }
                }
                break;
                
            case 'llm_switching':
                this.showNotification(data.message, 'info');
                break;
                
            case 'llm_model_switched':
                this.showNotification(data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    // Refresh models list to show new active model
                    this.sendSystemMessage('get_models');
                }
                break;
                
            case 'llm_restarting':
                this.showNotification(data.message, 'info');
                break;
                
            case 'llm_server_restarted':
                this.showNotification(data.message, data.success ? 'success' : 'error');
                break;
                
            case 'history_cleared':
                this.clearChatDisplay();
                this.showNotification('Conversation history cleared', 'success');
                break;
        }
    }
    
    sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();
        const readyAttachments = this.attachments.filter(a => a && a.status === 'ready');
        
        // Allow sending with just attachments even if no text
        if ((!message && readyAttachments.length === 0) || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Stop any ongoing chat voice when user sends new message
        this.stopCurrentSpeech();
        
        // Add user message to chat (with file indicator)
        let displayMessage = message;
        if (readyAttachments.length > 0) {
            const fileNames = readyAttachments.map(a => a.filename).join(', ');
            if (displayMessage) {
                displayMessage += `\n[Attached: ${fileNames}]`;
            } else {
                displayMessage = `[Attached: ${fileNames}]`;
            }
        }
        this.addUserMessage(displayMessage);
        
        // Get attachment IDs for the server
        const attachmentIds = readyAttachments.map(a => a.attachmentId);
        
        // Send to server
        this.ws.send(JSON.stringify({
            type: 'user_message',
            content: message,
            attachment_ids: attachmentIds
        }));
        
        // Clear input and attachments
        input.value = '';
        this.adjustTextareaHeight();
        this.clearAttachments();
    }
    
    sendSystemMessage(type, data = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.log('Cannot send - WebSocket not ready');
            return;
        }
        
        this.ws.send(JSON.stringify({
            type: type,
            ...data
        }));
    }
    
    addUserMessage(content) {
        const messages = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-user"></i></div>
            <div class="message-content">
                ${this.escapeHtml(content)}
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
        `;
        messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addAssistantMessage(content, toolCalls = null, skipVoice = false) {
        const messages = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        let toolInfo = '';
        if (toolCalls && this.showToolDetails) {
            toolInfo = `
                <div class="tool-calls" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color); font-size: 12px; color: var(--text-muted);">
                    <i class="fas fa-wrench"></i> Tools used: ${toolCalls.map(t => t.tool).join(', ')}
                </div>
            `;
        }

        // Check if music was played or stopped
        if (toolCalls) {
            const toolsUsed = toolCalls.map(t => t.tool);
            if (toolsUsed.includes('music_player')) {
                // Check the result to see if playing or stopped
                // For simplicity, we'll toggle based on action in the message
                const isPlaying = content.toLowerCase().includes('playing') && !content.toLowerCase().includes('stop');
                this.updateMusicButton(isPlaying);
            }
        }

        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                ${this.formatMessage(content)}
                ${toolInfo}
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
        `;
        messages.appendChild(messageDiv);
        this.scrollToBottom();

        // Voice output - use backend TTS with selected speaker
        // Skip voice if explicitly requested (e.g., when file reading is handling speech)
        if (this.voiceOutputEnabled && !skipVoice) {
            this.speakTextBackend(content);
        }
    }

    updateMusicButton(isPlaying) {
        this.isMusicPlaying = isPlaying;
        
        // Update button by ID (template button)
        const musicBtn = document.getElementById('music-btn');
        if (musicBtn) {
            if (isPlaying) {
                musicBtn.dataset.message = 'Stop the music';
                musicBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Music';
            } else {
                musicBtn.dataset.message = 'Play some music';
                musicBtn.innerHTML = '<i class="fas fa-music"></i> Play Music';
            }
        }
        
        // Also update button in welcome screen (JS rendered)
        const welcomeBtn = document.querySelector('.welcome-message #music-btn');
        if (welcomeBtn) {
            if (isPlaying) {
                welcomeBtn.dataset.message = 'Stop the music';
                welcomeBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Music';
            } else {
                welcomeBtn.dataset.message = 'Play some music';
                welcomeBtn.innerHTML = '<i class="fas fa-music"></i> Play Music';
            }
        }
    }
    
    addErrorMessage(content) {
        const messages = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        messageDiv.innerHTML = `
            <div class="message-avatar" style="background: var(--error-color);"><i class="fas fa-exclamation"></i></div>
            <div class="message-content" style="border-color: var(--error-color);">
                <i class="fas fa-triangle-exclamation"></i> ${this.escapeHtml(content)}
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
        `;
        messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showThinking() {
        document.getElementById('thinking-indicator').style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideThinking() {
        document.getElementById('thinking-indicator').style.display = 'none';
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the conversation?')) {
            this.sendSystemMessage('clear_history');
        }
    }
    
    clearChatDisplay() {
        const messages = document.getElementById('messages');
        messages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <h3>Welcome to Talkie!</h3>
                <p>I'm your voice assistant. You can chat with me using text or voice.</p>
                <div class="quick-actions">
                    <button class="quick-btn" id="music-btn" data-message="Play some music">
                        <i class="fas fa-music"></i>
                        Play Music
                    </button>
                    <button class="quick-btn" data-message="What's the weather today?">
                        <i class="fas fa-cloud-sun"></i>
                        Check Weather
                    </button>
                    <button class="quick-btn" data-message="Tell me a joke">
                        <i class="fas fa-laugh-beam"></i>
                        Tell a Joke
                    </button>
                    <button class="quick-btn" data-message="What can you do?">
                        <i class="fas fa-wand-magic-sparkles"></i>
                        My Capabilities
                    </button>
                </div>
            </div>
        `;
        
        // Re-attach event listeners
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.currentTarget.dataset.message;
                document.getElementById('message-input').value = message;
                this.sendMessage();
            });
        });
    }
    
    switchView(e) {
        const viewName = e.currentTarget.dataset.view;
        
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        e.currentTarget.classList.add('active');
        
        // Update views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(`${viewName}-view`).classList.add('active');
    }
    
    updateSystemStatus(data) {
        // Sidebar status
        const statusDot = document.querySelector('#system-status .status-dot');
        const statusText = document.querySelector('#system-status .status-text');
        
        if (data.mcp_server_ready && data.llm_client_ready) {
            statusDot.classList.remove('offline');
            statusDot.classList.add('online');
            statusText.textContent = 'Online';
        } else {
            statusDot.classList.remove('online');
            statusDot.classList.add('offline');
            statusText.textContent = 'Connecting...';
        }
        
        // Update model info
        if (data.config) {
            // Update TTS Engine display
            const engineNames = {
                'edge_tts': 'Edge TTS',
                'coqui': 'Coqui TTS',
                'pyttsx3': 'pyttsx3'
            };
            const engineDisplay = document.getElementById('current-tts-engine');
            if (engineDisplay) {
                const engineName = engineNames[data.config.tts_engine] || data.config.tts_engine || 'Unknown';
                engineDisplay.textContent = engineName;
            }
            document.getElementById('current-llm-model').textContent = data.config.llm_model || 'Unknown';
            
            // Update voice output mode dropdown
            const voiceOutputSelect = document.getElementById('voice-output-mode');
            if (voiceOutputSelect && data.config.voice_output) {
                voiceOutputSelect.value = data.config.voice_output;
            }
        }
        
        // Update control panel status
        const mcpStatus = document.getElementById('mcp-status');
        const llmStatus = document.getElementById('llm-status');
        const toolsCount = document.getElementById('tools-count');
        const conversationCount = document.getElementById('conversation-count');
        const availableTools = document.getElementById('available-tools');
        
        if (mcpStatus) {
            mcpStatus.textContent = data.mcp_server_ready ? 'Online' : 'Offline';
            mcpStatus.className = 'status-badge ' + (data.mcp_server_ready ? 'online' : 'offline');
        }
        
        if (llmStatus) {
            llmStatus.textContent = data.llm_client_ready ? 'Online' : 'Offline';
            llmStatus.className = 'status-badge ' + (data.llm_client_ready ? 'online' : 'offline');
        }
        
        if (toolsCount) {
            toolsCount.textContent = data.available_tools?.length || 0;
        }
        
        if (conversationCount) {
            conversationCount.textContent = data.conversation_count || 0;
        }
        
        // Update available tools grid
        if (availableTools && data.available_tools) {
            const toolIcons = {
                listen: 'fa-microphone',
                speak: 'fa-volume-high',
                weather: 'fa-cloud-sun',
                execute_command: 'fa-terminal',
                read_file: 'fa-file-lines',
                write_file: 'fa-pen-to-square',
                list_directory: 'fa-folder-open',
                wake_word: 'fa-hand-sparkles',
                voice_activity: 'fa-wave-square',
                timer: 'fa-clock',
                calculator: 'fa-calculator',
                web_search: 'fa-magnifying-glass',
                web_news: 'fa-newspaper'
            };
            
            availableTools.innerHTML = data.available_tools.map(tool => `
                <div class="tool-card">
                    <i class="fas ${toolIcons[tool] || 'fa-puzzle-piece'}"></i>
                    <span>${this.formatToolName(tool)}</span>
                </div>
            `).join('');
        }
    }
    
    updateModelsList(data) {
        this.currentModels = data;
        
        // Update TTS Engine selector
        const engineSelect = document.getElementById('tts-engine-select');
        if (engineSelect && data.current_tts_engine) {
            engineSelect.value = data.current_tts_engine;
            this.updateEngineUI(data.current_tts_engine);
        }
        
        // Update Edge TTS voices
        const edgeVoiceSelect = document.getElementById('edge-voice-select');
        if (edgeVoiceSelect && data.edge_voices) {
            edgeVoiceSelect.innerHTML = data.edge_voices.map(voice => `
                <option value="${voice.id}" ${voice.id === data.current_edge_voice ? 'selected' : ''}>
                    ${voice.name} (${voice.gender}) - ${voice.locale}
                </option>
            `).join('');
        }
        
        // Update TTS models (for Coqui)
        const ttsModelList = document.getElementById('tts-model-list');
        if (ttsModelList && data.tts_models) {
            ttsModelList.innerHTML = data.tts_models.map(model => `
                <div class="model-item ${model.id === data.current_tts_model ? 'active' : ''}" data-model-id="${model.id}" data-model-type="tts">
                    <div class="model-info-text">
                        <div class="model-name">${model.name}</div>
                        <div class="model-meta">${model.language} • ${model.size}</div>
                    </div>
                    ${model.id === data.current_tts_model ? 
                        '<span class="model-badge active">Active</span>' : 
                        (model.requires_license ? '<span class="model-badge license">License</span>' : '')
                    }
                </div>
            `).join('');
            
            // Add click handlers for TTS models
            ttsModelList.querySelectorAll('.model-item').forEach(item => {
                item.addEventListener('click', () => {
                    const modelId = item.dataset.modelId;
                    if (modelId !== data.current_tts_model) {
                        this.switchTTSModel(modelId);
                    }
                });
            });
        }
        
        // Update TTS speakers
        this.updateTTSSpeakers(data);
        
        // Update LLM models
        const llmModelList = document.getElementById('llm-model-list');
        const llmServerStatus = document.getElementById('llm-server-status');
        
        if (llmServerStatus) {
            if (data.llm_server_running) {
                llmServerStatus.innerHTML = '<i class="fas fa-circle" style="color: var(--success-color);"></i> Server Running';
                llmServerStatus.style.background = 'rgba(16, 185, 129, 0.1)';
                llmServerStatus.style.color = 'var(--success-color)';
            } else {
                llmServerStatus.innerHTML = '<i class="fas fa-circle" style="color: var(--error-color);"></i> Server Offline';
                llmServerStatus.style.background = 'rgba(239, 68, 68, 0.1)';
                llmServerStatus.style.color = 'var(--error-color)';
            }
        }
        
        if (llmModelList && data.llm_models) {
            llmModelList.innerHTML = data.llm_models.map(model => {
                const isActive = model.id === data.current_llm_model || 
                    (data.current_llm_model && data.current_llm_model.includes(model.file));
                const canSelect = model.exists;
                
                return `
                    <div class="model-item ${isActive ? 'active' : ''} ${!canSelect ? 'disabled' : ''}" 
                         data-model-id="${model.id}" 
                         data-model-type="llm"
                         style="${!canSelect ? 'opacity: 0.5; cursor: not-allowed;' : ''}">
                        <div class="model-info-text">
                            <div class="model-name">${model.name}</div>
                            <div class="model-meta">${model.parameters || 'Unknown params'} • ${model.size} • ${model.quantization || 'Unknown'}</div>
                            <div class="model-desc" style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">${model.description || ''}</div>
                        </div>
                        ${isActive ? 
                            '<span class="model-badge active">Active</span>' : 
                            (!canSelect ? '<span class="model-badge">Not Downloaded</span>' : '')
                        }
                    </div>
                `;
            }).join('');
            
            // Add click handlers for LLM models
            llmModelList.querySelectorAll('.model-item').forEach(item => {
                item.addEventListener('click', () => {
                    const modelId = item.dataset.modelId;
                    const isDisabled = item.classList.contains('disabled');
                    
                    if (!isDisabled && !item.classList.contains('active')) {
                        if (confirm(`Switch to ${modelId}? This will restart the LLM server and may take a moment.`)) {
                            this.switchLLMModel(modelId);
                        }
                    }
                });
            });
        }
    }
    
    switchTTSModel(modelId) {
        this.sendSystemMessage('switch_model', { model_id: modelId });
    }
    
    switchLLMModel(modelId) {
        this.sendSystemMessage('switch_llm_model', { model_id: modelId });
    }
    
    restartLLMServer() {
        if (confirm('Restart the LLM server? Current conversation will be interrupted.')) {
            this.sendSystemMessage('restart_llm_server');
        }
    }
    
    updateTTSSpeakers(data) {
        const speakerSelect = document.getElementById('tts-speaker-select');
        const speakerCountBadge = document.getElementById('speaker-count-badge');
        const currentSpeakerInfo = document.getElementById('current-speaker-info');
        const currentSpeakerName = document.getElementById('current-speaker-name');
        const currentSpeakerAvatar = document.getElementById('current-speaker-avatar');
        
        // Determine which speakers to use based on current engine
        const currentEngine = data.current_tts_engine || 'edge_tts';
        let speakers = [];
        let currentSpeaker = null;
        
        if (currentEngine === 'edge_tts' && data.edge_voices) {
            // Use Edge TTS voices
            speakers = data.edge_voices;
            currentSpeaker = data.current_edge_voice;
        } else if (data.tts_speakers) {
            // Use Coqui/pyttsx3 speakers
            speakers = data.tts_speakers;
            currentSpeaker = data.current_tts_speaker;
        }
        
        // Update speaker count badge
        if (speakerCountBadge && speakers) {
            const count = speakers.length;
            speakerCountBadge.textContent = `${count} voice${count !== 1 ? 's' : ''}`;
        }
        
        // Update dropdown options
        if (speakerSelect && speakers && speakers.length > 0) {
            // Enable select
            speakerSelect.disabled = false;
            
            // Build options
            let optionsHTML = '';
            
            speakers.forEach((speaker, index) => {
                const isSelected = speaker.id === currentSpeaker || 
                    (!currentSpeaker && index === 0);
                const selectedAttr = isSelected ? 'selected' : '';
                
                // For Edge TTS, show locale info
                if (currentEngine === 'edge_tts' && speaker.locale) {
                    optionsHTML += `<option value="${speaker.id}" ${selectedAttr}>${speaker.name} (${speaker.gender}) - ${speaker.locale}</option>`;
                } else {
                    optionsHTML += `<option value="${speaker.id}" ${selectedAttr}>${speaker.name}</option>`;
                }
            });
            
            speakerSelect.innerHTML = optionsHTML;
            
            // Show current speaker info
            if (currentSpeakerInfo) {
                currentSpeakerInfo.style.display = 'block';
            }
            
            // Update current speaker display
            if (currentSpeakerName) {
                const activeSpeaker = speakers.find(s => s.id === currentSpeaker) || 
                                     speakers[0];
                if (activeSpeaker) {
                    currentSpeakerName.textContent = activeSpeaker.name;
                    
                    // Update avatar color based on speaker name
                    if (currentSpeakerAvatar) {
                        const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#14b8a6'];
                        const colorIndex = activeSpeaker.name.length % colors.length;
                        currentSpeakerAvatar.style.background = colors[colorIndex];
                    }
                }
            }
            
            // Add change handler
            speakerSelect.onchange = (e) => {
                const speakerId = e.target.value;
                if (speakerId) {
                    if (currentEngine === 'edge_tts') {
                        this.switchEdgeVoice(speakerId);
                    } else {
                        this.switchTTSSpeaker(speakerId);
                    }
                    
                    // Auto-demo the voice with a greeting
                    const speakerName = e.target.options[e.target.selectedIndex].text.split(' (')[0]; // Remove locale info for demo
                    setTimeout(() => {
                        this.demoTTSSpeaker(speakerId, speakerName);
                    }, 500);
                }
            };
            
        } else if (speakerSelect) {
            // Show message if no speakers available
            speakerSelect.innerHTML = '<option value="">No voices available</option>';
            speakerSelect.disabled = true;
            
            if (currentSpeakerInfo) {
                currentSpeakerInfo.style.display = 'none';
            }
        }
    }
    
    switchTTSSpeaker(speakerId) {
        this.sendSystemMessage('switch_tts_speaker', { speaker_id: speakerId });
    }
    
    testTTSSpeaker(speakerId = null) {
        this.sendSystemMessage('test_tts', { speaker_id: speakerId });
    }
    
    demoTTSSpeaker(speakerId, speakerName) {
        // Send a demo message with the speaker's name
        const demoText = `Hello, this is ${speakerName}`;
        this.sendSystemMessage('demo_tts', { 
            speaker_id: speakerId,
            text: demoText
        });
    }
    
    updateConnectionStatus(connected) {
        const connectionStatus = document.getElementById('connection-status');
        const voiceInputToggle = document.getElementById('voice-input-toggle');
        
        if (connectionStatus) {
            connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
            connectionStatus.className = 'connection-status ' + (connected ? 'connected' : '');
        }
        
        if (voiceInputToggle) {
            voiceInputToggle.disabled = !connected;
        }
    }
    
    async startVoiceInput() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showNotification('Voice input not supported in this browser', 'error');
            return;
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.processVoiceInput(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Show recording modal
            document.getElementById('voice-modal').classList.add('active');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showNotification('Could not access microphone', 'error');
        }
    }
    
    stopVoiceInput() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            document.getElementById('voice-modal').classList.remove('active');
        }
    }
    
    async processVoiceInput(audioBlob) {
        // For now, we'll just show a placeholder message
        // In a full implementation, this would send the audio to the server
        this.showNotification('Voice input feature coming soon!', 'info');
        
        // Placeholder: simulate STT
        document.getElementById('message-input').value = "[Voice input - STT integration pending]";
    }
    
    speakText(text) {
        // Fallback to browser TTS if backend not available
        if ('speechSynthesis' in window) {
            // Stop any current speech
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }
    }
    
    speakTextBackend(text) {
        // Use backend TTS with the selected speaker from control panel
        this.sendSystemMessage('speak_assistant_response', { text: text });
    }
    
    toggleVoiceOutput(e) {
        this.voiceOutputEnabled = !this.voiceOutputEnabled;
        const icon = e.currentTarget.querySelector('i');
        
        if (this.voiceOutputEnabled) {
            icon.classList.remove('fa-volume-xmark');
            icon.classList.add('fa-volume-high');
            e.currentTarget.style.color = '';
        } else {
            icon.classList.remove('fa-volume-high');
            icon.classList.add('fa-volume-xmark');
            e.currentTarget.style.color = 'var(--error-color)';
        }
        
        // Stop current speech
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
    }
    
    stopCurrentSpeech() {
        // Stop browser-based TTS
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        // Backend TTS is stopped server-side when processing new message
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 8px;
            background: ${type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--error-color)' : 'var(--primary-color)'};
            color: white;
            font-weight: 500;
            z-index: 9999;
            animation: slideIn 0.3s ease;
            box-shadow: var(--shadow-lg);
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    adjustTextareaHeight() {
        const textarea = document.getElementById('message-input');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    scrollToBottom() {
        if (this.autoscrollEnabled) {
            const container = document.querySelector('.chat-container');
            container.scrollTop = container.scrollHeight;
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatMessage(text) {
        // Convert URLs to links
        text = text.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" target="_blank" style="color: var(--primary-color);">$1</a>'
        );
        
        // Convert newlines to <br>
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    formatToolName(tool) {
        return tool
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.talkieApp = new TalkieApp();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .tool-calls {
        background: rgba(99, 102, 241, 0.1);
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 8px;
    }
`;
document.head.appendChild(style);
