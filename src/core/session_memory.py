"""
Session Memory - Persistent conversation and file tracking for context awareness.

Records all chats, file uploads, and interactions to temp files for later retrieval.
Enables context-aware queries like "let's redo", "read the file I just uploaded", etc.
"""

import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import uuid


class SessionMemory:
    """
    Persistent session memory for tracking conversations and file uploads.
    
    Stores:
    - All chat messages with timestamps
    - File uploads with metadata and content references
    - Tool calls and their results
    - Session context and metadata
    """
    
    def __init__(self, session_id: Optional[str] = None, memory_dir: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.memory_dir = Path(memory_dir or os.path.join(tempfile.gettempdir(), 'talkie_sessions'))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.session_file = self.memory_dir / f"{self.session_id}.json"
        self.attachments_dir = self.memory_dir / f"{self.session_id}_attachments"
        self.attachments_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self.messages: List[Dict[str, Any]] = []
        self.attachments: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        
        # Session metadata
        self.started_at = datetime.now().isoformat()
        self.last_activity = self.started_at
        
        # Load existing session if present
        self._load_session()
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"talkie_{timestamp}_{unique_id}"
    
    def _load_session(self):
        """Load existing session data from disk."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                    self.attachments = data.get('attachments', [])
                    self.context = data.get('context', {})
                    self.started_at = data.get('started_at', self.started_at)
                    print(f"[SessionMemory] Loaded session: {self.session_id} ({len(self.messages)} messages, {len(self.attachments)} attachments)")
            except Exception as e:
                print(f"[SessionMemory] Warning: Failed to load session: {e}")
    
    def _save_session(self):
        """Save session data to disk."""
        try:
            data = {
                'session_id': self.session_id,
                'started_at': self.started_at,
                'last_activity': datetime.now().isoformat(),
                'messages': self.messages,
                'attachments': self.attachments,
                'context': self.context
            }
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SessionMemory] Warning: Failed to save session: {e}")
    
    def record_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Record a chat message.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (tool_calls, etc.)
        """
        message = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now().isoformat()
        
        # Auto-save every 5 messages
        if len(self.messages) % 5 == 0:
            self._save_session()
    
    def record_attachment(self, filename: str, file_type: str, content: Optional[str] = None,
                         file_path: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Record a file attachment.
        
        Args:
            filename: Original filename
            file_type: File type (pdf, text, etc.)
            content: File content (for text files)
            file_path: Path to saved file
            metadata: Additional metadata
            
        Returns:
            attachment_id: Unique ID for this attachment
        """
        attachment_id = str(uuid.uuid4())
        
        # Save content to separate file if provided
        content_ref = None
        if content and len(content) > 1000:  # Only save large content separately
            content_file = self.attachments_dir / f"{attachment_id}_content.txt"
            try:
                with open(content_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                content_ref = str(content_file)
            except Exception as e:
                print(f"[SessionMemory] Warning: Failed to save attachment content: {e}")
        
        attachment = {
            'id': attachment_id,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'filename': filename,
            'file_type': file_type,
            'file_path': file_path,
            'content_ref': content_ref,
            'content_preview': content[:500] if content else None,
            'metadata': metadata or {}
        }
        
        self.attachments.append(attachment)
        self.last_activity = datetime.now().isoformat()
        self._save_session()
        
        return attachment_id
    
    def record_action(self, action_type: str, description: str, result: Optional[str] = None,
                     metadata: Optional[Dict] = None):
        """
        Record a system action (tool call, file operation, etc.).
        
        Args:
            action_type: Type of action (tool_call, file_read, etc.)
            description: Human-readable description
            result: Action result
            metadata: Additional metadata
        """
        action = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'type': action_type,
            'description': description,
            'result': result,
            'metadata': metadata or {}
        }
        
        if 'actions' not in self.context:
            self.context['actions'] = []
        self.context['actions'].append(action)
        
        self.last_activity = datetime.now().isoformat()
        self._save_session()
    
    def search_messages(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search messages by content.
        
        Args:
            query: Search query (case-insensitive)
            limit: Maximum results to return
            
        Returns:
            List of matching messages
        """
        query_lower = query.lower()
        results = []
        
        for msg in reversed(self.messages):  # Most recent first
            if query_lower in msg['content'].lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_recent_messages(self, count: int = 5) -> List[Dict]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def get_recent_attachments(self, count: int = 3) -> List[Dict]:
        """Get the most recent attachments."""
        return self.attachments[-count:] if self.attachments else []
    
    def find_attachment_by_name(self, filename_pattern: str) -> Optional[Dict]:
        """
        Find an attachment by filename pattern.
        
        Args:
            filename_pattern: Partial filename to match
            
        Returns:
            Matching attachment or None
        """
        pattern_lower = filename_pattern.lower()
        
        # Search from most recent
        for attachment in reversed(self.attachments):
            if pattern_lower in attachment['filename'].lower():
                return attachment
        
        return None
    
    def get_attachment_content(self, attachment_id: str) -> Optional[str]:
        """
        Get full content of an attachment.
        
        Args:
            attachment_id: Attachment ID
            
        Returns:
            Content string or None
        """
        attachment = next((a for a in self.attachments if a['id'] == attachment_id), None)
        if not attachment:
            return None
        
        # Try to read from content reference
        if attachment.get('content_ref') and os.path.exists(attachment['content_ref']):
            try:
                with open(attachment['content_ref'], 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"[SessionMemory] Warning: Failed to read attachment content: {e}")
        
        # Return preview if available
        return attachment.get('content_preview')
    
    def get_last_attachment(self) -> Optional[Dict]:
        """Get the most recent attachment."""
        return self.attachments[-1] if self.attachments else None
    
    def get_last_user_request(self) -> Optional[Dict]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg['role'] == 'user':
                return msg
        return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            'session_id': self.session_id,
            'started_at': self.started_at,
            'last_activity': self.last_activity,
            'message_count': len(self.messages),
            'attachment_count': len(self.attachments),
            'recent_topics': self._extract_topics(),
            'recent_attachments': [
                {'filename': a['filename'], 'type': a['file_type']}
                for a in self.get_recent_attachments(3)
            ]
        }
    
    def _extract_topics(self, count: int = 3) -> List[str]:
        """Extract recent topics from messages."""
        topics = []
        
        # Look for keywords in recent messages
        keywords = ['weather', 'file', 'read', 'write', 'search', 'calculate', 'timer']
        
        for msg in reversed(self.messages[-20:]):  # Look at last 20 messages
            content_lower = msg['content'].lower()
            for keyword in keywords:
                if keyword in content_lower and keyword not in topics:
                    topics.append(keyword)
                    if len(topics) >= count:
                        break
            if len(topics) >= count:
                break
        
        return topics
    
    def set_context_value(self, key: str, value: Any):
        """Set a context value."""
        self.context[key] = value
        self._save_session()
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)
    
    def list_all_sessions(self) -> List[Dict]:
        """List all available sessions."""
        sessions = []
        
        for session_file in self.memory_dir.glob("*.json"):
            if session_file.name.endswith('_attachments'):
                continue
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id'),
                        'started_at': data.get('started_at'),
                        'last_activity': data.get('last_activity'),
                        'message_count': len(data.get('messages', [])),
                        'attachment_count': len(data.get('attachments', []))
                    })
            except:
                pass
        
        return sorted(sessions, key=lambda x: x.get('last_activity', ''), reverse=True)
    
    def clear(self):
        """Clear current session data."""
        self.messages = []
        self.attachments = []
        self.context = {}
        self._save_session()
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than specified hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for session_file in self.memory_dir.glob("*.json"):
            try:
                file_age = current_time - session_file.stat().st_mtime
                if file_age > max_age_seconds:
                    # Load to get session ID for cleanup
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        session_id = data.get('session_id')
                    
                    # Remove session file
                    session_file.unlink()
                    
                    # Remove attachments directory
                    attachments_dir = self.memory_dir / f"{session_id}_attachments"
                    if attachments_dir.exists():
                        import shutil
                        shutil.rmtree(attachments_dir)
                    
                    print(f"[SessionMemory] Cleaned up old session: {session_id}")
            except Exception as e:
                print(f"[SessionMemory] Warning: Failed to cleanup session: {e}")


# Singleton instance for global access
_session_memory_instance: Optional[SessionMemory] = None


def get_session_memory(session_id: Optional[str] = None) -> SessionMemory:
    """Get or create the global session memory instance."""
    global _session_memory_instance
    if _session_memory_instance is None or (session_id and _session_memory_instance.session_id != session_id):
        _session_memory_instance = SessionMemory(session_id)
    return _session_memory_instance


def reset_session_memory():
    """Reset the global session memory instance."""
    global _session_memory_instance
    _session_memory_instance = None
