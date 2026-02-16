"""
Session Memory Tools - Tools for querying conversation history and context.
"""

from typing import Any, Dict, List, Optional
from . import BaseTool


class SearchSessionMemoryTool(BaseTool):
    """
    Search through conversation history to find previous messages or context.
    
    Use this when the user references something from earlier in the conversation
    like "let's redo", "as I mentioned before", or asks about previous topics.
    """
    
    def __init__(self, config: dict, session_memory=None):
        super().__init__(config)
        self.session_memory = session_memory
    
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
    
    def _get_description(self) -> str:
        return (
            "Search through conversation history to find previous messages, "
            "context, or information. Use this when the user references something "
            "from earlier like 'let's redo', 'read the file I just uploaded', "
            "'as I mentioned before', or asks about previous topics."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant messages (e.g., 'weather', 'file uploaded', 'read aloud')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search session memory for matching messages."""
        if not self.session_memory:
            return {
                "success": False,
                "error": "Session memory not available"
            }
        
        results = self.session_memory.search_messages(query, limit=limit)
        
        if not results:
            # Try alternative searches
            alternative_queries = []
            if 'file' in query.lower() or 'upload' in query.lower():
                alternative_queries = ['attached', 'document', 'pdf', 'txt']
            elif 'redo' in query.lower() or 'again' in query.lower():
                alternative_queries = ['previous', 'last request']
            
            for alt_query in alternative_queries:
                alt_results = self.session_memory.search_messages(alt_query, limit=3)
                if alt_results:
                    return {
                        "success": True,
                        "query": query,
                        "results": [],
                        "suggestion": f"No direct matches found, but found related messages about '{alt_query}'",
                        "alternative_results": [
                            {
                                "role": r['role'],
                                "content": r['content'][:200] + "..." if len(r['content']) > 200 else r['content'],
                                "time": r['datetime']
                            }
                            for r in alt_results[:2]
                        ]
                    }
            
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": f"No messages found matching '{query}'. The conversation may have just started."
            }
        
        return {
            "success": True,
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "role": r['role'],
                    "content": r['content'][:300] + "..." if len(r['content']) > 300 else r['content'],
                    "time": r['datetime'],
                    "has_attachment": bool(r.get('metadata', {}).get('attachment_ids'))
                }
                for r in results
            ]
        }


class GetRecentAttachmentsTool(BaseTool):
    """
    Get list of recently uploaded files.
    
    Use this when the user asks about 'the file I uploaded', 'that document', 
    or references a recent attachment without specifying the name.
    """
    
    def __init__(self, config: dict, session_memory=None):
        super().__init__(config)
        self.session_memory = session_memory
    
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
    
    def _get_description(self) -> str:
        return (
            "Get information about recently uploaded files/attachments. "
            "Use this when the user refers to 'the file I uploaded', 'that document', "
            "'the PDF', or any recent attachment without specifying the exact filename."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of recent attachments to retrieve (default: 3)",
                    "default": 3
                }
            },
            "required": []
        }
    
    async def execute(self, count: int = 3) -> Dict[str, Any]:
        """Get recent attachments."""
        if not self.session_memory:
            return {
                "success": False,
                "error": "Session memory not available"
            }
        
        attachments = self.session_memory.get_recent_attachments(count)
        
        if not attachments:
            return {
                "success": True,
                "count": 0,
                "message": "No files have been uploaded in this session yet.",
                "attachments": []
            }
        
        return {
            "success": True,
            "count": len(attachments),
            "attachments": [
                {
                    "id": a['id'],
                    "filename": a['filename'],
                    "type": a['file_type'],
                    "uploaded_at": a['datetime'],
                    "preview": a.get('content_preview', 'No preview available')[:100] + "..." if a.get('content_preview') else 'No preview available',
                    "has_content": a.get('content_ref') is not None or a.get('content_preview') is not None
                }
                for a in attachments
            ]
        }


class GetAttachmentContentTool(BaseTool):
    """
    Get the full content of a specific attachment.
    
    Use this to read the content of a file that was previously uploaded.
    """
    
    def __init__(self, config: dict, session_memory=None):
        super().__init__(config)
        self.session_memory = session_memory
    
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
    
    def _get_description(self) -> str:
        return (
            "Get the full content of a previously uploaded file/attachment. "
            "Use this when you need to read or analyze the content of a file "
            "that was uploaded earlier in the conversation. You can reference "
            "the file by its ID or filename."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "attachment_id": {
                    "type": "string",
                    "description": "The attachment ID (from get_recent_attachments)"
                },
                "filename": {
                    "type": "string",
                    "description": "Partial filename to search for (e.g., 'report.pdf')"
                }
            },
            "required": []
        }
    
    async def execute(self, attachment_id: str = None, filename: str = None) -> Dict[str, Any]:
        """Get attachment content by ID or filename."""
        if not self.session_memory:
            return {
                "success": False,
                "error": "Session memory not available"
            }
        
        attachment = None
        
        # Find by ID if provided
        if attachment_id:
            attachment = next(
                (a for a in self.session_memory.attachments if a['id'] == attachment_id),
                None
            )
        
        # Find by filename if provided
        if not attachment and filename:
            attachment = self.session_memory.find_attachment_by_name(filename)
        
        # Get last attachment as fallback
        if not attachment and not attachment_id and not filename:
            attachment = self.session_memory.get_last_attachment()
        
        if not attachment:
            return {
                "success": False,
                "error": "Attachment not found",
                "message": "Could not find the specified attachment. Use get_recent_attachments to see available files."
            }
        
        # Get content
        content = self.session_memory.get_attachment_content(attachment['id'])
        
        return {
            "success": True,
            "attachment": {
                "id": attachment['id'],
                "filename": attachment['filename'],
                "type": attachment['file_type'],
                "uploaded_at": attachment['datetime']
            },
            "content": content or "Content not available",
            "content_length": len(content) if content else 0
        }


class GetSessionContextTool(BaseTool):
    """
    Get context about the current conversation session.
    
    Use this to understand the overall context: what has been discussed,
    what files are available, recent topics, etc.
    """
    
    def __init__(self, config: dict, session_memory=None):
        super().__init__(config)
        self.session_memory = session_memory
    
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
    
    def _get_description(self) -> str:
        return (
            "Get an overview of the current conversation session including "
            "message count, recent topics, uploaded files, and overall context. "
            "Use this to get a high-level understanding of the conversation history."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Get session context and summary."""
        if not self.session_memory:
            return {
                "success": False,
                "error": "Session memory not available"
            }
        
        summary = self.session_memory.get_session_summary()
        recent_messages = self.session_memory.get_recent_messages(3)
        recent_attachments = self.session_memory.get_recent_attachments(3)
        
        return {
            "success": True,
            "session": summary,
            "recent_messages": [
                {
                    "role": m['role'],
                    "content": m['content'][:150] + "..." if len(m['content']) > 150 else m['content'],
                    "time": m['datetime']
                }
                for m in recent_messages
            ],
            "recent_attachments": [
                {
                    "filename": a['filename'],
                    "type": a['file_type']
                }
                for a in recent_attachments
            ]
        }


class GetLastUserRequestTool(BaseTool):
    """
    Get the last thing the user asked for.
    
    Use this when the user says 'let's redo', 'do that again', 'repeat',
    or references their previous request.
    """
    
    def __init__(self, config: dict, session_memory=None):
        super().__init__(config)
        self.session_memory = session_memory
    
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
    
    def _get_description(self) -> str:
        return (
            "Get the user's last request/message. Use this when the user says "
            "'let's redo', 'do that again', 'repeat', 'I said', or references "
            "their previous request without specifying what it was."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Get the last user message."""
        if not self.session_memory:
            return {
                "success": False,
                "error": "Session memory not available"
            }
        
        last_message = self.session_memory.get_last_user_request()
        
        if not last_message:
            return {
                "success": True,
                "message": "No previous user message found",
                "request": None
            }
        
        return {
            "success": True,
            "request": {
                "content": last_message['content'],
                "time": last_message['datetime'],
                "had_attachments": bool(last_message.get('metadata', {}).get('attachment_ids'))
            }
        }
