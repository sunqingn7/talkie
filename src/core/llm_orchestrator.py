import asyncio
from typing import Dict, List, Any, Optional
from .llm_factory import LLMFactory
from .llm_providers.base import LLMProviderBase


class LLMOrchestrator:
    """
    Orchestrates multi-LLM system with main, fallback, and agents.
    
    Features:
    - Auto-detects available agents on startup
    - Uses fallback when main fails
    - Parallel agent execution for responsiveness
    - MCP tools available to all LLMs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.settings = config.get("settings", {})
        
        self.main_provider: Optional[LLMProviderBase] = None
        self.fallback_provider: Optional[LLMProviderBase] = None
        self.agents: Dict[str, LLMProviderBase] = {}
        self.available_agents: List[str] = []
        
        self.parallel_agents = self.settings.get("parallel_agents", True)
        self.auto_detect_agents = self.settings.get("auto_detect_agents", True)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize providers from config."""
        # Create providers using factory
        providers = LLMFactory.create_from_config(self.config)
        
        self.main_provider = providers.get("main")
        self.fallback_provider = providers.get("fallback")
        self.agents = providers.get("agents", {})
        
        print(f"[LLMOrchestrator] Initialized:")
        print(f"  Main: {self.main_provider.provider_name}/{self.main_provider.model if self.main_provider else 'None'}")
        print(f"  Fallback: {self.fallback_provider.provider_name}/{self.fallback_provider.model if self.fallback_provider else 'None'}")
        print(f"  Agents: {list(self.agents.keys())}")
    
    async def detect_available_agents(self) -> List[str]:
        """Auto-detect which agents are available."""
        available = []
        
        for agent_name, agent in self.agents.items():
            try:
                is_avail = await agent.is_available()
                if is_avail:
                    available.append(agent_name)
                    print(f"[LLMOrchestrator] Agent '{agent_name}' is available")
                else:
                    print(f"[LLMOrchestrator] Agent '{agent_name}' is NOT available")
            except Exception as e:
                print(f"[LLMOrchestrator] Agent '{agent_name}' check failed: {e}")
        
        self.available_agents = available
        return available
    
    async def initialize(self):
        """Async initialization - call detect_available_agents."""
        if self.auto_detect_agents:
            await self.detect_available_agents()
    
    def _get_delegate_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for delegating to agents."""
        return {
            "type": "function",
            "function": {
                "name": "delegate_to_agent",
                "description": "Delegate a task to a specialized LLM agent. Use this when you need expertise beyond your capabilities or when another agent would be more suitable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(self.agents.keys()),
                            "description": "The specialized agent to delegate the task to"
                        },
                        "task": {
                            "type": "string",
                            "description": "Clear description of what you need the agent to do"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context or background information for the agent"
                        }
                    },
                    "required": ["agent", "task"]
                }
            }
        }
    
    def _format_tools_for_llm(self, tools: Dict[str, Any], include_delegate: bool = True) -> List[Dict]:
        """Format tools in OpenAI-compatible format."""
        formatted = []
        
        # Add MCP tools
        for name, tool in tools.items():
            formatted.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        
        # Add delegate tool if agents are available
        if include_delegate and self.available_agents:
            formatted.append(self._get_delegate_tool_definition())
        
        return formatted
    
    async def _call_provider(
        self, 
        provider: LLMProviderBase, 
        messages: List[Dict],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Call a provider with error handling."""
        try:
            return await provider.chat_completion(messages, tools)
        except Exception as e:
            print(f"[LLMOrchestrator] Provider error: {e}")
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def _execute_agent(
        self, 
        agent_name: str, 
        task: str, 
        context: str,
        tools: Dict[str, Any]
    ) -> str:
        """Execute a single agent and return its response."""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Error: Agent '{agent_name}' not found"
        
        # Check if agent is available
        is_avail = await agent.is_available()
        if not is_avail:
            return f"Error: Agent '{agent_name}' is not available"
        
        # Build messages for agent
        user_content = task
        if context:
            user_content = f"Context: {context}\n\nTask: {task}"
        
        messages = agent.prepare_messages(user_content)
        
        # Format tools for agent (no delegate tool for agents)
        formatted_tools = self._format_tools_for_llm(tools, include_delegate=False)
        
        # Call agent
        response = await self._call_provider(agent, messages, formatted_tools)
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or "Agent returned empty response"
    
    async def _execute_agents_parallel(
        self,
        agent_tasks: Dict[str, Dict[str, str]],
        tools: Dict[str, Any]
    ) -> Dict[str, str]:
        """Execute multiple agents in parallel."""
        if not self.parallel_agents:
            # Sequential execution
            results = {}
            for agent_name, task_info in agent_tasks.items():
                results[agent_name] = await self._execute_agent(
                    agent_name,
                    task_info["task"],
                    task_info.get("context", ""),
                    tools
                )
            return results
        
        # Parallel execution
        tasks = []
        for agent_name, task_info in agent_tasks.items():
            task = self._execute_agent(
                agent_name,
                task_info["task"],
                task_info.get("context", ""),
                tools
            )
            tasks.append((agent_name, task))
        
        # Wait for all to complete
        results = {}
        completed = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        
        for i, (agent_name, _) in enumerate(tasks):
            result = completed[i]
            if isinstance(result, Exception):
                results[agent_name] = f"Error: {str(result)}"
            else:
                results[agent_name] = result
        
        return results
    
    def _parse_delegation(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to detect if delegation is needed.
        This is a simple implementation - could be enhanced with function calling.
        """
        # Look for delegation patterns in response
        # In a more advanced version, we'd use actual tool calling
        # For now, this is a placeholder that checks for delegation keywords
        
        if "delegate_to_agent" in llm_response.lower():
            # Try to extract agent name and task
            # This would be better handled via actual function calling
            pass
        
        return None
    
    async def process(
        self,
        user_input: str,
        tools: Dict[str, Any],
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user input.
        
        Flow:
        1. If no agents available, main handles everything
        2. Main LLM processes with tools + delegate tool
        3. If delegation needed, execute agents in parallel
        4. Main LLM synthesizes final response
        """
        
        # Prepare messages
        messages = self.main_provider.prepare_messages(
            user_input, 
            conversation_history,
            system_prompt
        )
        
        # Format tools (include delegate tool if agents available)
        formatted_tools = self._format_tools_for_llm(tools, include_delegate=True)
        
        # Step 1: Call main LLM
        response = await self._call_provider(self.main_provider, messages, formatted_tools)
        
        # Check for errors - use fallback if main fails
        if "error" in response and self.fallback_provider:
            print(f"[LLMOrchestrator] Main failed, trying fallback...")
            response = await self._call_provider(self.fallback_provider, messages, formatted_tools)
            
            if "error" in response:
                return {
                    "success": False,
                    "error": f"Both main and fallback failed: {response.get('error')}",
                    "content": "I apologize, but I'm having trouble processing your request right now."
                }
        
        # Extract content and check for tool calls
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        
        # Step 2: Handle tool calls
        if tool_calls:
            # Process each tool call
            tool_results = []
            agent_tasks = {}  # For parallel execution
            
            for tool_call in tool_calls:
                func_name = tool_call.get("function", {}).get("name", "")
                arguments = tool_call.get("function", {}).get("arguments", {})
                
                # Handle delegation to agent
                if func_name == "delegate_to_agent":
                    agent_name = arguments.get("agent")
                    task = arguments.get("task", "")
                    context = arguments.get("context", "")
                    
                    if agent_name in self.available_agents:
                        agent_tasks[agent_name] = {"task": task, "context": context}
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call.get("id"),
                            "content": f"Agent '{agent_name}' is not available"
                        })
                else:
                    # Regular tool - execute via MCP (placeholder)
                    tool_results.append({
                        "tool_call_id": tool_call.get("id"),
                        "content": f"Tool '{func_name}' would be executed here"
                    })
            
            # Execute agents in parallel if any
            if agent_tasks:
                agent_responses = await self._execute_agents_parallel(agent_tasks, tools)
                
                for agent_name, agent_result in agent_responses.items():
                    tool_results.append({
                        "tool_call_id": f"agent_{agent_name}",
                        "content": f"[{agent_name}]: {agent_result}"
                    })
                
                # Add tool results to messages and get final response
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                messages.append({
                    "role": "system",
                    "content": "Here are the results from tools/agents:\n" + 
                              "\n".join([r["content"] for r in tool_results])
                })
                
                # Get final synthesized response
                final_response = await self._call_provider(self.main_provider, messages, None)
                content = final_response.get("choices", [{}])[0].get("message", {}).get("content", content)
        
        return {
            "success": True,
            "content": content,
            "provider": self.main_provider.provider_name if self.main_provider else "unknown",
            "model": self.main_provider.model if self.main_provider else "unknown",
            "used_agents": list(self.available_agents) if self.available_agents else []
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of all providers."""
        status = {
            "main": {
                "provider": self.main_provider.provider_name if self.main_provider else None,
                "model": self.main_provider.model if self.main_provider else None,
                "available": await self.main_provider.is_available() if self.main_provider else False
            },
            "fallback": {
                "provider": self.fallback_provider.provider_name if self.fallback_provider else None,
                "model": self.fallback_provider.model if self.fallback_provider else None,
                "available": await self.fallback_provider.is_available() if self.fallback_provider else False
            } if self.fallback_provider else None,
            "agents": {},
            "available_agents": self.available_agents,
            "settings": self.settings
        }
        
        # Check each agent
        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = {
                "provider": agent.provider_name,
                "model": agent.model,
                "available": agent_name in self.available_agents
            }
        
        return status
