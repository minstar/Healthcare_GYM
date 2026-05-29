"""Base Environment class for BIOAgents domains.

Manages the domain state, tool execution, and agent interaction protocol.
"""

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB
from bioagents.environment.toolkit import ToolKitBase, ToolDefinition


class ToolCall(BaseModel):
    """Represents a tool call from an agent."""
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool")
    requestor: Literal["user", "assistant"] = Field(default="assistant")


class ToolMessage(BaseModel):
    """Response from a tool execution."""
    id: str = Field(description="Matches the tool call id")
    content: str = Field(description="Result of the tool execution")
    role: str = Field(default="tool")
    error: bool = Field(default=False, description="Whether the tool call resulted in an error")


class Message(BaseModel):
    """A message in the conversation."""
    role: Literal["system", "user", "assistant", "tool"] = Field(description="Message role")
    content: Optional[str] = Field(default=None, description="Message content")
    tool_calls: Optional[list[ToolCall]] = Field(default=None, description="Tool calls")
    tool_call_id: Optional[str] = Field(default=None, description="For tool responses")
    timestamp: Optional[str] = Field(default=None)
    turn_idx: Optional[int] = Field(default=None)


class EnvironmentState(BaseModel):
    """Current state of the environment."""
    messages: list[Message] = Field(default_factory=list)
    turn_count: int = Field(default=0)
    terminated: bool = Field(default=False)
    truncated: bool = Field(default=False)
    reward: float = Field(default=0.0)
    info: dict = Field(default_factory=dict)


class Environment:
    """Base environment for BIOAgents domains.
    
    Manages:
    - Domain policy enforcement
    - Tool execution and state management
    - Conversation history tracking
    - Reward computation interface
    
    Usage:
        db = ClinicalDB.load("data/domains/clinical_diagnosis/db.json")
        tools = ClinicalTools(db)
        env = Environment(
            domain_name="clinical_diagnosis",
            policy=open("policy.md").read(),
            tools=tools,
        )
    """

    # Tools whose execution naturally terminates an episode. Mirrors the
    # AgentRunner's `is_submit` check (agent_runner.py) so the raw gym.Env
    # interface and the runner agree on when an episode ends.
    _TERMINAL_TOOLS = frozenset({"submit_answer", "submit_report"})

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: Optional[ToolKitBase] = None,
        user_tools: Optional[ToolKitBase] = None,
        max_turns: int = 20,
    ):
        self.domain_name = domain_name
        self.policy = policy
        self.tools = tools
        self.user_tools = user_tools
        self.max_turns = max_turns
        self.state = EnvironmentState()
        self._initial_db_hash = tools.get_db_hash() if tools else None

    def reset(self) -> tuple[Optional[str], dict]:
        """Reset the environment to initial state.
        
        Returns:
            observation: Initial observation (system prompt + context)
            info: Dictionary with tools, policy, etc.
        """
        self.state = EnvironmentState()
        
        info = {
            "domain_name": self.domain_name,
            "policy": self.policy,
            "tools": self.tools.get_tool_definitions_dict() if self.tools else [],
            "max_turns": self.max_turns,
        }
        
        return None, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Execute an agent action (message or tool call).
        
        Args:
            action: Either a text message or a JSON tool call string
            
        Returns:
            observation: Result of the action
            reward: Step reward (0.0 during interaction, final reward at end)
            terminated: Whether the episode ended naturally
            truncated: Whether the episode was cut short (max turns)
            info: Additional information
        """
        self.state.turn_count += 1
        
        # Try to parse as tool call
        tool_response = self._try_tool_call(action)
        
        if tool_response is not None:
            # It was a tool call
            observation = tool_response.content
            self.state.messages.append(Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(
                    id=tool_response.id,
                    name=self._last_tool_name,
                    arguments=self._last_tool_args,
                )],
                turn_idx=self.state.turn_count,
            ))
            self.state.messages.append(Message(
                role="tool",
                content=tool_response.content,
                tool_call_id=tool_response.id,
                turn_idx=self.state.turn_count,
            ))
        else:
            # It was a regular message
            observation = action
            self.state.messages.append(Message(
                role="assistant",
                content=action,
                turn_idx=self.state.turn_count,
            ))
        
        # Terminating tools (submit_answer / submit_report) end the episode.
        # Without this, the Gymnasium `terminated` flag never fires on submit
        # and only `truncated` (max_turns) can end an episode.
        if tool_response is not None and self._last_tool_name in self._TERMINAL_TOOLS:
            self.state.terminated = True

        # Check termination conditions
        terminated = self.state.terminated
        truncated = self.state.turn_count >= self.max_turns
        
        reward = 0.0
        info = {
            "turn_count": self.state.turn_count,
            "tool_response": tool_response.model_dump() if tool_response else None,
        }
        
        return observation, reward, terminated, truncated, info

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.state.messages.append(Message(
            role="user",
            content=content,
            turn_idx=self.state.turn_count,
        ))

    def get_observation(self) -> str:
        """Get the current conversation as a formatted string."""
        parts = []
        for msg in self.state.messages:
            if msg.content:
                parts.append(f"{msg.role}: {msg.content}")
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(f"assistant: {tc.name}({json.dumps(tc.arguments)})")
        return "\n".join(parts)

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the agent."""
        if self.tools is None:
            return []
        return self.tools.get_tool_definitions_dict()

    def execute_tool(self, tool_name: str, **kwargs) -> ToolMessage:
        """Execute a tool and return the response."""
        call_id = f"call_{self.state.turn_count}_{tool_name}"
        
        try:
            result = self.tools.use_tool(tool_name, **kwargs)
            content = self._to_json_str(result)
            return ToolMessage(id=call_id, content=content, error=False)
        except Exception as e:
            return ToolMessage(id=call_id, content=f"Error: {e}", error=True)

    def _try_tool_call(self, action: str) -> Optional[ToolMessage]:
        """Try to parse and execute a tool call from the action string."""
        self._last_tool_name = None
        self._last_tool_args = {}
        
        # Try JSON format: {"name": "tool_name", "arguments": {...}}
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_name = parsed["name"]
                arguments = parsed.get("arguments", {})
                self._last_tool_name = tool_name
                self._last_tool_args = arguments
                return self.execute_tool(tool_name, **arguments)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try functional format: tool_name(arg1='val1', arg2='val2')
        import re
        match = re.match(r'(\w+)\((.*)\)$', action.strip(), re.DOTALL)
        if match and self.tools and self.tools.has_tool(match.group(1)):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            try:
                # Parse keyword arguments
                if args_str:
                    # Safely evaluate the arguments
                    arguments = {}
                    # Simple parsing for key=value pairs
                    for arg_match in re.finditer(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", args_str):
                        key = arg_match.group(1)
                        val = arg_match.group(2).strip().strip("'\"")
                        arguments[key] = val
                else:
                    arguments = {}
                self._last_tool_name = tool_name
                self._last_tool_args = arguments
                return self.execute_tool(tool_name, **arguments)
            except Exception:
                pass
        
        return None

    @staticmethod
    def _to_json_str(resp: Any) -> str:
        """Convert a response to a JSON string."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, BaseModel):
            return json.dumps(resp.model_dump(), default=str, ensure_ascii=False)
        if isinstance(resp, (list, dict)):
            return json.dumps(resp, default=str, ensure_ascii=False)
        return str(resp)

    def get_state_snapshot(self) -> dict:
        """Get a snapshot of the current environment state for logging."""
        return {
            "domain_name": self.domain_name,
            "turn_count": self.state.turn_count,
            "messages": [m.model_dump() for m in self.state.messages],
            "terminated": self.state.terminated,
            "truncated": self.state.truncated,
            "db_hash": self.tools.get_db_hash() if self.tools else None,
            "timestamp": datetime.now().isoformat(),
        }
