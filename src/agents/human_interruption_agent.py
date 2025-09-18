from typing import Any, Generator, Optional, Sequence, Union, List
import uuid
import os
import pathlib
import sys
import json
from psycopg import Connection

agent_root = pathlib.Path(__file__).resolve().parent
src_path = agent_root / 'code'  #for agent deployment
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from src.lakebase.database import LakebaseDatabase

mlflow.autolog()

default_system_prompt="""You are a helpful assistant. Answer questions directly based on the conversation history.
Use tools when specifically prompted by a user.
For questions about previous conversation, use the conversation history available to you.

IMPORTANT: When you need to use tools to help the user, you will prepare the tool call and then pause for approval. 
When this happens, inform the user that you're ready to execute the tool and are awaiting their approval. 
Let them know they need to approve the action before the tool can run."""

class LangGraphChatAgent(ChatAgent):
    def __init__(self, conn_string: str, model: LanguageModelLike, tools: Union[Sequence[BaseTool], ToolNode], require_human_approval: bool = True):
        """
        Initialize with connection string, model, and tools.
        
        Args:
            conn_string: Database connection string
            model: Language model
            tools: Tools to use
            require_human_approval: If True, require human approval before tool execution
        """
        self.conn_string = conn_string
        self.model = model
        self.tools = tools
        self.thread_id = "default"
        self.require_human_approval = require_human_approval

    def _convert_messages_to_dict(self, messages: List[Union[ChatAgentMessage, dict]]) -> List[dict]:
        """
        Convert messages to dict format, handling both ChatAgentMessage objects and dicts.
        Override the parent method to handle dict inputs properly.
        """
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                # Already a dict, ensure it has an ID
                if "id" not in msg:
                    msg["id"] = str(uuid.uuid4())
                result.append(msg)
            elif hasattr(msg, 'model_dump_compat'):
                # MLflow ChatAgentMessage with model_dump_compat method
                msg_dict = msg.model_dump_compat(exclude_none=True)
                if "id" not in msg_dict:
                    msg_dict["id"] = str(uuid.uuid4())
                result.append(msg_dict)
            elif hasattr(msg, 'model_dump'):
                # Pydantic v2 style
                msg_dict = msg.model_dump(exclude_none=True)
                if "id" not in msg_dict:
                    msg_dict["id"] = str(uuid.uuid4())
                result.append(msg_dict)
            elif hasattr(msg, 'dict'):
                # Pydantic v1 style
                msg_dict = msg.dict(exclude_none=True)
                if "id" not in msg_dict:
                    msg_dict["id"] = str(uuid.uuid4())
                result.append(msg_dict)
            else:
                # Fallback: try to extract role and content
                result.append({
                    "id": getattr(msg, 'id', str(uuid.uuid4())),
                    "role": getattr(msg, 'role', 'user'),
                    "content": getattr(msg, 'content', str(msg))
                })
        return result

    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # Extract thread_id from custom_inputs if available, otherwise generate new
        if custom_inputs and "thread_id" in custom_inputs:
            thread_id = custom_inputs["thread_id"]
            # For continuing conversations, only send the new message
            # The checkpointer will automatically load previous messages
            messages_to_send = [messages[-1]] if messages else []
        else:
            thread_id = str(uuid.uuid4())
            # For new conversations, send all messages
            messages_to_send = messages
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Track interruption state
        is_interrupted = False
        interrupt_data = None
        pending_tool_calls = None
        
        # Connect and create checkpointer within context manager
        with Connection.connect(self.conn_string) as conn:
            checkpointer = PostgresSaver(conn)
            
            # Create the agent fresh with the checkpointer
            agent = self._create_agent_with_checkpointer(checkpointer)
            
            # Convert messages for the request
            converted_messages = self._convert_messages_to_dict(messages_to_send)
            
            # Invoke the agent
            result = agent.invoke({"messages": converted_messages}, config)
            
            # Check the current state for interruption
            state = agent.get_state(config)
            if state.next:
                # If there are next nodes and we're interrupted before tools
                if "tools" in state.next and self.require_human_approval:
                    is_interrupted = True
                    # Get the last message which should have tool calls
                    last_message = result["messages"][-1] if result.get("messages") else None
                    if last_message:
                        # Extract tool calls from the last message
                        if hasattr(last_message, 'tool_calls'):
                            pending_tool_calls = last_message.tool_calls
                        elif isinstance(last_message, dict):
                            pending_tool_calls = last_message.get('tool_calls')
                        else:
                            pending_tool_calls = getattr(last_message, 'additional_kwargs', {}).get('tool_calls')
                        
                        # Create interrupt data in expected format
                        interrupt_data = [{
                            "value": {
                                "tool_calls": pending_tool_calls,
                                "message": "Tool execution requires approval"
                            }
                        }]
            
            # Parse messages
            out_messages = []
            if result.get("messages"):
                for msg in result["messages"]:
                    out_messages.append(self._parse_message(msg))
            
            # Add a helpful message about approval if interrupted
            if is_interrupted and pending_tool_calls:
                # Extract tool details for the message
                tool_info = []
                for tool_call in pending_tool_calls:
                    if isinstance(tool_call, dict):
                        func_name = tool_call.get('function', {}).get('name', 'Unknown')
                        func_args = tool_call.get('function', {}).get('arguments', '{}')
                    else:
                        func_name = getattr(tool_call, 'name', 'Unknown')
                        func_args = getattr(tool_call, 'arguments', '{}')
                    tool_info.append(f"• Function: `{func_name}`\n• Arguments: `{func_args}`")
                
                approval_message = ChatAgentMessage(
                    role='assistant',
                    content=(
                        f"📋 **Tool Approval Required**\n\n"
                        f"I'm ready to execute the following tool:\n"
                        f"{tool_info[0] if tool_info else 'Tool details unavailable'}\n\n"
                        f"Please approve this action to continue, or provide alternative instructions."
                    ),
                    id=str(uuid.uuid4())
                )
                out_messages.append(approval_message)
        
        # Build custom_outputs with interruption info
        custom_outputs = {
            "thread_id": thread_id,
            "is_interrupted": is_interrupted,
            "interrupt_data": interrupt_data[0] if interrupt_data else None,
            "requires_approval": is_interrupted and self.require_human_approval,
            "pending_tool_calls": pending_tool_calls,
            "next_action": "approve_tools" if is_interrupted else "complete"
        }
        
        # Return ChatAgentResponse with interruption state
        try:
            return ChatAgentResponse(
                messages=out_messages,
                custom_outputs=custom_outputs
            )
        except TypeError:
            response = ChatAgentResponse(messages=out_messages)
            response.custom_outputs = custom_outputs
            return response
    
    def _parse_message(self, msg) -> ChatAgentMessage:
        """Helper to parse different message types into ChatAgentMessage"""
        if isinstance(msg, dict):
            return ChatAgentMessage(
                id=msg.get("id", str(uuid.uuid4())),
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
                name=msg.get("name"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                attachments=msg.get("attachments")
            )
        else:
            # Handle LangChain message objects
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
            
            content = getattr(msg, 'content', '')
            
            # Determine role based on message type
            if isinstance(msg, AIMessage):
                role = 'assistant'
            elif isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, SystemMessage):
                role = 'system'
            elif isinstance(msg, ToolMessage):
                role = 'tool'
            else:
                role = getattr(msg, 'role', 'assistant')
            
            # Handle tool calls - ensure we get them properly
            tool_calls = None
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Convert tool calls to dict format if needed
                if isinstance(msg.tool_calls, list):
                    tool_calls = []
                    for tc in msg.tool_calls:
                        if hasattr(tc, 'dict'):
                            tool_calls.append(tc.dict())
                        elif isinstance(tc, dict):
                            tool_calls.append(tc)
                        else:
                            # Try to construct dict from attributes
                            tool_calls.append({
                                'id': getattr(tc, 'id', str(uuid.uuid4())),
                                'type': getattr(tc, 'type', 'function'),
                                'function': getattr(tc, 'function', {})
                            })
                else:
                    tool_calls = msg.tool_calls
            elif hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('tool_calls'):
                tool_calls = msg.additional_kwargs.get('tool_calls')
            
            return ChatAgentMessage(
                id=getattr(msg, 'id', str(uuid.uuid4())),
                role=role,
                content=content,
                name=getattr(msg, 'name', None),
                tool_calls=tool_calls,
                tool_call_id=getattr(msg, 'tool_call_id', None),
                attachments=getattr(msg, 'attachments', None)
            )
    
    def _create_agent_with_checkpointer(self, checkpointer: PostgresSaver) -> CompiledStateGraph:
        """
        Create the agent with the given checkpointer using proper LangGraph interrupt pattern.
        """
        # Store the model with tools bound
        model_with_tools = self.model.bind_tools(self.tools) if self.tools else self.model

        def should_continue(state: ChatAgentState):
            messages = state["messages"]
            last = messages[-1]
            # If there are any tool_calls on the last message, continue to tools
            if hasattr(last, 'tool_calls'):
                return "continue" if last.tool_calls else "end"
            elif isinstance(last, dict):
                return "continue" if last.get("tool_calls") else "end"
            else:
                additional_kwargs = getattr(last, 'additional_kwargs', {})
                return "continue" if additional_kwargs.get("tool_calls") else "end"

        def call_model(state: ChatAgentState, config: RunnableConfig):
            messages_with_system = state["messages"]
            if default_system_prompt:
                if not messages_with_system or messages_with_system[0].get("role") != "system":
                    messages_with_system = [{"role": "system", "content": default_system_prompt}] + messages_with_system
            
            # Use model_with_tools instead of self.model
            response = model_with_tools.invoke(messages_with_system, config)
            return {"messages": [response]}

        # Build the graph
        workflow = StateGraph(ChatAgentState)
        workflow.add_node("agent", call_model)
        
        if self.tools:
            # Add tools node
            workflow.add_node("tools", ChatAgentToolNode(self.tools))
            workflow.set_entry_point("agent")
            
            # Agent -> Tools (conditionally) -> Agent
            workflow.add_conditional_edges(
                "agent", 
                should_continue, 
                {"continue": "tools", "end": END}
            )
            workflow.add_edge("tools", "agent")
            
            if self.require_human_approval:
                # Compile with interrupt before tools for human approval
                graph = workflow.compile(
                    checkpointer=checkpointer,
                    interrupt_before=["tools"]  # Interrupt BEFORE tool execution
                )
            else:
                # No interruption - direct tool execution
                graph = workflow.compile(checkpointer=checkpointer)
        else:
            # No tools
            workflow.set_entry_point("agent")
            workflow.add_edge("agent", END)
            graph = workflow.compile(checkpointer=checkpointer)
        
        return graph

    def predict_stream(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # Extract thread_id from custom_inputs if available
        if custom_inputs and "thread_id" in custom_inputs:
            thread_id = custom_inputs["thread_id"]
            messages = [messages[-1]] if messages else []
        else:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        
        with Connection.connect(self.conn_string) as conn:
            checkpointer = PostgresSaver(conn)
            agent = self._create_agent_with_checkpointer(checkpointer)
            
            request = {"messages": self._convert_messages_to_dict(messages)}
            
            # Track if we're interrupted
            is_interrupted = False
            
            for chunk in agent.stream(request, config, stream_mode="values"):
                # Check state for interruption
                state = agent.get_state(config)
                if state.next and "tools" in state.next and self.require_human_approval:
                    is_interrupted = True
                    # Don't stream further if interrupted
                    break
                    
                if chunk.get("messages"):
                    for msg in chunk["messages"]:
                        parsed_msg = self._parse_message(msg)
                        yield ChatAgentChunk(delta=parsed_msg.__dict__)
            
            # If interrupted, yield a final chunk with interrupt info
            if is_interrupted:
                yield ChatAgentChunk(
                    delta={
                        "custom_outputs": {
                            "thread_id": thread_id,
                            "is_interrupted": True,
                            "requires_approval": True,
                            "next_action": "approve_tools"
                        }
                    }
                )

    def resume(self, command_value: Any = None, thread_id: Optional[str] = None) -> ChatAgentResponse:
        """
        Resume after interruption.
        
        Args:
            command_value: Value to pass when resuming.
                          If None or "approved", continues with tool execution.
                          If "rejected", skips tool execution and adds rejection message.
                          Any other string provides custom feedback.
            thread_id: Thread ID to resume
        """
        thread_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        is_interrupted = False
        interrupt_data = None
        pending_tool_calls = None
        
        with Connection.connect(self.conn_string) as conn:
            checkpointer = PostgresSaver(conn)
            agent = self._create_agent_with_checkpointer(checkpointer)
            
            # Handle different command values
            if command_value == "rejected":
                # Update the state to add a rejection message and clear tool calls
                state = agent.get_state(config)
                # Add a system message about rejection
                rejection_message = {
                    "role": "assistant",
                    "content": "I understand. I won't execute those tools. How else can I help you?",
                    "id": str(uuid.uuid4())
                }
                # Update state to skip tools
                agent.update_state(
                    config,
                    {"messages": [rejection_message]},
                    as_node="tools"  # Pretend we're the tools node to skip it
                )
                # Now continue execution from after tools
                result = agent.invoke(None, config)
            elif command_value is None or command_value == "approved":
                # Continue normally - tools will execute
                result = agent.invoke(None, config)
            else:
                # Custom feedback - add as user message and continue
                feedback_message = {
                    "role": "user",
                    "content": str(command_value),
                    "id": str(uuid.uuid4())
                }
                result = agent.invoke({"messages": [feedback_message]}, config)
            
            # Check if still interrupted (in case there are more tool calls)
            state = agent.get_state(config)
            if state.next:
                if "tools" in state.next and self.require_human_approval:
                    is_interrupted = True
                    last_message = result["messages"][-1] if result.get("messages") else None
                    if last_message:
                        if hasattr(last_message, 'tool_calls'):
                            pending_tool_calls = last_message.tool_calls
                        elif isinstance(last_message, dict):
                            pending_tool_calls = last_message.get('tool_calls')
                        
                        interrupt_data = [{
                            "value": {
                                "tool_calls": pending_tool_calls,
                                "message": "Tool execution requires approval"
                            }
                        }]
            
            # Parse messages
            out_messages = []
            if result.get("messages"):
                for msg in result["messages"]:
                    out_messages.append(self._parse_message(msg))
        
        # Build response with state info
        custom_outputs = {
            "thread_id": thread_id,
            "is_interrupted": is_interrupted,
            "interrupt_data": interrupt_data[0] if interrupt_data else None,
            "requires_approval": is_interrupted and self.require_human_approval,
            "pending_tool_calls": pending_tool_calls,
            "next_action": "approve_tools" if is_interrupted else "complete"
        }
        
        try:
            return ChatAgentResponse(
                messages=out_messages,
                custom_outputs=custom_outputs
            )
        except TypeError:
            response = ChatAgentResponse(messages=out_messages)
            response.custom_outputs = custom_outputs
            return response


# --------------------------------------------------------------------------------------
# Updated factory to pass connection string to agent
# --------------------------------------------------------------------------------------
class HumanInterruptionAgent:
    """
    Factory for building a UC-tool-calling LangGraph agent with human-in-the-loop.
    """

    DEFAULT_UC_FUNCTIONS = ["system.ai.python_exec"]

    @staticmethod
    def create(
        *,
        endpoint_name: str,
        conn_string: Optional[str] = None,
        uc_function_names: Optional[Sequence[str]] = None,
        databricks_function_client: Optional[DatabricksFunctionClient] = None,
        require_human_approval: bool = False
    ) -> ChatAgent:
        """
        Build and return a ready-to-use ChatAgent that can be passed to MLflow or called directly.
        
        Args:
            endpoint_name: Name of the Databricks model serving endpoint
            conn_string: Database connection string
            uc_function_names: List of Unity Catalog function names to use as tools
            databricks_function_client: Optional Databricks function client
            require_human_approval: If True, require human approval before tool execution
        """
        # Get connection string - either from parameter or environment
        if not conn_string:
            lakebase_db = LakebaseDatabase(host=os.getenv("DATABRICKS_HOST"))
            conn_string = lakebase_db.initialize_connection(
                user=os.getenv("DATABRICKS_CLIENT_ID"), 
                instance_name=os.getenv("LAKEBASE_INSTANCE")
            )
        
        # Wire UC function client once
        client = databricks_function_client or DatabricksFunctionClient()
        set_uc_function_client(client)

        # LLM
        llm = ChatDatabricks(endpoint=endpoint_name)

        # Tools from UC
        tool_names = list(uc_function_names or HumanInterruptionAgent.DEFAULT_UC_FUNCTIONS)
        uc_toolkit = UCFunctionToolkit(function_names=tool_names)
        tools = uc_toolkit.tools  # type: ignore

        # Return agent with connection string, model, and tools
        return LangGraphChatAgent(conn_string, llm, tools, require_human_approval)

    @staticmethod
    def from_defaults(conn_string: Optional[str] = None, require_human_approval: bool = False) -> ChatAgent:
        """
        Convenience constructor with your existing default endpoint and prompt.
        """
        LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
        return HumanInterruptionAgent.create(
            endpoint_name=LLM_ENDPOINT_NAME,
            conn_string=conn_string,
            uc_function_names=HumanInterruptionAgent.DEFAULT_UC_FUNCTIONS,
            require_human_approval=require_human_approval
        )


# Create the agent instance for MLflow
from mlflow import models as _models

# Get connection string from environment or use LakebaseDatabase
conn_string = os.getenv("DB_CONNECTION_STRING")
if not conn_string:
    lakebase_db = LakebaseDatabase(host=os.getenv("DATABRICKS_HOST"))
    conn_string = lakebase_db.initialize_connection(
        user=os.getenv("DATABRICKS_CLIENT_ID"),
        instance_name=os.getenv("LAKEBASE_INSTANCE")
    )

# Check if human approval should be required from environment
require_approval = os.getenv("REQUIRE_HUMAN_APPROVAL", "true").lower() == "true"

AGENT = HumanInterruptionAgent.create(
    endpoint_name=os.getenv("ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct"),
    conn_string=conn_string,
    uc_function_names=os.getenv("UC_FUNCTIONS", "").split(",") if os.getenv("UC_FUNCTIONS") else None,
    require_human_approval=require_approval
)
mlflow.models.set_model(AGENT)