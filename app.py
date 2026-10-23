import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from backend import get_all_threads, chatbot, ingest_rag_document

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")


# ---------- helpers ----------
def extract_text(content) -> str:
    """Handle both plain-string and list-of-blocks message content."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type", "text") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def thread_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }


def get_thread_messages(thread_id: str) -> list[dict]:
    """Load a thread's visible messages (user + assistant text) from the checkpointer."""
    try:
        state = chatbot.get_state(thread_config(thread_id))
    except Exception:
        return []
    if not state or not state.values:
        return []

    history = []
    for m in state.values.get("messages", []):
        if m.type == "human":
            role = "user"
        elif m.type == "ai":
            role = "assistant"
        else:  # skip tool messages etc.
            continue
        text = extract_text(m.content).strip()
        if text:  # skip AI messages that are only tool calls
            history.append({"role": role, "content": text})
    return history


def list_saved_threads() -> list[str]:
    """All thread ids in the checkpointer, most recently active first."""
    latest = {}
    try:
        for tup in chatbot.checkpointer.list(None):
            tid = tup.config["configurable"]["thread_id"]
            ts = tup.checkpoint.get("ts", "")
            if tid not in latest or ts > latest[tid]:
                latest[tid] = ts
    except Exception:
        pass
    return [tid for tid, _ in sorted(latest.items(), key=lambda x: x[1], reverse=True)]


def thread_title(thread_id: str) -> str:
    """Title = first user message (cached once found)."""
    cache = st.session_state.titles
    if thread_id in cache:
        return cache[thread_id]
    for msg in get_thread_messages(thread_id):
        if msg["role"] == "user":
            title = msg["content"].replace("\n", " ")
            title = title if len(title) <= 32 else title[:32] + "…"
            cache[thread_id] = title
            return title
    return "New chat"


def start_new_chat():
    st.session_state.thread_id = str(uuid.uuid4())


def index_uploaded_file(uploaded_file):
    """
    Save the uploaded file to a temp file, index it with the RAG backend,
    then delete the temp file.
    ingest_rag_document needs a real file PATH, not a Streamlit UploadedFile.
    """
    temp_path = None
    try:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        return ingest_rag_document(temp_path)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------- shared streaming generator ----------
def stream_graph(payload, config, status_box):
    """
    Stream a LangGraph run (new message OR Command(resume=...)).
    Yields only assistant answer text, and reports tool usage in status_box.
    """
    shown_calls = set()

    for chunk, metadata in chatbot.stream(
        payload,
        config=config,
        stream_mode="messages",
    ):
        # Tool finished
        if isinstance(chunk, ToolMessage):
            tool_name = getattr(chunk, "name", None) or "tool"
            status_box.write(f"✅ Tool `{tool_name}` finished")
            continue

        if not isinstance(chunk, AIMessage):
            continue

        # Tool being called
        tool_calls = getattr(chunk, "tool_calls", []) or []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            if tool_name:
                call_id = tool_call.get("id") or tool_name
                if call_id not in shown_calls:
                    shown_calls.add(call_id)
                    status_box.write(f"🔧 Calling tool: `{tool_name}`")
                    status_box.update(
                        label=f"Using {tool_name}...",
                        state="running",
                        expanded=True,
                    )

        # Skip tool-call chunks (not answer text)
        if tool_calls or getattr(chunk, "tool_call_chunks", None):
            continue

        text = extract_text(chunk.content)
        if text:
            yield text


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """Return the first unresolved LangGraph interrupt for a thread, or None."""
    try:
        state_snapshot = chatbot.get_state(thread_config(thread_id))

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        # Others store interrupts inside tasks
        for task in getattr(state_snapshot, "tasks", ()) or ():
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        # A new thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """Save the pending interrupt information inside Streamlit state."""
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    """
    Keep Streamlit HITL state in sync with the LangGraph checkpoint, so a
    pending approval reappears after a rerun, browser refresh, or when
    switching between chats.
    """
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted graph run.
    decision: "yes" approves, "no" rejects.
    """
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    # The SAME thread id must be used when resuming
    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = thread_config(interrupted_thread_id)
    resume_config["run_name"] = "hitl_resume_trace"

    resumed_ok = False

    with st.chat_message("assistant"):
        status_box = st.status("🔄 Resuming the requested action...", expanded=True)

        try:
            st.write_stream(
                stream_graph(Command(resume=decision), resume_config, status_box)
            )

            # Did the graph interrupt again?
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_box.update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False,
                )
            else:
                st.session_state["pending_hitl"] = None
                status_box.update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False,
                )

            resumed_ok = True

        except Exception as error:
            status_box.update(label="An error occurred", state="error")
            st.error(f"Could not resume the requested action: {error}")

    # Rerun OUTSIDE the try block so the answer shows in normal chat order
    if resumed_ok:
        st.rerun()


# ---------- session state ----------
if "titles" not in st.session_state:
    st.session_state.titles = {}
if "thread_id" not in st.session_state:
    start_new_chat()
if "document_results" not in st.session_state:
    st.session_state["document_results"] = []
if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = set()
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

current_id = st.session_state.thread_id

# Restore any pending approval for THIS chat from the checkpoint
sync_pending_interrupt(current_id)

current_messages = get_thread_messages(current_id)

pending_hitl = st.session_state.get("pending_hitl")
has_pending = bool(pending_hitl and pending_hitl.get("thread_id") == current_id)

# ---------- sidebar ----------
with st.sidebar:
    st.header("💬 Chats")

    # Don't stack up empty chats
    if st.button("➕ New chat", use_container_width=True, disabled=not current_messages):
        start_new_chat()
        st.rerun()

    saved = list_saved_threads()

    # The active thread has no checkpoints until the first message is sent
    if current_id not in saved:
        st.button("✏️ New chat", key="pending", use_container_width=True,
                  type="primary", disabled=True)

    if not saved:
        st.caption("No past chats yet.")

    for tid in saved:
        is_active = tid == current_id
        if st.button(
            thread_title(tid),
            key=f"thread_{tid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state.thread_id = tid
                st.rerun()

# ---------- main UI ----------
st.title("🤖 Agentic Chatbot")

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- HITL approval prompt ----------
if has_pending:
    approval_slot = st.empty()

    with approval_slot.container():
        with st.chat_message("assistant"):
            st.warning(f"⚠️ **Approval required**\n\n{pending_hitl['prompt']}")
            col_yes, col_no = st.columns(2)
            approve_clicked = col_yes.button(
                "✅ Approve",
                key=f"approve_{current_id}",
                use_container_width=True,
                type="primary",
            )
            reject_clicked = col_no.button(
                "❌ Reject",
                key=f"reject_{current_id}",
                use_container_width=True,
            )

    if approve_clicked or reject_clicked:
        approval_slot.empty()  # remove the buttons while resuming
        resume_hitl_execution("yes" if approve_clicked else "no")

# Block new messages while an approval is pending
user_message = st.chat_input(
    "Approve or reject the action above first..." if has_pending else "Type here...",
    accept_file=True,
    file_type=["pdf", "docx", "txt"],
    disabled=has_pending,
)

if user_message:
    message = (user_message.text or "").strip()
    uploaded_files = user_message.files or []

    # 1. Show what the user sent
    with st.chat_message("user"):
        if message:
            st.markdown(message)
        for uploaded_file in uploaded_files:
            st.markdown(f"📄 **Uploaded:** {uploaded_file.name}")

    # 2. Index every uploaded document
    ingestion_success = True

    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"

        if file_key in st.session_state["indexed_files"]:
            st.info(f"ℹ️ Already indexed: {uploaded_file.name}")
            continue

        try:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                result = index_uploaded_file(uploaded_file)

            st.session_state["document_results"].append({
                "filename": uploaded_file.name,
                "result": result,
            })
            st.session_state["indexed_files"].add(file_key)

            st.success(f"✅ Indexed: {uploaded_file.name}")

        except Exception as e:
            ingestion_success = False
            st.error(f"Failed to index {uploaded_file.name}: {e}")

    # 3. Stream the agent response when the user entered a message
    if message and (not uploaded_files or ingestion_success):
        is_first_message = not current_messages
        stream_ok = False
        interrupted = False

        with st.chat_message("assistant"):
            status_box = st.status("Thinking...", expanded=True)

            try:
                st.write_stream(
                    stream_graph(
                        {"messages": [HumanMessage(content=message)]},
                        thread_config(current_id),
                        status_box,
                    )
                )

                # The stream ends quietly when the graph hits interrupt(),
                # so check the checkpoint for a pending approval.
                new_interrupt = get_pending_interrupt(current_id)

                if new_interrupt is not None:
                    save_pending_interrupt(current_id, new_interrupt)
                    interrupted = True
                    status_box.update(
                        label="⚠️ Waiting for your approval",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label="Response completed",
                        state="complete",
                        expanded=False,
                    )

                stream_ok = True

            except Exception as e:
                status_box.update(label="An error occurred", state="error")
                st.error(f"⚠️ Error: {e}")

        # Rerun so that:
        #  - the Approve / Reject buttons appear when an interrupt happened
        #  - the sidebar shows the new chat title after the first message
        if stream_ok and (interrupted or is_first_message):
            st.rerun()

    elif uploaded_files and ingestion_success:
        st.info(
            "Your document has been indexed. "
            "Ask a question about it in the chat."
        )