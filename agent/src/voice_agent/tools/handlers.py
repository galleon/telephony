from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

# Ticket status mock — replace with a real ITSM API call (ServiceNow, Jira, etc.)
_MOCK_TICKETS = {
    "INC001": {"status": "In Progress", "priority": "High", "assigned_to": "Alice (Network Team)", "summary": "VPN connection dropping every 30 minutes"},
    "INC002": {"status": "Resolved", "priority": "Medium", "assigned_to": "Bob (Desktop Support)", "summary": "Outlook not syncing emails"},
    "INC003": {"status": "Open", "priority": "Critical", "assigned_to": "Unassigned", "summary": "Production database unreachable"},
}


async def fetch_ticket_status(params: FunctionCallParams):
    """Look up an IT support ticket by ID."""
    ticket_id = params.arguments.get("ticket_id", "").strip().upper()
    logger.info(f"Looking up ticket: {ticket_id}")

    ticket = _MOCK_TICKETS.get(ticket_id)
    if ticket:
        result = {"ticket_id": ticket_id, **ticket}
    else:
        result = {"ticket_id": ticket_id, "error": "Ticket not found. Please verify the ticket number."}

    await params.result_callback(result)


async def escalate_to_engineer(params: FunctionCallParams):
    """Escalate the call to a human engineer."""
    reason = params.arguments.get("reason", "No reason provided")
    team = params.arguments.get("team", "general")
    logger.warning(f"Escalating call to {team} team. Reason: {reason}")

    # In a real ARI setup, bridge the call to the on-call engineer queue.
    await params.result_callback({"status": "escalating", "queue": f"{team}-oncall"})
