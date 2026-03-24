from loguru import logger
from pipecat.services.llm_service import FunctionCallParams


# 1. Define the actual function logic
async def fetch_order_status(params: FunctionCallParams):
    """Handler for getting order status from a DB"""
    order_id = params.arguments.get("order_id")
    logger.info(f"Checking status for order: {order_id}")

    # Mock database lookup
    result = {"order_id": order_id, "status": "Shipped", "delivery_date": "2026-03-25"}

    # Send the result back to the LLM so it can answer the user
    await params.result_callback(result)


async def transfer_to_human(params: FunctionCallParams):
    """Handler to signal the system to transfer the call"""
    reason = params.arguments.get("reason", "No reason provided")
    logger.warning(f"Transferring call. Reason: {reason}")

    # In a real ARI setup, you'd send a 'sip_call_transfer' frame here
    await params.result_callback({"status": "transferring", "agent_group": "support"})
