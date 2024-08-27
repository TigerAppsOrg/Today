from memory import Memory, MessageType, ToolInvocation
from prompts import user_query_with_context, agent_system_prompt
from utils import async_openai_stream

async def generate_response(memory: Memory, tool_use: ToolInvocation):
    query = tool_use["input"]
    context = tool_use["output"]
    response = await async_openai_stream([
        agent_system_prompt(),
        user_query_with_context(context, query)
    ])

    full_response = []
    async for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content != None:
            full_response.append(chunk_content)
            yield chunk_content

    content = ''.join(full_response)
    memory.add_message(MessageType.AI, content, tool_use=tool_use)
