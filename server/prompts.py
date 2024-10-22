from utils import system_prompt, user_prompt, time_to_date_string
from memory import Memory
from models import Tools

@user_prompt
def user_query(text: str):
    return text

@user_prompt
def user_query_with_context(context: str, query: str):
    return f"""The user will now supply you with a query.\n\nAnswer the user's question using the following 
    documents as context. You need to be as accurate as possible, so if you don't know some details do not
    guess at them. Instead tell them that you don't have that information. ***IMPORTANT: Don't be lazy. Give
    full detailed answers to the user when all the details are available.*** Here are the documents:\n\n{context}
    \n\nUser query: {query}"""

@system_prompt
def get_courses_search_query():
    return """You will receive a piece of text related to looking up a college course at Princeton, and you need to 
    shorten it down into a course code or a couple concise keywords. Here are a couple examples:
    
    Example 1
    Input: "class about artificial intelligence and education for undergraduates"
    Output: "AI and education"

    Example 2
    Input: "what semester is baby wants candy offered in"
    Output: "Baby wants candy"

    Example 3
    Input: "should I take COS217"
    Output: "COS 217"

    Example 4
    Input: "is cos597h hard?"
    Output: "COS 597H"

    Example 5
    Input: "cool music course"
    Output: "Music"

    Example 6
    Input: "computer science classes"
    Output: "COS"

    ***IMPORTANT: Never include extra descriptive words like "classes" or "undergraduate."
    Also don't add words such as "difficulty". You must keep it very simple otherwise the 
    search will not work. Basically only the course code and keywords. Also if the query
    is general for a department, such as computer science classes or music classes, use
    the department code such as COS or MUS.***

    Return a JSON where your output is a string under the key "search_query". For example,
    {
        "search_query": "AI and education"
    }
    """

@system_prompt
def agent_system_prompt():
    return f"""Your name is Tay, and you are an AI assistant geared toward Princeton students. 
    You have access to the latest Princeton listserv emails, including:

    - WHITMANWIRE
    - WestWire
    - allug
    - BSE-soph
    - freefood
    - transit-alert
    - MatheyMail
    - public-lectures
    - CampusRecInfoList
    - pace-center

    This allows you to provide real-time information about campus events, clubs, job opportunities, deadlines, and other activities.

    ...

    The current date is {time_to_date_string()}.
    ...
    """

@system_prompt
def tool_and_rewrite(tools: Tools, memory: Memory):
    messages = memory.get_messages()
    conversation_context = "\n\n".join(messages)

    tool_names = ", ".join([tool['name'].value for tool in tools])
    tool_descriptions = [f"Tool Name: {tool['name'].value}\nTool Description:{tool['description']}" for tool in tools]
    tool_context = "\n\n".join(tool_descriptions)
    
    prompt = f"""Here is the ongoing conversation between you and the user:
    {conversation_context}

    Here are the tools available to you to answer the user query:
    {tool_context}

    You need to return a JSON with the following two keys:

    "tool": TOOL_NAME where TOOL_NAME is one of {tool_names} or null. Return null if using a tool is
    unnecessary, such as for questions like "who are you?" which are not related to any Princeton context.
    IMPORTANT: If the query is specifically relevant to Princeton, whether its asking about a professor or 
    a club, do not return null! You should definitely use one of the tools in this case.

    "query_rewrite": CONTEXTUALIZED_QUERY where CONTEXTUALIZED_QUERY is a rewritten version of the user query,
    which is imbued with context from the ongoing conversation. An example is if the user asks "Tell me more
    about him" and the ongoing conversation is about Professor Arvind Narayanan, CONTEXTUALIZED_QUERY should be
    something like "Tell me more about Professor Arvind Narayanan." The contextualized rewrite of the query
    should include whatever information you think is necessary to make it an effective, standalone query. Note
    that for questions like "who are you?" there will be information already supplied in your system prompt,
    so you don't have to rewrite the query. IMPORTANT: you are already located in the context of Princeton
    University, so you don't have contextualize it with phrases like "at Princeton University." Furthermore,
    this tool is primarily geared for undergraduates, so for any queries about things like classes or academics,
    include "for undergraduates" in the query rewrite unless explicitly asked for graduate work.

    ***VERY IMPORTANT: Never guess at any unknown acronyms that are supplied and rewrite them. Keep the acronyms as they
    are, especially any potentially relating to student groups or academic departments. You may only expand the 
    most obvious ones such as "AI" = artificial intelligence.***
    """

    return prompt
