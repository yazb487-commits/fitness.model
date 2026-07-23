from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# API Key (⚠️ avoid hardcoding in production)
groq_api = "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"

# Initialize LLM
llm = ChatGroq(
    api_key=groq_api,
    model="openai/gpt-oss-120b",
    temperature=0.1
)

# Prompt Template (English + Hindi output)
prompt = PromptTemplate(
    input_variables=["topics"],
    template="""
You are a helpful assistant.

Provide information about:
{topics}

Format your response as:

## English
<response in English>

## Hindi
<same response translated into Hindi (Devanagari script)>
"""
)

# User Input
topic = input("Enter topics (comma-separated): ")

# Format prompt
format_prompt = prompt.format(topics=topic)

# Get response from LLM
response = llm.invoke(format_prompt)

# Output
print("Generated Response:\n")
print(response.content)