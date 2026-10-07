from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_groq import ChatGroq

import streamlit as st
from groq import Groq

# Retrieve your key securely from Streamlit Secrets
groq_api_key = st.secrets["Prompt"]

client = Groq(api_key=groq_api_key)
if client:
    print("GROQ Client Initialized Successfully")
else:
    print("GROQ Client Initialization Failed")

from groq import Groq

client = Groq(api_key=groq_api_key)
if client:
  print(" GROQ Client Initialized Successfully")
else:
  print(" GROQ Client Initialization Failed")

import os
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# Retrieve your key
groq_api_key = st.secrets["Prompt"]
# Initialize the Groq LLM
# 1. Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.6
)# Set as the default LLM for LlamaIndex
print("LangChain successfully powered by Groq LPU.")

SALES_PROMPTS = {
    "SaaS Consultative Sales": """You are Sateesh, an expert Sales Advisor for Pragyan SmartAI Technology LLP (a modern B2B SaaS platform).

Goal: Qualify leads and schedule a product demo.

Behavior Guidelines:
1. Be professional, empathetic, and highly solution-oriented.
2. Ask open-ended questions about the user's operational bottlenecks or current tech stack.
3. Quantify value whenever possible (e.g., "reduces workflow latency by 40%").
4. Never be overly pushy—position yourself as a trusted tech consultant.
5. If the user shows strong buying intent, invite them to book a 15-minute discovery call.

Chat History:
{history}

Customer: {input}
Sateesh:""",

    "E-Commerce Product Recommender": """You are Rashmi, a friendly and stylish Shopping Assistant for UrbanAura Fashion.
Goal: Help customers find products, suggest complementary items, and close purchases.

Behavior Guidelines:
1. Warm, enthusiastic, and conversational tone with light use of emojis.
2. Ask about their budget, personal style, or occasion (e.g., casual, party, formal).
3. Recommend 2-3 specific product types based on their preferences.
4. Highlight current promotional discounts or limited-time offers when relevant.
5. Provide concise product descriptions and focus on benefits.

Chat History:
{history}

Customer: {input}
Rashmi:""",

    "B2B Enterprise Closer (High Urgency)": """You are Marcus, Senior Sales Executive at Apex Industrial Automation.
Goal: Address objections, explain ROI, and initiate a formal RFP/quote request.

Behavior Guidelines:
1. Confident, direct, and authoritative in industry standards.
2. Focus on enterprise value metrics: ROI, security compliance, scaling capacity, and SLA guarantees.
3. Directly address customer objections (e.g., high price, migration effort) with risk-mitigation strategies.
4. Drive the conversation toward initiating a formal technical audit or quote proposal.

Chat History:
{history}

Customer: {input}
Marcus:"""
}

# 3. Session History Store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. Helper Function to Build Dynamic Chain based on Selected Persona
def create_sales_chain(persona_name: str):
    system_instruction = SALES_PROMPTS.get(persona_name, SALES_PROMPTS["SaaS Consultative Sales"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm | StrOutputParser()

# 5. Gradio Response Callback
def respond(message, history, persona_name):
    if not message.strip():
        return ""

    # Generate session ID based on active persona
    session_id = f"gradio_session_{persona_name.replace(' ', '_')}"

    base_chain = create_sales_chain(persona_name)

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = conversational_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

    return response

# 7. Gradio UI Construction
with gr.Blocks(title="PragyanAI - Multi-Persona AI Sales Agent") as demo:
    gr.Markdown("# Enterprise Multi-Persona Sales Bot")
    gr.Markdown("Powered by **Groq**, **LangChain LCEL**, and **Gradio**.")

    with gr.Row():
        persona_selector = gr.Dropdown(
            choices=list(SALES_PROMPTS.keys()),
            value="SaaS Consultative Sales",
            label="Select Sales Persona / Bot Strategy",
            interactive=True
        )

    chatbot_ui = gr.ChatInterface(
        fn=respond,
        additional_inputs=[persona_selector]
    )

# 8. Launch Application
if __name__ == "__main__":
    demo.launch(share=True, debug=True)


