import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
import pandas as pd
import io
import plotly.express as px

st.set_page_config(page_title="My Personal Finance Agent", page_icon="💰")
st.title("💰 My Personal Finance Agent")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your expenses CSV (columns: date, amount, description, category)",
    type="csv"
)

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
    st.dataframe(st.session_state.df)

# Tools for the agent
@tool
def analyze_expenses() -> str:
    """Analyze the current uploaded expenses data."""
    if st.session_state.df is None:
        return "No expenses data uploaded yet."
    df = st.session_state.df
    total = df['amount'].sum()
    category_totals = df.groupby('category')['amount'].sum().round(2)
    top_expense = df.loc[df['amount'].idxmax()]
    
    fig = px.pie(values=category_totals.values, names=category_totals.index, title="Spending by Category")
    st.plotly_chart(fig)
    
    return f"""
Total spending: ${total:.2f}
Top category: {category_totals.idxmax()} (${category_totals.max():.2f})
Most expensive item: {top_expense['description']} (${top_expense['amount']:.2f})

Category breakdown:
{category_totals.to_string()}
"""

@tool
def suggest_budget(monthly_income: float) -> str:
    """Suggest a budget based on current expenses and monthly income."""
    if st.session_state.df is None:
        return "Upload expenses first."
    total_spent = st.session_state.df['amount'].sum()
    savings = monthly_income - total_spent
    return f"""
With ${monthly_income} monthly income and ${total_spent:.2f} in expenses:
• Suggested savings: ${max(savings, 0):.2f} ({'Good job!' if savings > 0 else 'Consider reducing spending'})
• Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings
"""

tools = [analyze_expenses, suggest_budget]

# LLM setup
llm = ChatGroq(
    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=st.secrets["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"],
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and expert personal finance assistant. Use tools when needed to analyze data and give practical advice."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything about your finances... (e.g., 'Analyze my spending' or 'Suggest budget for $5000 income')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Convert chat history to LangChain messages
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            answer = response["output"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
