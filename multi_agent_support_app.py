import streamlit as st
import pandas as pd
import random
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup GROQ client
client = OpenAI(api_key=st.secrets["grok_api_key"], base_url="https://api.groq.com/openai/v1")
GROQ_MODEL = "llama3-8b-8192"

# GROQ-powered summary (simplified)
def summarize_conversation(text):
    try:
        prompt = f"""
        Summarize the customer's issue in 1-2 sentences:

        {text}
        """
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary Error: {str(e)}"

# Simplified action extraction
def extract_actions(text):
    try:
        prompt = f"""
        List 2-3 short action steps to resolve this issue:

        {text}
        """
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        return [f"Action Error: {str(e)}"]

# Load historical data
def load_historical_data():
    try:
        df = pd.read_csv("C:\\Users\\pooji\\OneDrive\\Desktop\\HACKathon\\Dataset\\[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents\\Conversation\\Historical_ticket_data.csv")
        df.columns = [col.strip() for col in df.columns]
        if 'Issue Category' not in df.columns or 'Sentiment' not in df.columns or 'Solution' not in df.columns:
            st.error("CSV must contain 'Issue Category', 'Sentiment', and 'Solution'.")
            return []
        df['query'] = df['Issue Category'].str.strip() + " - " + df['Sentiment'].str.strip()
        df['solution'] = df['Solution'].str.strip()
        return df[['query', 'solution']].to_dict(orient='records')
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return []

# LLM fallback solution
def generate_llm_solution(query):
    try:
        prompt = f"""
        Provide a brief, actionable solution for this support query:

        {query}
        """
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"

# Resolution recommendation
def recommend_resolution(customer_query, historical_data):
    if not historical_data:
        return generate_llm_solution(customer_query)
    try:
        vectorizer = TfidfVectorizer()
        queries = [data["query"] for data in historical_data]
        vectorizer.fit(queries + [customer_query])
        query_vector = vectorizer.transform([customer_query])
        similarity_scores = cosine_similarity(query_vector, vectorizer.transform(queries))
        best_idx = similarity_scores.argmax()
        best_score = similarity_scores[0][best_idx]

        if best_score > 0.3:
            return historical_data[best_idx]["solution"]
        else:
            return generate_llm_solution(customer_query)
    except Exception as e:
        return f"Recommendation error: {e}"

# Routing logic
def route_task(conversation_text, category, priority):
    if "payment" in conversation_text.lower() or category == "Billing Support":
        return "Payment Support Team"
    elif priority == "Critical":
        return "Urgent Support Queue"
    elif "API" in conversation_text:
        return "API Support Team"
    elif category == "Technical Support":
        return "Technical Support Team"
    else:
        return "General Support Team"

# Resolution time
def estimate_resolution_time(category):
    avg_times = {
        "Technical Support": [10, 20, 30, 15],
        "Billing Support": [5, 10, 15],
        "General Inquiry": [2, 5, 8]
    }
    if category in avg_times:
        return round(sum(avg_times[category]) / len(avg_times[category]), 2)
    return random.randint(10, 60)

# Main agent call
def orchestrate_workflow(conversation_text, category, priority, historical_data):
    return {
        "summary": summarize_conversation(conversation_text),
        "actions": extract_actions(conversation_text),
        "route": route_task(conversation_text, category, priority),
        "recommendation": recommend_resolution(conversation_text, historical_data),
        "estimated_time": estimate_resolution_time(category)
    }

# UI
def main():
    st.set_page_config(page_title="IntelliSupport", layout="centered")
    st.title("🤖 IntelliSupport: Smart Agent System")
    st.markdown("""
    <style>
    .stTextArea textarea {
        background-color: #f9f9f9;
        font-size: 16px;
        color: black;
    }
    .stButton button {
        background-color: #0366d6;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    customer_query = st.text_area("Enter Customer Query:")
    category = st.selectbox("Category:", ["Technical Support", "Billing Support", "General Inquiry"])
    priority = st.selectbox("Priority:", ["Low", "Medium", "High", "Critical"])

    historical_data = load_historical_data()

    if st.button("Run Agents"):
        if not customer_query:
            st.warning("Please input a customer query.")
            return
        with st.spinner("Running agents..."):
            result = orchestrate_workflow(customer_query, category, priority, historical_data)

        st.subheader("🔍 Results")
        st.markdown(f"**Summary:** {result['summary']}")
        st.markdown("**Actions:**")
        for action in result["actions"]:
            st.markdown(f"- {action}")
        st.markdown(f"**Task Route:** `{result['route']}`")
        st.markdown(f"**Recommended Resolution:** {result['recommendation']}")
        st.markdown(f"**Estimated Resolution Time:** ⏱️ {result['estimated_time']} minutes")

if __name__ == "__main__":
    main()
