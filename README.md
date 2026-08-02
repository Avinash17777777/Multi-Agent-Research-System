# 🧠 Multi-Agent Research System

A Multi-Agent AI Research Assistant built using **LangChain**, **Groq**, **Tavily Search API**, and **Streamlit**. This application automates the research process by searching the web, reading relevant content, generating a research report, and reviewing the final output.

---

## 🚀 Features

- 🔍 Search the web for reliable information
- 📖 Scrape content from relevant websites
- ✍️ Generate structured research reports
- 🧐 Review and score the generated report
- 🎨 Simple Streamlit user interface
- ⚡ Fast response using Groq LLM

---

## 🛠️ Tech Stack

- Python
- LangChain
- Groq
- Tavily API
- Streamlit
- BeautifulSoup
- Requests

---

## 📂 Project Structure

```
Multi-Agent-Research-System/
│
├── app.py
├── agents.py
├── pipeline.py
├── tools.py
├── requirements.txt
├── README.md
└── .env
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Avinash17777777/Multi-Agent-Research-System.git

cd Multi-Agent-Research-System
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add your API keys.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually **(https://multi-agent-research-system-raat78sluukgs6tt6xe4z3.streamlit.app/)**).

---

## 🔄 How It Works

1. Enter a research topic.
2. The Search Agent finds relevant information.
3. The Reader Agent extracts useful content.
4. The Writer Agent generates a research report.
5. The Critic Agent reviews the report and provides feedback.

---

## 📸 Screenshots

Add screenshots of your application here.

- Home Page
- Generated Report
- Critic Feedback

---

## 🌱 Future Improvements

- Export report as PDF
- Add citation support
- Multi-source summarization
- Memory-enabled agents
- LangGraph integration

---

## 👨‍💻 Author

**Avinash Kumar Singh**

- GitHub: https://github.com/Avinash17777777
- LinkedIn: https://www.linkedin.com/in/avinash-kumar-singh-

---

## ⭐ Support

If you found this project useful, consider giving it a **Star ⭐** on GitHub.
