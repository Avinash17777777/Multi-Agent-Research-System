import re
import time
import groq
from agents import (
    build_reader_agent, build_search_agent,
    writer_chain, writer_chain_fallback,
    critic_chain, critic_chain_fallback,
)
from tools import web_search , scrape_url
from langchain_core.messages import ToolMessage


def extract_tool_output(agent_result: dict) -> str:
    """
    Pull the raw content returned by any tools the agent called during its run.
    This is more reliable than the agent's final, paraphrased answer, which can
    drop or reword details (like URLs) that downstream steps depend on.
    Falls back to the final message if the agent never actually called a tool.
    """
    tool_texts = [
        m.content for m in agent_result["messages"]
        if isinstance(m, ToolMessage) and m.content
    ]
    if tool_texts:
        return "\n\n".join(tool_texts)
    return agent_result["messages"][-1].content


def extract_urls(text: str, limit: int = 5) -> list:
    """Pull URLs out of raw text, in order, without duplicates."""
    found = re.findall(r'https?://[^\s\)\]"\'>]+', text)
    seen = []
    for url in found:
        url = url.rstrip('.,;:')  # drop trailing punctuation the regex may have grabbed
        if url not in seen:
            seen.append(url)
    return seen[:limit]


def _retry_after_seconds(error, default=None):
    """
    Prefer Groq's `retry-after` response header (authoritative, in seconds).
    Falls back to parsing "try again in Xm Ys" out of the error message if the
    header isn't present for some reason.
    """
    try:
        header_val = error.response.headers.get("retry-after")
        if header_val is not None:
            return float(header_val)
    except Exception:
        pass

    match = re.search(r'try again in (?:(\d+)m)?([\d.]+)s', str(error))
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    return default


def call_with_rate_limit_retry(fn, *args, max_wait_s=120, max_retries=3, fallback_fn=None, **kwargs):
    """
    Call fn(*args, **kwargs).
    - On a Groq 429, wait exactly as long as Groq says to (typically a few
      seconds for a per-minute burst) and retry automatically.
    - If the wait Groq wants is longer than max_wait_s (the model's *daily*
      quota is actually exhausted, not just a burst), don't block silently:
      use fallback_fn if one was given (e.g. a backup model with its own
      separate quota), or raise a clear error explaining how long to wait.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except groq.RateLimitError as e:
            last_error = e
            wait_s = _retry_after_seconds(e, default=max_wait_s + 1)
            if wait_s <= max_wait_s:
                print(f"\n rate limited by Groq; waiting {wait_s:.0f}s before retry {attempt}/{max_retries} ...")
                time.sleep(wait_s + 1)
                continue
            if fallback_fn is not None:
                print(f"\n Groq's daily quota looks exhausted (~{wait_s/60:.1f} min wait needed); using backup model instead")
                return fallback_fn(*args, **kwargs)
            raise RuntimeError(
                f"Groq rate/quota limit hit and no backup model was configured for this step. "
                f"Groq says to wait ~{wait_s/60:.1f} more minutes. Original error: {e}"
            ) from e
    raise last_error


def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    try:
        search_result = call_with_rate_limit_retry(
            search_agent.invoke,
            {"messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        state["search_results"] = extract_tool_output(search_result)
    except Exception as e:
        print(f"\n search agent failed ({e}); falling back to a direct web_search call")
        state["search_results"] = web_search.invoke({"query": topic})

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    urls = extract_urls(state["search_results"])
    reader_agent = build_reader_agent()

    if not urls:
        print("\n no URLs found in search results; skipping scrape step")
        state['scraped_content'] = "No URLs were found in the search results to scrape."
    else:
        url_list_text = "\n".join(f"- {u}" for u in urls)
        reader_prompt = (
            f"Here are URLs found while researching '{topic}':\n{url_list_text}\n\n"
            f"Call the scrape_url tool on the single URL most likely to have deep, "
            f"reliable coverage of the topic."
        )
        try:
            reader_result = call_with_rate_limit_retry(
                reader_agent.invoke,
                {"messages": [("user", reader_prompt)]}
            )
            state['scraped_content'] = extract_tool_output(reader_result)
        except Exception as e:
            print(f"\n reader agent failed ({e}); falling back to a direct scrape of the top URL")
            state['scraped_content'] = scrape_url.invoke({"url": urls[0]})

    print("\nscraped content: \n", state['scraped_content'])

    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = call_with_rate_limit_retry(
        writer_chain.invoke,
        {
            "topic" : topic,
            "research" : research_combined
        },
        fallback_fn=writer_chain_fallback.invoke,
    )

    print("\n Final Report\n",state['report'])

    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = call_with_rate_limit_retry(
        critic_chain.invoke,
        {"report":state['report']},
        fallback_fn=critic_chain_fallback.invoke,
    )

    print("\n critic report \n", state['feedback'])

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)