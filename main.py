from datetime import datetime, timedelta
import pandas as pd
from langchain_ollama import OllamaLLM
import feedparser
import urllib, urllib.request
import os
import requests
import json
import time
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from constants import arxiv_cats
from dotenv import load_dotenv

def get_dates():
    last_day = datetime.today() - timedelta(days=3)
    today = datetime.today()
    return last_day.strftime('%Y%m%d0000'), today.strftime('%Y%m%d0000')

def fetch_data():
    max_res = 2000
    start_date, end_date = get_dates()
    url = f'http://export.arxiv.org/api/query?search_query=submittedDate:[{start_date}+TO+{end_date}]&sortBy=submittedDate&sortOrder=ascending&start=0&max_results={max_res}'
    print(url)
    data = urllib.request.urlopen(url)
    xml = data.read().decode('utf-8')
    return xml

def get_prop_array(parsed_xml, prop):
    entries = parsed_xml["entries"]
    prop_array = list(map(lambda e: e[prop], entries))
    return prop_array

def parse_parsed_xml_to_df(parsed_xml):
    ids = get_prop_array(parsed_xml, "id")
    links = get_prop_array(parsed_xml, "link")
    summaries = get_prop_array(parsed_xml, "summary")
    authors = get_prop_array(parsed_xml, "authors")
    authors = list(map(lambda author_array: ",".join([a["name"] for a in author_array]), authors))
    tags = get_prop_array(parsed_xml, "tags")
    tags = list(map(lambda t_arr: ",".join([t["term"] for t in t_arr]), tags))
    primary_cat = get_prop_array(parsed_xml, "arxiv_primary_category")
    primary_cat = list(map(lambda c: c["term"], primary_cat))
    df_data = {
    "ids": ids,
    "links": links,
    "summaries": summaries,
    "authors": authors,
    "tags":tags,
    "primary_cat": primary_cat
    }
    return pd.DataFrame(df_data)
    
def store_xml_as_df():
    retrieved_xml = fetch_data()
    parsed_xml = feedparser.parse(retrieved_xml)
    df = parse_parsed_xml_to_df(parsed_xml)
    df = df.drop_duplicates()
    os.makedirs("papers", exist_ok=True)
    df.to_csv(f"papers/{datetime.today().strftime('%Y%m%d0000')}.csv", index = False)
    return df

def get_prompt():
    prompt_explain_term = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HTML/CSS generating AI assistant. Your sole and primary task is to generate a single, consolidated English summary website fragment in raw, well-formatted HTML and embedded CSS ONLY. This summary will be based on a list of research contributions that will be provided to you. Each contribution includes the author(s), a brief summary of their work, and a direct HTML link for further details.
    
    ABSOLUTELY NO MARKDOWN. Do not use any Markdown syntax like '#', '*', '-', '_', '[]()', or backticks for code blocks. All output must be pure HTML and CSS.
    
    MANDATORY HTML CONVERSION:
    You must actively convert any inclination to use Markdown into pure HTML. For example:
    - Markdown `**text**` or `__text__` MUST become HTML `<strong>text</strong>`.
    - Markdown `*text*` or `_text_` (for italics) MUST be converted to `<em>text</em>` if emphasis is intended and appropriate, though the primary instruction is to use `<strong>` or `<b>` for important aspects.
    - Markdown `[link text](URL)` MUST become HTML `<a href="URL">link text</a>`.
    - Markdown headings like `# Heading 1` or `## Heading 2` MUST become HTML heading tags like `<h1>Heading 1</h1>` or `<h2>Heading 2</h2>` (use appropriate heading levels like `<h3>` as needed, such as in the "Key Concepts Explained" section).
    - Markdown lists like `* Item 1` or `- Item 1` MUST become HTML `<li>Item 1</li>` (and be correctly nested within `<ul>` or `<ol>` tags).
    Your output MUST NOT contain any raw Markdown characters used for styling or structure, such as `*`, `_`, `#` (when used for headings), `[` or `]` (when used for Markdown links), or `(` and `)` (when used for Markdown link URLs). All such constructs must be expressed using their HTML equivalents.
    
    The main goal is to synthesize these individual summaries into a single, cohesive HTML narrative. This narrative should clearly highlight what the authors have accomplished or discovered. The narrative should be structured using standard HTML tags. Paragraphs of text should be enclosed in <p> tags. Authors should be mentioned naturally within the text. Hyperlinks must be HTML <a> tags with an href attribute, embedding links directly related to the discussed contribution. Important aspects should be bolded using <strong> or <b> HTML tags.
    
    The final output must be only the HTML and CSS content itself, as a single string. Do not include any of your own conversational text, introductions, or conclusions before or after the HTML content. Do not output <html>, <head>, or <body> tags. The output should be a fragment ready to be inserted into an existing HTML document.
    
    Input Data Format:
    
    You will receive the data as a single block of text. Within this block, each research entry will be distinctly separated from the next. Each entry will adhere to the following structure:
    
    Authors: [Full Names of Authors]
    Summary: [A concise summary of the work done]
    Link: [A full HTML URL to the source or more information]
    --- (This line with three hyphens acts as a separator between individual entries)
    
    Example of Input Data Block (Illustrative):
    
    Authors: Dr. Ada Lovelace, Charles Babbage
    Summary: They conceptualized the Analytical Engine, a groundbreaking mechanical general-purpose computer, detailing its potential for complex calculations and even for composing music.
    Link: https://example.com/analytical_engine_details
    ---
    Authors: Dr. Grace Hopper
    Summary: A pioneer in computer programming, Dr. Hopper developed the first compiler (A-0 System) and was instrumental in the development of COBOL. She also popularized the term "debugging" after a moth was found in a relay.
    Link: https://example.com/grace_hopper_compiler_work
    ---
    Authors: Dr. Tim Berners-Lee
    Summary: He is credited with inventing the World Wide Web. His work included developing the first web browser, web server, HTTP (Hypertext Transfer Protocol), and HTML (Hypertext Markup Language), fundamentally changing how information is accessed and shared globally.
    Link: https://example.com/world_wide_web_invention
    
    Instructions for Generating the HTML Summary:
    
    1. Parse Input Carefully: Thoroughly read and understand all the provided research entries from {df_string}.
    
    2. Synthesize into an HTML Narrative:
    Do not simply list the summaries. Weave them into a flowing and coherent narrative using HTML <p> tags to structure the text.
    You may use the provided {topic} for general thematic guidance or contextual understanding when crafting the narrative, but it's not mandatory to explicitly include the topic title in the output unless it naturally enhances the summary.
    Focus specifically on the actions, discoveries, inventions, or key findings of the authors. What did they do?
    If there are thematic connections or a progression of research evident across different entries, try to highlight these relationships within the narrative.
    Bold print important aspects or keywords using <strong> or <b> HTML tags. (This means you should generate these HTML tags, not Markdown for bolding).
    
    3. Integrate Links Naturally using HTML <a> Tags:
    For each significant contribution or point you discuss from an entry, you must seamlessly incorporate the corresponding HTML link using an HTML <a> tag with a relevant href attribute.
    The anchor text for the link should be descriptive and fit naturally within the narrative. Make sure to always include the link provided in the input.
    Example: <p>Dr. Ada Lovelace and Charles Babbage conceptualized the <strong>Analytical Engine</strong>, a precursor to modern computing (<a href="https://example.com/analytical_engine_details">explore their concepts</a>).</p> (This example shows correct HTML link and strong tag usage).
    
    4. Concept Explanation Box (Conditional):
    If specific concepts, definitions, or terminology mentioned in the summaries are not common knowledge and warrant explanation for a general audience, create an "Information" section.
    This "Information" section must be a single <div> element, placed after the main narrative summary (i.e., at the end of your HTML output, if included).
    The styling for this div must be defined in a single <style> tag. This <style> tag must be placed at the very beginning of your entire HTML output string. If no explanation box div is generated, then this <style> tag must also NOT be generated.
    Assign a class (e.g., explanation-box) to the div and use this class selector in your CSS.
    Inside the div, you may include a heading (e.g., <h3>Key Concepts Explained</h3>) followed by paragraphs (<p>) or an unordered list (<ul><li>...</li></ul>) defining the terms. Use <strong> tags for the terms being defined.
    
    Example CSS for the <style> tag (if an explanation box is needed, this goes at the start of the entire output):
            <style>
            .explanation-box {{
              background-color: #e6ffe6; /* Light green background */
              border: 1px solid #478f1b; /* Darker green border */
              padding: 15px;
              margin-top: 20px;
              border-radius: 5px; /* Optional: for rounded corners */
            }}
            .explanation-box h3 {{
              color: #2e7d32; /* Dark green for heading */
              margin-top: 0;
            }}
            </style>
    
    Example HTML for the explanation div element (if needed, this goes after the narrative summary):
            <div class="explanation-box">
              <h3>Key Concepts Explained</h3>
              <p><strong>Analytical Engine:</strong> An early mechanical general-purpose computer design.</p>
              <p><strong>Compiler:</strong> A program that translates source code from a high-level programming language to a lower-level language.</p>
            </div>
    
    5. Output Requirements (Strict Adherence Required):
    The entire output must be a single string of raw HTML and embedded CSS.
    NO CONVERSATIONAL TEXT. No "Here is the summary:", "Certainly:", or any other text outside the HTML itself.
    NO MARKDOWN AT ALL. Confirm all output is valid HTML.
    All links from the input must be present and functional as <a> tags in the output.
    If the df_string input is empty or does not contain any valid research entries, output only the following HTML paragraph: <p>No research contributions were provided to summarize.</p>. In this specific case, do not generate any other content, including an explanation box or style tags.
    If an explanation box (div) is not generated because no concepts require it, then the <style> tag for it must also not be generated. Conversely, the <style> tag should only be present if the explanation box div is also present.
    
    Based on the input data that will be provided immediately following this line, generate the raw English HTML summary with embedded CSS (ONLY HTML, NO MARKDOWN). Return only the HTML content string.
    """),
        ("human", "Topic: {topic}\n\nHere is the structured text data (df_string):\n{df_string}")
    ])
    return prompt_explain_term

def send_email_via_api(subject: str, content: str, api_token:str, api_url: str, bypass_proxy: bool = True) -> tuple[int, dict]:
    """
    Sends an email by making a POST request to the specified API endpoint.

    Args:
        subject: The subject of the email.
        content: The HTML or Markdown content of the email.
        api_token: The Bearer token for API authorization.
        api_url: The URL of the API endpoint for sending mail.
                 Defaults to "https://daniel-walke.com/mail/send".
        bypass_proxy: If True, attempts to bypass system-defined proxies for this request.
                      Set to False to use system proxies if configured.

    Returns:
        A tuple containing the HTTP status code and the JSON response from the API.
        If the request fails at the network level or JSON decoding fails,
        it might raise an exception (e.g., requests.exceptions.RequestException, json.JSONDecodeError).
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "subject": subject,
        "content": content,
    }

    proxies_to_use = {
            "http": os.environ.get('PROXY'),
            "https": os.environ.get('PROXY'),
        }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), proxies=proxies_to_use)

        try:
            response_json = response.json()
        except json.JSONDecodeError:
            response_json = {"error": "Failed to decode JSON", "text_response": response.text}

        return response.status_code, response_json

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return 0, {"error": str(e)}

load_dotenv()
data_df = store_xml_as_df()
max_len = len(data_df.groupby("primary_cat"))
md_responses = []
i = 0
overall_llm_time = time.time()
for group, group_data in data_df.groupby("primary_cat"):
    print(f"{i} of {max_len}")
    i+=1
    if group not in ["cs.AI", "cs.LG", "stat.ML"]: continue
    group_df_string = ""
    for index, row in group_data.iterrows():
        group_df_string += f"""
        Authors: {row["authors"]}
        Summary: {row["summaries"]}
        Link: {row["links"]}
        ---
        """
    category = group
    if category in arxiv_cats:
        category = arxiv_cats[category]
    prompt = get_prompt()
    llm_model = OllamaLLM(model="gemma3:27b")
    chain = prompt | llm_model | StrOutputParser()
    llm_response = chain.invoke({"topic": category, "df_string": group_df_string})
    if llm_response is None: continue
    if "</think>" in llm_response:
        llm_response = llm_response.split("</think>")[-1]
    print(llm_response)
    md_response = {
        "cat": group,
        "resp": llm_response
    }
    md_responses.append(md_response)
print(f"{str(time.time() - overall_llm_time)} s needed overall for the LLM")
start_date, end_date = get_dates()
final_html = f"<h1>Summaries between {start_date} and {end_date}</h1>"
for md_resp in md_responses:
    category = md_resp["cat"]
    if category in arxiv_cats:
        category = arxiv_cats[category]
    final_html += md_resp["resp"]

os.makedirs("html", exist_ok = True)
with open(f"html/{start_date}.html", "w", encoding="utf-8") as f:
    f.write(final_html)

send_email_via_api(f"Summary {start_date}", final_html, os.environ.get('API_TOKEN'), os.environ.get('API_URL'))
