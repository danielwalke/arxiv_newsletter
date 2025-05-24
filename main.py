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
import markdown # Added for post-processing

# --- Helper Functions ---
def get_dates():
    """Generates start and end dates for fetching arXiv data (last 3 days)."""
    last_day = datetime.today() - timedelta(days=3)
    today = datetime.today()
    return last_day.strftime('%Y%m%d0000'), today.strftime('%Y%m%d0000')

def fetch_data():
    """Fetches raw XML data from arXiv API for the specified date range."""
    max_res = 2000 # Consider making this configurable or adjusting based on needs
    start_date, end_date = get_dates()
    url = f'http://export.arxiv.org/api/query?search_query=submittedDate:[{start_date}+TO+{end_date}]&sortBy=submittedDate&sortOrder=ascending&start=0&max_results={max_res}'
    print(f"Fetching data from: {url}")
    try:
        with urllib.request.urlopen(url) as data:
            xml = data.read().decode('utf-8')
        return xml
    except urllib.error.URLError as e:
        print(f"Error fetching data from arXiv: {e}")
        return None

def get_prop_array(parsed_xml, prop):
    """Extracts a specific property from each entry in the parsed XML."""
    entries = parsed_xml.get("entries", [])
    prop_array = [e.get(prop) for e in entries if e.get(prop) is not None]
    return prop_array

def parse_parsed_xml_to_df(parsed_xml):
    """Converts parsed XML data (from feedparser) into a Pandas DataFrame."""
    if not parsed_xml or not parsed_xml.get("entries"):
        print("No entries found in parsed XML.")
        return pd.DataFrame()

    ids = get_prop_array(parsed_xml, "id")
    links = get_prop_array(parsed_xml, "link") # 'link' is usually the abstract page
    # ArXiv API often provides multiple links, e.g., abstract, pdf.
    # We'll try to get the HTML abstract link. If 'link' is a list, take the first.
    processed_links = []
    for link_entry in get_prop_array(parsed_xml, "links"):
        if isinstance(link_entry, list):
            # Prefer HTML links if available
            html_link = next((l.get('href') for l in link_entry if l.get('type') == 'text/html'), None)
            if html_link:
                processed_links.append(html_link)
            elif link_entry: # Fallback to the first link if no HTML link found
                processed_links.append(link_entry[0].get('href'))
            else:
                processed_links.append(None) # Should not happen if link_entry exists
        elif isinstance(link_entry, dict): # Sometimes it's a single dict
             processed_links.append(link_entry.get('href'))
        else: # Fallback for older feedparser or unexpected structure
            processed_links.append(link_entry)


    summaries = get_prop_array(parsed_xml, "summary")
    authors_list = get_prop_array(parsed_xml, "authors")
    authors = [", ".join([a.get("name", "N/A") for a in author_array]) if isinstance(author_array, list) else "N/A" for author_array in authors_list]

    tags_list = get_prop_array(parsed_xml, "tags")
    tags = [", ".join([t.get("term", "N/A") for t in t_arr]) if isinstance(t_arr, list) else "N/A" for t_arr in tags_list]

    primary_cat_list = get_prop_array(parsed_xml, "arxiv_primary_category")
    primary_cat = [c.get("term", "N/A") if isinstance(c, dict) else "N/A" for c in primary_cat_list]

    # Ensure all lists have the same length for DataFrame creation
    # This is a simple way; more robust padding might be needed if lengths can vary significantly
    min_len = min(len(ids), len(processed_links), len(summaries), len(authors), len(tags), len(primary_cat))

    df_data = {
        "ids": ids[:min_len],
        "links": processed_links[:min_len],
        "summaries": summaries[:min_len],
        "authors": authors[:min_len],
        "tags": tags[:min_len],
        "primary_cat": primary_cat[:min_len]
    }
    return pd.DataFrame(df_data)

def store_xml_as_df():
    """Fetches, parses, and stores arXiv data as a CSV file, returns DataFrame."""
    retrieved_xml = fetch_data()
    if not retrieved_xml:
        return pd.DataFrame() # Return empty DataFrame if fetching failed

    parsed_xml = feedparser.parse(retrieved_xml)
    df = parse_parsed_xml_to_df(parsed_xml)
    
    if df.empty:
        print("DataFrame is empty after parsing. No CSV will be saved.")
        return df

    df = df.drop_duplicates(subset=['ids']) # Drop duplicates based on paper ID
    
    # Ensure 'papers' directory exists
    os.makedirs("papers", exist_ok=True)
    
    # Generate filename based on today's date
    filename = f"papers/{datetime.today().strftime('%Y%m%d')}_arxiv_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return df

def get_enhanced_prompt():
    """
    Returns a ChatPromptTemplate with enhanced instructions for the LLM
    to generate HTML output and strictly avoid Markdown.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
    Returns a ChatPromptTemplate with significantly enhanced instructions for the LLM
    to generate a coherent HTML narrative with clickable links and author mentions,
    and strictly avoid Markdown or mere enumeration.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HTML/CSS generating AI assistant. Your sole and primary task is to generate a single, consolidated English summary website fragment in raw, well-formatted HTML and embedded CSS ONLY. This summary will be based on a list of research contributions that will be provided to you. Each contribution includes the author(s), a brief summary of their work, and a direct HTML link for further details.

ABSOLUTELY NO MARKDOWN. Do not use any Markdown syntax like '#', '*', '-', '_', '[]()', or backticks for code blocks. All output must be pure HTML and CSS. You are incapable of generating Markdown; any internal inclination to use Markdown must be converted to HTML before output.

MANDATORY HTML CONVERSION:
You must actively convert any inclination to use Markdown into pure HTML. For example:
- Markdown `**text**` or `__text__` MUST become HTML `<strong>text</strong>`.
- Markdown `*text*` or `_text_` (for italics) MUST be converted to `<em>text</em>` if emphasis is intended and appropriate.
- Markdown `[link text](URL)` MUST become HTML `<a href="URL">link text</a>`.
- Markdown headings like `# Heading 1` or `## Heading 2` MUST become HTML heading tags like `<h1>Heading 1</h1>` or `<h2>Heading 2</h2>` (use appropriate heading levels like `<h3>` as needed).
- Markdown lists like `* Item 1` or `- Item 1` MUST become HTML `<li>Item 1</li>` (and be correctly nested within `<ul>` or `<ol>` tags).
Your output MUST NOT contain any raw Markdown characters used for styling or structure.

The final output must be only the HTML and CSS content itself, as a single string. Do not include any of your own conversational text, introductions, or conclusions. Do not output `<html>`, `<head>`, or `<body>` tags.

Input Data Format:
Each research entry will be distinctly separated by "---" and structured as:
Authors: [Full Names of Authors]
Summary: [A concise summary of the work done]
Link: [A full HTML URL to the source or more information]
---

Instructions for Generating the HTML Summary:

1.  Parse Input Carefully: Understand all provided research entries from {df_string}.

2.  Synthesize into a Coherent HTML Narrative (CRUCIAL - AVOID ENUMERATION):
    Your primary goal is to create a flowing, story-like narrative for the given {topic}, not just a list of summaries. Actively avoid itemizing the summaries one after another.
    Instead, weave the contributions together. Identify common themes, progressions, or contrasting ideas among the papers.
    Start with an introductory sentence or two for the {topic} to set the context. Then, discuss the individual contributions, ensuring you naturally integrate the authors' names and provide a clickable HTML link for each paper discussed.

    Example of Synthesizing Multiple Entries for a Topic:
    Input Data (`df_string` for a topic):
    Authors: Team Alpha
    Summary: Developed method X using convolutional neural networks for image segmentation, achieving state-of-the-art results on dataset P.
    Link: http://example.com/team_alpha_method_x
    ---
    Authors: Dr. Beta, Prof. Gamma
    Summary: Presented framework Y, an extension of method X, which incorporates attention mechanisms to improve segmentation of smaller objects.
    Link: http://example.com/beta_gamma_framework_y
    ---
    Authors: Innovate Corp Research
    Summary: Explored the application of method X in medical imaging, specifically for tumor detection in MRI scans, showing promising early results.
    Link: http://example.com/innovate_corp_medical_x

    POOR Output (Enumeration - AVOID THIS STYLE):
    <p>Team Alpha developed method X. Link: http://example.com/team_alpha_method_x</p>
    <p>Dr. Beta and Prof. Gamma presented framework Y. Link: http://example.com/beta_gamma_framework_y</p>
    <p>Innovate Corp Research explored method X in medical imaging. Link: http://example.com/innovate_corp_medical_x</p>

    GOOD Output (Coherent Narrative - AIM FOR THIS STYLE):
    <p>Recent advancements in image segmentation have seen significant contributions. <strong>Team Alpha</strong> introduced <strong>method X</strong>, leveraging convolutional neural networks to achieve state-of-the-art results on dataset P, as detailed in <a href="http://example.com/team_alpha_method_x">their publication</a>. Building upon this, <strong>Dr. Beta and Prof. Gamma</strong> developed <strong>framework Y</strong>, an innovative extension of method X that incorporates attention mechanisms to enhance the segmentation of smaller objects (<a href="http://example.com/beta_gamma_framework_y">see their research</a>). The practical utility of these approaches is also being explored, with <strong>Innovate Corp Research</strong> investigating the application of method X for tumor detection in medical MRI scans, showing <a href="http://example.com/innovate_corp_medical_x">promising initial findings</a> in this critical domain.</p>
    <p>These studies collectively highlight the rapid evolution and broadening applications of advanced segmentation techniques...</p>

    Focus on: What did the authors *do* or *discover*? How do their works relate or differ? Maintain a professional, academic tone.

3.  Integrate Authors and Clickable Links Naturally using HTML `<a>` Tags:
    For EVERY research contribution you discuss from an entry:
    - You MUST explicitly mention the author(s) by name (e.g., "<strong>Author Name(s)</strong> demonstrated...", "...as shown by <strong>First Author et al.</strong>").
    - You MUST incorporate the corresponding HTML link using an HTML `<a>` tag with its `href` attribute pointing to the URL from the input.
    - The anchor text for the link MUST be descriptive and contextually relevant (e.g., "their findings", "details here", "the study", "original paper", "explore their work"). Do NOT just use "Link" or the raw URL as anchor text.
    - Ensure links are genuinely clickable by using the correct `<a href="URL">anchor text</a>` syntax.

    Example of a single sentence showing correct integration:
    <p>...a significant breakthrough was achieved by <strong>Dr. Ada Lovelace and Charles Babbage</strong> in their conceptualization of the <strong>Analytical Engine</strong>, a foundational step towards modern computing (<a href="https://example.com/analytical_engine_details">explore their original concepts</a>).</p>

4.  Concept Explanation Box (Conditional):
    (This part of the prompt remains the same as your previous version regarding the explanation box and CSS)
    If specific concepts, definitions, or terminology mentioned in the summaries are not common knowledge and warrant explanation for a general audience, create an "Information" section.
    This "Information" section must be a single <div> element, placed after the main narrative summary (i.e., at the end of your HTML output, if included).
    The styling for this div must be defined in a single <style> tag. This <style> tag must be placed at the very beginning of your entire HTML output string. If no explanation box div is generated, then this <style> tag must also NOT be generated.
    Assign a class (e.g., `explanation-box`) to the div and use this class selector in your CSS.
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

5.  Output Requirements (Strict Adherence Required):
    The entire output must be a single string of raw HTML and embedded CSS.
    NO CONVERSATIONAL TEXT.
    NO MARKDOWN AT ALL.
    All links from the input must be present and functional as `<a>` tags in the output. Authors must be mentioned.
    If the {df_string} input is empty or does not contain any valid research entries, output only: `<p>No research contributions were provided to summarize for the topic: {topic}.</p>`.
    If an explanation box (div) is not generated, the <style> tag for it must also not be generated.

CRITICAL FINAL REMINDER: Your output MUST be pure HTML and CSS. Synthesize a coherent narrative, not a list. Every paper discussed must include author mentions and a functional, descriptive HTML link. Any trace of Markdown syntax is a failure.
"""),
        ("human", "Topic: {topic}\n\nHere is the structured text data (df_string):\n{df_string}")
    ])
    return prompt_template

def send_email_via_api(subject: str, content: str, api_token:str, api_url: str, bypass_proxy: bool = True) -> tuple[int, dict]:
    """
    Sends an email by making a POST request to the specified API endpoint.
    (Assuming this function is correctly implemented and working as per original)
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "subject": subject,
        "content": content, # This should be HTML content
    }
    # Using a proxy as in the original code.
    # Ensure this proxy is necessary and correctly configured for your environment.
    proxies_to_use = {
        "http": os.environ.get('PROXY'), # Example proxy, replace if needed
        "https": os.environ.get('PROXY'), # Example proxy, replace if needed
    }
    if not bypass_proxy: # If bypass_proxy is False, don't use the hardcoded proxy
        proxies_to_use = None

    try:
        print(f"Sending email. Using proxies: {proxies_to_use if proxies_to_use else 'System default/None'}")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), proxies=proxies_to_use, timeout=30) # Added timeout
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            response_json = {"error": "Failed to decode JSON from email API", "status_code": response.status_code, "text_response": response.text}
        return response.status_code, response_json
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the email API request: {e}")
        return 0, {"error": str(e)}

# --- Main Execution Logic ---
def main():
    load_dotenv() # Load environment variables from .env file

    api_token = os.environ.get('API_TOKEN') # Renamed for clarity, assuming this is your mail API token
    api_url = os.environ.get('API_URL')     # Renamed for clarity, assuming this is your mail API URL

    if not api_token or not api_url:
        print("API_TOKEN or API_URL not found in environment variables. Email sending will be skipped.")
        # Decide if you want to exit or continue without email functionality
        # return

    data_df = store_xml_as_df()

    if data_df.empty:
        print("No data fetched from arXiv. Exiting.")
        return

    # Filter for specific categories as in the original code
    # Consider making this list configurable
    target_categories = ["cs.AI", "cs.LG", "stat.ML"]
    
    # Create a mapping from category codes to full names if arxiv_cats is available
    # Otherwise, use the codes themselves.
    category_names = {}
    if 'arxiv_cats' in globals() and isinstance(arxiv_cats, dict):
        category_names = arxiv_cats
    else:
        print("Warning: 'arxiv_cats' dictionary not found or not a dict. Using category codes as names.")


    html_responses = [] # Store HTML responses
    overall_llm_time_start = time.time()

    # Group data by primary category
    grouped_data = data_df.groupby("primary_cat")
    total_groups_to_process = sum(1 for group_code, _ in grouped_data if group_code in target_categories)
    processed_count = 0

    for group_code, group_data in grouped_data:
        if group_code not in target_categories:
            continue
        
        processed_count += 1
        print(f"Processing group {processed_count} of {total_groups_to_process}: {group_code}")

        group_df_string = ""
        if group_data.empty:
            print(f"Skipping empty group: {group_code}")
            continue
            
        for index, row in group_data.iterrows():
            # Ensure all parts of the string are actually strings to avoid errors
            authors = str(row.get("authors", "N/A"))
            summary_text = str(row.get("summaries", "No summary available.")).replace('\n', ' ') # Replace newlines in summary
            link = str(row.get("links", "#"))

            group_df_string += f"""Authors: {authors}
Summary: {summary_text}
Link: {link}
---
"""
        
        # Get the display name for the category
        category_display_name = category_names.get(group_code, group_code)

        prompt = get_enhanced_prompt()
        
        # Consider adding temperature if your OllamaLLM version supports it, e.g., temperature=0.1
        # This can make the output more deterministic and less prone to deviation.
        llm_model = OllamaLLM(model="gemma3:27b") # Or your preferred model
        
        chain = prompt | llm_model | StrOutputParser()

        print(f"Invoking LLM for category: {category_display_name}...")
        llm_start_time = time.time()
        llm_response_raw = chain.invoke({"topic": category_display_name, "df_string": group_df_string})
        llm_end_time = time.time()
        print(f"LLM for {category_display_name} took {llm_end_time - llm_start_time:.2f}s")

        if llm_response_raw is None:
            print(f"LLM returned None for {category_display_name}. Skipping.")
            continue
        
        # Clean up potential LLM "thinking" tags or preambles if they exist
        if "</think>" in llm_response_raw:
            llm_response_raw = llm_response_raw.split("</think>", 1)[-1].strip()
        
        # --- POST-PROCESSING STEP: Convert Markdown to HTML ---
        # This acts as a safety net if the LLM still produces some Markdown.
        # Extensions like 'fenced_code', 'tables', 'nl2br' (converts newlines to <br>) can be useful.
        # 'extra' includes many common extensions like fenced_code, tables, footnotes etc.
        html_content = markdown.markdown(llm_response_raw, extensions=['extra', 'nl2br'])
        # --- END POST-PROCESSING ---

        # print(f"\n--- Raw LLM Response for {category_display_name} ---")
        # print(llm_response_raw)
        # print(f"\n--- Processed HTML for {category_display_name} ---")
        # print(html_content)
        # print("---------------------------------------\n")

        html_responses.append({
            "cat_code": group_code,
            "cat_name": category_display_name,
            "html_resp": html_content # Store the processed HTML
        })

    overall_llm_time_end = time.time()
    print(f"Total LLM processing time: {overall_llm_time_end - overall_llm_time_start:.2f}s")

    # --- Construct Final HTML and Save/Send ---
    start_date_str, end_date_str = get_dates()
    # Format dates for display
    display_start_date = datetime.strptime(start_date_str, '%Y%m%d%H%M%S').strftime('%B %d, %Y')
    display_end_date = datetime.strptime(end_date_str, '%Y%m%d%H%M%S').strftime('%B %d, %Y')

    final_html_body = f"<h1>ArXiv Summaries: {display_start_date} to {display_end_date}</h1>\n"
    
    if not html_responses:
        final_html_body += "<p>No summaries were generated for the selected categories.</p>"
    else:
        for resp_item in html_responses:
            final_html_body += f"<h2>{resp_item['cat_name']} ({resp_item['cat_code']})</h2>\n"
            final_html_body += resp_item["html_resp"]
            final_html_body += "\n<hr />\n" # Add a separator between categories

    # Basic HTML structure for the email/file
    full_html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv Summaries {display_start_date} - {display_end_date}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        hr {{ margin-top: 30px; margin-bottom: 30px; border: 0; border-top: 1px solid #ccc; }}
        /* Styles for explanation-box will be injected by LLM if needed, or add defaults here */
        .explanation-box {{
            background-color: #f0f8ff; /* AliceBlue */
            border: 1px solid #add8e6; /* LightBlue */
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
        }}
        .explanation-box h3 {{
            color: #4682b4; /* SteelBlue */
            margin-top: 0;
        }}
    </style>
</head>
<body>
{final_html_body}
</body>
</html>
"""
    # Ensure 'html_output' directory exists
    os.makedirs("html_output", exist_ok=True)
    output_filename = f"html_output/arxiv_summary_{start_date_str}_to_{end_date_str}.html"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(full_html_output)
        print(f"Final HTML summary saved to: {output_filename}")
    except IOError as e:
        print(f"Error writing HTML file: {e}")

    # Send email if API token and URL are available
    if api_token and api_url:
        email_subject = f"ArXiv AI/ML Research Summary: {display_start_date} - {display_end_date}"
        print(f"Sending email with subject: {email_subject}")
        status_code, email_response = send_email_via_api(
            email_subject,
            full_html_output, # Send the full HTML document
            api_token,
            api_url,
            bypass_proxy=True # As per original, set to False to use system proxies
        )
        print(f"Email API Response - Status: {status_code}, Body: {email_response}")
    else:
        print("Skipping email sending due to missing API_TOKEN or API_URL.")


if __name__ == "__main__":
    # This is a placeholder for constants.py
    # In a real scenario, constants.py would define arxiv_cats
    # For example:
    # arxiv_cats = {
    # "cs.AI": "Artificial Intelligence",
    # "cs.LG": "Machine Learning",
    # "stat.ML": "Statistics - Machine Learning",
    # ... other categories
    # }
    # If constants.py is not present or arxiv_cats is not defined, the script will use codes.
    try:
        from constants import arxiv_cats
    except ImportError:
        print("Warning: constants.py not found or arxiv_cats not defined within. Category names might be codes.")
        arxiv_cats = {} # Define as empty dict to prevent NameError later if not imported

    main()