from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import os

# For pretty printing
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text

import torch
from transformers import AutoTokenizer

# Severity levels are used classify the severity of a security event.
# High severity events are those that should be escalated to a human
# for further investigation.
class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

# Attack types are used to classify security events. This is not an exhaustive
# list of attack vectors!
class AttackType(str, Enum):
    BRUTE_FORCE = "BRUTE_FORCE"
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    FILE_INCLUSION = "FILE_INCLUSION"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    UNKNOWN = "UNKNOWN"

# A WebTrafficPattern is a pattern of traffic to a web server --
# it highlights commonly accessed URLs, methods, and response codes.
#
# WebTrafficPatterns are low-priority summarizations used to help
# with understanding the overall traffic patterns to a web server.
class WebTrafficPattern(BaseModel):
    url_path: str
    http_method: str
    hits_count: int
    response_codes: dict[str, int]  # Maps status code to count
    unique_ips: int

# A LogID is a unique identifier for a log entry. The code in this
# script injects a LOGID-<LETTERS> identifier at the beginning of
# each log entry, which we can use to identify the log entry.
# Language models are fuzzy and they often cannot completely
# copy the original log entry verbatim, so we use the LOGID
# to retrieve the original log entry.
class LogID(BaseModel):
    log_id: str = Field(
        description="""
        The ID of the log entry in the format of LOGID-<LETTERS> where
        <LETTERS> indicates the log identifier at the beginning of
        each log entry.
        """,

        # This is a regular expression that matches the LOGID-<LETTERS> format.
        # The model will fill in the <LETTERS> part.
        pattern=r"LOGID-([A-Z]+)",
    )

    # Find the log entry in a list of logs. Simple
    # conveience function.
    def find_in(self, logs: list[str]) -> Optional[str]:
        for log in logs:
            if self.log_id in log:
                return log
        return None

# Class for an IP address.
class IPAddress(BaseModel):
    ip_address: str = Field(
        pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    )

# Class for an HTTP response code.
class ResponseCode(BaseModel):
    response_code: str = Field(
        pattern=r"^\d{3}$",
    )

# A WebSecurityEvent is a security event that occurred on a web server.
#
# WebSecurityEvents are high-priority events that should be escalated
# to a human for further investigation.
class WebSecurityEvent(BaseModel):
    # The log entry IDs that are relevant to this event.
    relevant_log_entry_ids: list[LogID]

    # The reasoning for why this event is relevant.
    reasoning: str

    # The type of event.
    event_type: str

    # The severity of the event.
    severity: SeverityLevel

    # Whether this event requires human review.
    requires_human_review: bool

    # The confidence score for this event. I'm not sure if this
    # is meaningful for language models, but it's here if we want it.
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

    # Web-specific fields
    url_pattern: str = Field(
        min_length=1,
        description="URL pattern that triggered the event"
    )

    http_method: Literal["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE", "CONNECT"]
    source_ips: list[IPAddress]
    response_codes: list[ResponseCode]
    user_agents: list[str]

    # Possible attack patterns for this event.
    possible_attack_patterns: list[AttackType]

    # Recommended actions for this event.
    recommended_actions: list[str]

# A LogAnalysis is a high-level analysis of a set of logs.
class LogAnalysis(BaseModel):
    # A summary of the analysis.
    summary: str

    # Observations about the logs.
    observations: list[str]

    # Planning for the analysis.
    planning: list[str]

    # Security events found in the logs.
    events: list[WebSecurityEvent]

    # Traffic patterns found in the logs.
    traffic_patterns: list[WebTrafficPattern]

    # The highest severity event found.
    highest_severity: Optional[SeverityLevel]
    requires_immediate_attention: bool

def format_log_analysis(analysis: LogAnalysis, logs: list[str]):
    """Format a LogAnalysis object into a rich console output.

    Args:
        analysis: A LogAnalysis object (not a list)
        logs: List of original log entries with LOGID prefixes
    """
    console = Console()

    # Create header
    header = Panel(
        f"[bold yellow]Log Analysis Report[/]\n[blue]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        border_style="yellow"
    )

    # Create observations section
    observations = Table(show_header=True, header_style="bold magenta", show_lines=True)
    observations.add_column("Key Observations", style="cyan")
    for obs in analysis.observations:
        observations.add_row(obs)

    # Create security events section
    events_table = Table(show_header=True, header_style="bold red", show_lines=True)
    events_table.add_column("Security Events", style="red")
    events_table.add_column("Details", style="yellow")

    # Create a log table if there are any relevant log entry IDs
    event_logs_table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    event_logs_table.add_column("Related Log Entries", style="cyan", width=100)

    for event in analysis.events:
        event_details = [
            f"Type: {event.event_type}",
            f"Severity: {event.severity.value}",
            f"Confidence: {event.confidence_score * 100}%",
            f"Source IPs: {', '.join([ip.ip_address for ip in event.source_ips])}",
            f"URL Pattern: {event.url_pattern}",
            f"Possible Attacks: {', '.join([attack.value for attack in event.possible_attack_patterns])}"
        ]
        events_table.add_row(
            Text(event.event_type, style="bold red"),
            "\n".join(event_details)
        )

        # Add related logs to the table
        for log_id in event.relevant_log_entry_ids:
            log = log_id.find_in(logs)
            if log:
                event_logs_table.add_row(log)

    # Create traffic patterns section
    traffic_table = Table(show_header=True, header_style="bold green", show_lines=True)
    traffic_table.add_column("URL Path", style="green")
    traffic_table.add_column("Method", style="cyan")
    traffic_table.add_column("Hits", style="yellow")
    traffic_table.add_column("Status Codes", style="magenta")

    for pattern in analysis.traffic_patterns:
        traffic_table.add_row(
            pattern.url_path,
            pattern.http_method,
            str(pattern.hits_count),
            ", ".join(f"{k}: {v}" for k, v in pattern.response_codes.items()),
        )

    # Create summary panel
    summary_text = f"[bold white]Summary:[/]\n[cyan]{analysis.summary}[/]\n\n"
    if analysis.highest_severity:
        summary_text += f"[bold red]Highest Severity: {analysis.highest_severity.value}[/]\n"
    summary_text += f"[bold {'red' if analysis.requires_immediate_attention else 'green'}]" + \
                   f"Requires Immediate Attention: {analysis.requires_immediate_attention}[/]"

    summary = Panel(
        summary_text,
        border_style="blue"
    )

    # Print everything
    console.print(header)
    console.print("\n[bold blue]📝 Analysis Summary:[/]")
    console.print(summary)
    console.print(observations)
    console.print("\n[bold red]⚠️  Security Events:[/]")
    console.print(events_table)
    console.print(event_logs_table)
    console.print("\n[bold green]📊 Traffic Patterns:[/]")
    console.print(traffic_table)

class STRESSED:
    def __init__(
        self,
        model,
        tokenizer,
        log_type: str,
        prompt_template_path: str,
        token_max: int,
        stressed_out: bool = False
    ):
        if token_max <= 0:
            raise ValueError("token_max must be positive")
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

        self.model = model
        self.tokenizer = tokenizer
        self.log_type = log_type
        self.token_max = token_max
        self.stressed_out = stressed_out
        # Load prompt template
        with open(prompt_template_path, "r") as file:
            self.prompt_template = file.read()

        # Initialize generator
        self.logger = outlines.generate.json(
            self.model,
            LogAnalysis,
            sampler=outlines.samplers.greedy(),
        )

    def _to_prompt(self, text: str, pydantic_class: BaseModel) -> str:
        if self.stressed_out:
            stress_prompt = """
            You are a computer security intern that's really stressed out.
            Your job is hard and you're not sure you're doing it well.

            Your observations and summaries should reflect your anxiety.
            Convey a sense of urgency and panic, be apologetic, and
            generally act like you're not sure you can do your job.

            In your summary, address your boss as "boss" and apologize for
            any mistakes you've made even if you haven't made any.

            Use "um" and "ah" a lot.
            """
        else:
            stress_prompt = ""

        messages = []

        if self.stressed_out:
            messages.append({"role": "system", "content": stress_prompt})

        messages.append(
            {"role": "user", "content": self.prompt_template.format(
                log_type=self.log_type,
                logs=text,
                model_schema=pydantic_class.model_json_schema(),
                stress_prompt=stress_prompt,
            )}
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def analyze_logs(
        self,
        logs: list[str],
        chunk_size: int = 10,
        format_output: bool = True
    ) -> list[LogAnalysis]:
        """
        Analyze a list of log entries.

        Args:
            logs: List of log entries to analyze
            chunk_size: Number of logs to analyze at once
            format_output: Whether to print formatted output

        Returns:
            List of LogAnalysis objects
        """
        results = []

        for i in range(0, len(logs), chunk_size):
            chunked_logs = [log for log in logs[i:i+chunk_size] if log]

            if not chunked_logs:
                continue

            # Create log IDs
            log_ids = [f"LOGID-{chr(65 + (j // 26) % 26)}{chr(65 + j % 26)}"
                      for j in range(len(chunked_logs))]

            logs_with_ids = [f"{log_id} {log}"
                            for log_id, log in zip(log_ids, chunked_logs)]
            chunk = "\n".join(logs_with_ids)

            # Analyze chunk
            prompt = self._to_prompt(chunk, LogAnalysis)
            analysis = self.logger(prompt, max_tokens=self.token_max)

            if format_output:
                format_log_analysis(analysis, logs_with_ids)

            results.append(analysis)

        return results

import outlines

# The model we're using
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# The type of logs we're parsing. You don't have to use this, but it's
# helpful for the model to understand the context of the logs.
log_type = "web server"

# The path to the prompt template we're using. This should be a file in
# the repo.
prompt_template_path = "/content/security-prompt.txt"

# Load the model
model = outlines.models.vllm(
    # The model we're using
    model_name,

    # The dtype to use for the model. bfloat16 is faster
    # than the native float size.
    dtype=torch.bfloat16,

    # Enable prefix caching for faster inference
    enable_prefix_caching=True,

    # Disable sliding window -- this is required
    # for prefix caching to work.
    disable_sliding_window=True,

    # The maximum sequence length for the model.
    # Modify this if you have more memory available,
    # and/or if your logs are longer.
    max_model_len=32000,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initializing
parser = STRESSED(
    model=model,
    tokenizer=tokenizer,
    log_type=log_type,
    prompt_template_path=prompt_template_path,
    token_max=32000,  # Maximum tokens to generate
    stressed_out=True
)

# Load the logs you want to parse.
test_logs = [
    # Linux system log
    "/content/linux-2k.log",
    "/content/apache-10k.log",
    "/content/access-10k.log"
]

# Choose the access log for giggles
log_path = test_logs[0]

# Load the logs into memory
with open(log_path, "r") as file:
    logs = file.readlines()

print(logs[:2])

# Start the analysis
results = parser.analyze_logs(
    logs,

    # Chunk the logs into 20 lines at a time.
    # Using a higher number can degenerate the model's performance,
    # but it will generally be faster.
    chunk_size=20,

    # Format output prints a helpful display in your terminal.
    format_output=True
)

# You can do stuff with the results here. results is a list of LogAnalysis objects.
for analysis in results:
    # Do stuff with the analysis
    print(analysis.summary)

# Choose the access log for giggles
log_path = test_logs[1]

# Load the logs into memory
with open(log_path, "r") as file:
    logs = file.readlines()

# Start the analysis
results = parser.analyze_logs(
    logs[:2500],

    # Chunk the logs into 20 lines at a time.
    # Using a higher number can degenerate the model's performance,
    # but it will generally be faster.
    chunk_size=20,

    # Format output prints a helpful display in your terminal.
    format_output=True
)

# You can do stuff with the results here. results is a list of LogAnalysis objects.
for analysis in results:
    # Do stuff with the analysis
    print(analysis.summary)

print(results)

def flatten_log_analyses(logs: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    import pandas as pd
    from collections import defaultdict

    event_records = []
    traffic_records = []
    severity_list = []

    for log in logs:
        # Safe get severity
        if hasattr(log, 'highest_severity') and log.highest_severity:
            severity_list.append(log.highest_severity.value)
        else:
            severity_list.append("UNKNOWN")

        # Events
        if hasattr(log, 'events') and log.events:
            for event in log.events:
                try:
                    event_records.append({
                        'event_type': getattr(event, 'event_type', 'UNKNOWN'),
                        'severity': getattr(event.severity, 'value', 'OK'),
                        'confidence': getattr(event, 'confidence_score', None),
                        'url_pattern': getattr(event, 'url_pattern', 'UNKNOWN'),
                        'http_method': getattr(event, 'http_method', 'UNKNOWN'),
                        'source_ips': [ip.ip_address for ip in getattr(event, 'source_ips', [])],
                        'attack_patterns': [a.value for a in getattr(event, 'possible_attack_patterns', [])],
                        'log_ids': [lid.log_id for lid in getattr(event, 'relevant_log_entry_ids', [])],
                    })
                except Exception as e:
                    print(f"Error parsing event: {e}")

        # Traffic patterns
        if hasattr(log, 'traffic_patterns') and log.traffic_patterns:
            for pattern in log.traffic_patterns:
                try:
                    traffic_records.append({
                        'url_path': getattr(pattern, 'url_path', 'UNKNOWN'),
                        'method': getattr(pattern, 'http_method', 'UNKNOWN'),
                        'hits': getattr(pattern, 'hits_count', 0),
                        'response_codes': getattr(pattern, 'response_codes', {}),
                        'unique_ips': getattr(pattern, 'unique_ips', 0)
                    })
                except Exception as e:
                    print(f"Error parsing traffic pattern: {e}")

    df_events = pd.DataFrame(event_records)
    df_traffic = pd.DataFrame(traffic_records)
    df_severity = pd.Series(severity_list, name="severity")

    return df_events, df_traffic, df_severity

df_events, df_traffic, df_severity = flatten_log_analyses(results)

df_events.head()

df_traffic.head()

df_severity.head()

import matplotlib.pyplot as plt

severity_counts = df_severity.value_counts()

severity_counts.plot(kind='bar', color='salmon', title='Log Severity Summary')
plt.ylabel("Count")
plt.xlabel("Severity")
plt.xticks(rotation=0)
plt.show()

event_type_map = {
    'Brute Force': 'Brute Force Attack',
    '': 'Brute Force Attack',
    'BRUTE_FORCE': 'Brute Force Attack'
}

# Apply the mapping
df_events['event_type'] = df_events['event_type'].map(
    lambda x: event_type_map.get(x, x)
)

df_events['event_type'].value_counts().plot(kind='barh', title='Security Event Types', color='orange')
plt.xlabel("Occurrences")
plt.tight_layout()
plt.show()

from collections import Counter
from itertools import chain

all_ips = list(chain.from_iterable(df_events['source_ips']))
ip_counts = pd.Series(Counter(all_ips)).sort_values(ascending=False).head(10)

ip_counts.plot(kind='bar', color='teal', title='Top Source IPs')
plt.ylabel("Events")
plt.xlabel("IP Address")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

all_attacks = list(chain.from_iterable(df_events['attack_patterns']))
all_attacks = [attack for attack in all_attacks if attack != 'UNKNOWN']
attack_counts = pd.Series(Counter(all_attacks)).sort_values(ascending=True)

attack_counts.plot(kind='barh', color='darkred', title='Detected Attack Patterns')
plt.xlabel("Count")
plt.tight_layout()
plt.show()

import plotly.express as px

if not df_traffic.empty:
    # Pivot table to aggregate traffic per URL
    traffic_plot = df_traffic.pivot_table(index="url_path", values="hits", aggfunc="sum").reset_index()

    # Plot using Plotly
    fig = px.bar(traffic_plot,
                 y="url_path",
                 x="hits",
                 orientation='h',
                 title="Traffic per URL")

    # Update layout for scrollable chart
    fig.update_layout(
        autosize=True,
        margin=dict(l=100, r=100, t=40, b=40),
        height=800,  # Adjust height for better visibility
    )

    # Update bar colors to navy
    fig.update_traces(marker=dict(color='navy'))

    fig.show()

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import torch
# from transformers import AutoTokenizer
# import outlines
# import os
# from pydantic import BaseModel
# from typing import Optional
# 
# # Imports
# from enum import Enum
# from typing import Literal, Optional
# from pydantic import BaseModel, Field
# from datetime import datetime
# import os
# 
# # For pretty printing
# from rich import print
# from rich.panel import Panel
# from rich.table import Table
# from rich.console import Console
# from rich.text import Text
# 
# import torch
# from transformers import AutoTokenizer
# # Severity levels are used classify the severity of a security event.
# # High severity events are those that should be escalated to a human
# # for further investigation.
# class SeverityLevel(str, Enum):
#     CRITICAL = "CRITICAL"
#     HIGH = "HIGH"
#     MEDIUM = "MEDIUM"
#     LOW = "LOW"
#     INFO = "INFO"
# 
# # Attack types are used to classify security events. This is not an exhaustive
# # list of attack vectors!
# class AttackType(str, Enum):
#     BRUTE_FORCE = "BRUTE_FORCE"
#     SQL_INJECTION = "SQL_INJECTION"
#     XSS = "XSS"
#     FILE_INCLUSION = "FILE_INCLUSION"
#     COMMAND_INJECTION = "COMMAND_INJECTION"
#     PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
#     UNKNOWN = "UNKNOWN"
# 
# # A WebTrafficPattern is a pattern of traffic to a web server --
# # it highlights commonly accessed URLs, methods, and response codes.
# #
# # WebTrafficPatterns are low-priority summarizations used to help
# # with understanding the overall traffic patterns to a web server.
# class WebTrafficPattern(BaseModel):
#     url_path: str
#     http_method: str
#     hits_count: int
#     response_codes: dict[str, int]  # Maps status code to count
#     unique_ips: int
# # A LogID is a unique identifier for a log entry. The code in this
# # script injects a LOGID-<LETTERS> identifier at the beginning of
# # each log entry, which we can use to identify the log entry.
# # Language models are fuzzy and they often cannot completely
# # copy the original log entry verbatim, so we use the LOGID
# # to retrieve the original log entry.
# class LogID(BaseModel):
#     log_id: str = Field(
#         description="""
#         The ID of the log entry in the format of LOGID-<LETTERS> where
#         <LETTERS> indicates the log identifier at the beginning of
#         each log entry.
#         """,
# 
#         # This is a regular expression that matches the LOGID-<LETTERS> format.
#         # The model will fill in the <LETTERS> part.
#         pattern=r"LOGID-([A-Z]+)",
#     )
# 
#     # Find the log entry in a list of logs. Simple
#     # conveience function.
#     def find_in(self, logs: list[str]) -> Optional[str]:
#         for log in logs:
#             if self.log_id in log:
#                 return log
#         return None
# # Class for an IP address.
# class IPAddress(BaseModel):
#     ip_address: str = Field(
#         pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
#     )
# 
# # Class for an HTTP response code.
# class ResponseCode(BaseModel):
#     response_code: str = Field(
#         pattern=r"^\d{3}$",
#     )
# # A WebSecurityEvent is a security event that occurred on a web server.
# #
# # WebSecurityEvents are high-priority events that should be escalated
# # to a human for further investigation.
# class WebSecurityEvent(BaseModel):
#     # The log entry IDs that are relevant to this event.
#     relevant_log_entry_ids: list[LogID]
# 
#     # The reasoning for why this event is relevant.
#     reasoning: str
# 
#     # The type of event.
#     event_type: str
# 
#     # The severity of the event.
#     severity: SeverityLevel
# 
#     # Whether this event requires human review.
#     requires_human_review: bool
# 
#     # The confidence score for this event. I'm not sure if this
#     # is meaningful for language models, but it's here if we want it.
#     confidence_score: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="Confidence score between 0 and 1"
#     )
# 
#     # Web-specific fields
#     url_pattern: str = Field(
#         min_length=1,
#         description="URL pattern that triggered the event"
#     )
# 
#     http_method: Literal["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE", "CONNECT"]
#     source_ips: list[IPAddress]
#     response_codes: list[ResponseCode]
#     user_agents: list[str]
# 
#     # Possible attack patterns for this event.
#     possible_attack_patterns: list[AttackType]
# 
#     # Recommended actions for this event.
#     recommended_actions: list[str]
# # A LogAnalysis is a high-level analysis of a set of logs.
# class LogAnalysis(BaseModel):
#     # A summary of the analysis.
#     summary: str
# 
#     # Observations about the logs.
#     observations: list[str]
# 
#     # Planning for the analysis.
#     planning: list[str]
# 
#     # Security events found in the logs.
#     events: list[WebSecurityEvent]
# 
#     # Traffic patterns found in the logs.
#     traffic_patterns: list[WebTrafficPattern]
# 
#     # The highest severity event found.
#     highest_severity: Optional[SeverityLevel]
#     requires_immediate_attention: bool
# def format_log_analysis(analysis: LogAnalysis, logs: list[str]):
#     """Format a LogAnalysis object into a rich console output.
# 
#     Args:
#         analysis: A LogAnalysis object (not a list)
#         logs: List of original log entries with LOGID prefixes
#     """
#     console = Console()
# 
#     # Create header
#     header = Panel(
#         f"[bold yellow]Log Analysis Report[/]\n[blue]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
#         border_style="yellow"
#     )
# 
#     # Create observations section
#     observations = Table(show_header=True, header_style="bold magenta", show_lines=True)
#     observations.add_column("Key Observations", style="cyan")
#     for obs in analysis.observations:
#         observations.add_row(obs)
# 
#     # Create security events section
#     events_table = Table(show_header=True, header_style="bold red", show_lines=True)
#     events_table.add_column("Security Events", style="red")
#     events_table.add_column("Details", style="yellow")
# 
#     # Create a log table if there are any relevant log entry IDs
#     event_logs_table = Table(show_header=True, header_style="bold cyan", show_lines=True)
#     event_logs_table.add_column("Related Log Entries", style="cyan", width=100)
# 
#     for event in analysis.events:
#         event_details = [
#             f"Type: {event.event_type}",
#             f"Severity: {event.severity.value}",
#             f"Confidence: {event.confidence_score * 100}%",
#             f"Source IPs: {', '.join([ip.ip_address for ip in event.source_ips])}",
#             f"URL Pattern: {event.url_pattern}",
#             f"Possible Attacks: {', '.join([attack.value for attack in event.possible_attack_patterns])}"
#         ]
#         events_table.add_row(
#             Text(event.event_type, style="bold red"),
#             "\n".join(event_details)
#         )
# 
#         # Add related logs to the table
#         for log_id in event.relevant_log_entry_ids:
#             log = log_id.find_in(logs)
#             if log:
#                 event_logs_table.add_row(log)
# 
#     # Create traffic patterns section
#     traffic_table = Table(show_header=True, header_style="bold green", show_lines=True)
#     traffic_table.add_column("URL Path", style="green")
#     traffic_table.add_column("Method", style="cyan")
#     traffic_table.add_column("Hits", style="yellow")
#     traffic_table.add_column("Status Codes", style="magenta")
# 
#     for pattern in analysis.traffic_patterns:
#         traffic_table.add_row(
#             pattern.url_path,
#             pattern.http_method,
#             str(pattern.hits_count),
#             ", ".join(f"{k}: {v}" for k, v in pattern.response_codes.items()),
#         )
# 
#     # Create summary panel
#     summary_text = f"[bold white]Summary:[/]\n[cyan]{analysis.summary}[/]\n\n"
#     if analysis.highest_severity:
#         summary_text += f"[bold red]Highest Severity: {analysis.highest_severity.value}[/]\n"
#     summary_text += f"[bold {'red' if analysis.requires_immediate_attention else 'green'}]" + \
#                    f"Requires Immediate Attention: {analysis.requires_immediate_attention}[/]"
# 
#     summary = Panel(
#         summary_text,
#         border_style="blue"
#     )
# 
#     # Print everything
#     console.print(header)
#     console.print("\n[bold blue]📝 Analysis Summary:[/]")
#     console.print(summary)
#     console.print(observations)
#     console.print("\n[bold red]⚠️  Security Events:[/]")
#     console.print(events_table)
#     console.print(event_logs_table)
#     console.print("\n[bold green]📊 Traffic Patterns:[/]")
#     console.print(traffic_table)
# 
# # Define the STRESSED class as previously
# class STRESSED:
#     def __init__(
#         self,
#         model,
#         tokenizer,
#         log_type: str,
#         prompt_template_path: str,
#         token_max: int,
#         stressed_out: bool = False
#     ):
#         if token_max <= 0:
#             raise ValueError("token_max must be positive")
#         if not os.path.exists(prompt_template_path):
#             raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
# 
#         self.model = model
#         self.tokenizer = tokenizer
#         self.log_type = log_type
#         self.token_max = token_max
#         self.stressed_out = stressed_out
#         # Load prompt template
#         with open(prompt_template_path, "r") as file:
#             self.prompt_template = file.read()
# 
#         # Initialize generator
#         self.logger = outlines.generate.json(
#             self.model,
#             LogAnalysis,
#             sampler=outlines.samplers.greedy(),
#         )
# 
#     def _to_prompt(self, text: str, pydantic_class: BaseModel) -> str:
#         if self.stressed_out:
#             stress_prompt = """
#             You are a computer security intern that's really stressed out.
#             Your job is hard and you're not sure you're doing it well.
# 
#             Your observations and summaries should reflect your anxiety.
#             Convey a sense of urgency and panic, be apologetic, and
#             generally act like you're not sure you can do your job.
# 
#             In your summary, address your boss as "boss" and apologize for
#             any mistakes you've made even if you haven't made any.
# 
#             Use "um" and "ah" a lot.
#             """
#         else:
#             stress_prompt = ""
# 
#         messages = []
#         if self.stressed_out:
#             messages.append({"role": "system", "content": stress_prompt})
# 
#         messages.append(
#             {"role": "user", "content": self.prompt_template.format(
#                 log_type=self.log_type,
#                 logs=text,
#                 model_schema=pydantic_class.model_json_schema(),
#                 stress_prompt=stress_prompt,
#             )}
#         )
# 
#         return self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
# 
#     def analyze_logs(
#         self,
#         logs: list[str],
#         chunk_size: int = 10,
#         format_output: bool = True
#     ) -> list[LogAnalysis]:
#         results = []
#         for i in range(0, len(logs), chunk_size):
#             chunked_logs = [log for log in logs[i:i+chunk_size] if log]
#             if not chunked_logs:
#                 continue
#             log_ids = [f"LOGID-{chr(65 + (j // 26) % 26)}{chr(65 + j % 26)}"
#                       for j in range(len(chunked_logs))]
#             logs_with_ids = [f"{log_id} {log}" for log_id, log in zip(log_ids, chunked_logs)]
#             chunk = "\n".join(logs_with_ids)
#             prompt = self._to_prompt(chunk, LogAnalysis)
#             analysis = self.logger(prompt, max_tokens=self.token_max)
#             if format_output:
#                 format_log_analysis(analysis, logs_with_ids)
#             results.append(analysis)
#         return results
# 
# # Streamlit UI for log analysis
# def run_log_analysis_app():
#     st.title("Log File Analysis with Security Prompt")
# 
#     # Upload log file
#     log_file = st.file_uploader("Upload Log File", type=["txt", "log"])
#     if log_file:
#         log_content = log_file.read().decode("utf-8").splitlines()
#         st.write(f"Uploaded Log File: {log_file.name}")
#         st.text_area("Log Content", "\n".join(log_content), height=200)
# 
#     # Upload the security prompt template file
#     prompt_file = st.file_uploader("Upload Security Prompt Template", type=["txt"])
#     if prompt_file:
#         prompt_content = prompt_file.read().decode("utf-8")
#         st.write(f"Uploaded Prompt Template File: {prompt_file.name}")
# 
#     # Choose whether to stress the analysis
#     stressed_out = st.checkbox("Stressed Out Intern Mode", value=True)
# 
#     # Configuration
#     token_max = st.slider("Max Tokens for Generation", min_value=500, max_value=32000, value=32000, step=500)
#     model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
#     log_type = "web server"
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 
#     if st.button("Analyze Logs"):
#         # Load the model and tokenizer
#         model = outlines.models.vllm(
#             model_name,
#             dtype=torch.bfloat16,
#             enable_prefix_caching=True,
#             disable_sliding_window=True,
#             max_model_len=token_max,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
# 
#         # Initialize the STRESSED parser
#         parser = STRESSED(
#             model=model,
#             tokenizer=tokenizer,
#             log_type=log_type,
#             prompt_template_path=prompt_file.name,
#             token_max=token_max,
#             stressed_out=stressed_out
#         )
# 
#         # Run the analysis
#         results = parser.analyze_logs(
#             log_content,
#             chunk_size=20,
#             format_output=False
#         )
# 
#         # Display results
#         for analysis in results:
#             st.subheader("Log Analysis Summary")
#             st.write(analysis.summary)
#             st.subheader("Observations")
#             st.write("\n".join(analysis.observations))
#             st.subheader("Security Events")
#             for event in analysis.events:
#                 st.write(f"**Event Type:** {event.event_type}")
#                 st.write(f"**Severity:** {event.severity.value}")
#                 st.write(f"**Confidence Score:** {event.confidence_score * 100}%")
#                 st.write(f"**Source IPs:** {', '.join([ip.ip_address for ip in event.source_ips])}")
#                 st.write(f"**URL Pattern:** {event.url_pattern}")
#                 st.write(f"**Possible Attacks:** {', '.join([attack.value for attack in event.possible_attack_patterns])}")
#                 st.write(f"**Recommended Actions:** {', '.join(event.recommended_actions)}")
#                 st.write("---")
# 
# if __name__ == "__main__":
#     run_log_analysis_app()

# import urllib
# print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

# !streamlit run /content/app.py & npx localtunnel --port 8501

