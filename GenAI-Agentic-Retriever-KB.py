import boto3
import json
import uuid
import logging
import pprint
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# initialize Bedrock agent
bedrock_agent = boto3.client("bedrock-agent-runtime")


def extract_citations_from_chunk(chunk_event):
    """
    Extracts citations from a chunk event
    Args:
        chunk_event: the chunk event
    Returns:
        citations: list of citations
    """
    citations = []
    citations_output = []
    try:
        payload = chunk_event.get("chunk", {})
        if not payload:
            return []

        # Look for citations
        if "attribution" in payload:
            citations = payload["attribution"].get("citations", [])
            for citation in citations:
                generated_text = (
                    citation.get("generatedResponsePart", {})
                    .get("textResponsePart", {})
                    .get("text", "")
                )
                retrieved_refs = citation.get("retrievedReferences", [])

                for ref in retrieved_refs:
                    doc_uri = (
                        ref.get("location", {}).get("s3Location", {}).get("uri", "")
                    )
                    page = ref.get("metadata", {}).get(
                        "x-amz-bedrock-kb-document-page-number", "N/A"
                    )
                    snippet = ref.get("content", {}).get("text", "")
                    metadata = ref.get("metadata", {})

                    citations_output.append(
                        {
                            "text_excerpt": generated_text,
                            "document_uri": doc_uri,
                            "page_number": page,
                            "reference_snippet": snippet,
                            "metadata": metadata
                        }
                    )

    except Exception as e:
        print("Error parsing citation:", e)

    return citations_output


def deduplicate_citations(citations):
    """
    Deduplicate citations
    Args:
        citations: list of citations
    Returns:
        unique: list of unique citations
    """
    seen = set()
    unique = []

    for c in citations:
        # Create a tuple of identifying fields to detect duplicates
        key = (c["document_uri"], c["page_number"])

        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def lambda_handler(event, context):
    """
    Lambda handler for the KB retrieval
    Args:
        event: the event object
            Note: Keep session_id unchanged to preserve agent memory
        context: the context object
    Returns:
        response: the response object
    """

    start_time = time.time()
    logger.info(pprint.pprint(event))
    #agent_id="5NSQXINYPO" #agent-lhasa-dev-documents-pdf-kb
    #agent_alias_id="GZAHDB0PAB" #v5
    agent_id = os.getenv("agentId", '')
    agent_alias_id = os.getenv("agentAliasId", '')
    question = event.get("question", "")
    current_subquery = event.get("current_subquery", "")
    query_history_kb = event.get("query_history_kb", [])
    retrieval_history_kb = event.get("retrieval_history_kb", [])
    retry_count = event.get("retry_count", 0)
    session_id = event.get("session_id", str(uuid.uuid4()))
    metadata = event.get("metadata", {})
    number_of_results = int(event.get("number_of_results", 8))


    # Ensure metadata values are strings
    session_attributes = {k: str(v) for k, v in metadata.items()}
    process_time = round(time.time() - start_time, 4)

    response = bedrock_agent.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,  # keep unchanged to preserve memory
        inputText=current_subquery,
        sessionState={
            "sessionAttributes": session_attributes, # $session_attributes$
            "knowledgeBaseConfigurations": [{
                "knowledgeBaseId": "ZTPJ2E7RTK", 
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results,
                    }
                }
            }]
        }
    )
    response_time = round(time.time() - start_time - process_time, 4)

    logger.info(pprint.pprint(response))

    # Parse the response body - should be treated as an event stream
    event_stream = response['completion']
    final_answer = None
    citations = []  # Initialize citations to avoid unbound variable error
    try:
        for event in event_stream:
            citations = extract_citations_from_chunk(event)
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                final_answer = data.decode("utf8")
                logger.info(f"Final answer ->\n{final_answer}")
            elif "trace" in event:
                logger.info(json.dumps(event["trace"], indent=2))
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)

    parse_time = round(time.time() - start_time - process_time - response_time, 4)
    process_duration =  {
        "total": round(time.time() - start_time, 4),
        "process": process_time,
        "response": response_time,
        "parse": parse_time
    }
                
    # Response with added citations
    retrieval_history_kb = retrieval_history_kb + [{"query": current_subquery, "result": final_answer, "citations": citations}]
    # Increase retry count
    retry_count = retry_count + 1
    
    # prepare and return result
    return {
        "question": question, 
        "current_subquery": current_subquery,
        "query_history_kb": query_history_kb,
        "retrieval_history_kb": retrieval_history_kb,
        "process_duration": process_duration, 
        "retry_count": retry_count,
        "sessionId": session_id 
    }
