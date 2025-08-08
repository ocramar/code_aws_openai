import boto3
import json

# Initialize Lambda and Bedrock Agent Runtime clients
lambda_client = boto3.client('lambda')
brt = boto3.client("bedrock-agent-runtime", region_name="eu-west-2")

TARGET_LAMBDA = "test_function_26May2025"
MODEL_ARN = "arn:aws:bedrock:eu-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
KNOWLEDGE_BASE_ID = "ZTPJ2E7RTK"

def lambda_handler(event, context):
    question = event.get("question", "")
    schema_id = event.get("schema_id", "vitic_lhasa")
    session_id = event.get("session_id")  # optional

    if not question:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'question' in input"})
        }

    try:
        # Step 1: Invoke the structured DB summarizer function
        response = lambda_client.invoke(
            FunctionName=TARGET_LAMBDA,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                "question": question,
                "schema_id": schema_id
            }),
        )
        payload = json.loads(response['Payload'].read())
        narrative_response = payload.get("response", "No summary available.")

        # Step 2: Compose combined prompt (embed structured data into input text)
        combined_input = (
            f"[Structured Database Response]\n"
            f"{narrative_response}\n\n"
            f"[Follow-up Question]\n"
            f"{question}"
        )

        # Step 3: Build RAG request (no 'system' key)
        rag_request = {
            "input": {
                "text": combined_input
            },
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN
                }
            }
        }

        # Add sessionId only if provided
        if session_id:
            rag_request["sessionId"] = session_id

        # Step 4: Call Bedrock Agent
        rag_response = brt.retrieve_and_generate(**rag_request)
        final_answer = rag_response["output"]["text"]
        citations = rag_response.get("citations", [])
        new_session_id = rag_response.get("sessionId", None)
        # Print everything BEFORE return
        if citations:
            final_answer += "\n\n[Sources Cited]\n"
            for idx, citation in enumerate(citations, 1):
                refs = citation.get("retrievedReferences", [])
                for ref in refs:
                    uri = ref.get("location", {}).get("s3Location", {}).get("uri", "Unknown URI")
                    snippet = ref.get("content", {}).get("text", "")
                    final_answer += f"{idx}. {uri}\n"
                    if snippet:
                        final_answer += f"   → {snippet[:200]}...\n"
        else:
            final_answer += "\n\n[No citations returned from the knowledge base.]"
        print(f"Structured Database Response (Narrative Summary):\n{narrative_response}\n")
        print(f"Unstructured RAG Response (from Knowledge Base):\n{final_answer}\n")
        print(f"Session ID: {new_session_id}")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "Structured Database Response (Narrative Summary)": narrative_response,
                "Unstructured RAG Response (from Knowledge Base)": final_answer,
                "Original Question": question,
                "session_id": new_session_id,
                "citations": citations
            }, indent=2)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Invocation failed: {str(e)}"})
        }
