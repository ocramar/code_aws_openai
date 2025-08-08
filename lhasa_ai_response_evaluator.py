import traceback
import re
import asyncio
import boto3
import json
from s3_utils import load_json_from_s3
from deepeval.scorer import Scorer
from typing import Optional
from bert_score import score as bert_score_function
from langchain_aws import ChatBedrock
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    BaseMetric,
)


class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model, model_name: Optional[str] = None):
        self.model = model
        self._model_name = model_name or "AWS Bedrock Model"

    def load_model(self):
        return self.model

    def generate(self, prompt: str, **kwargs) -> str:
        return asyncio.run(self.a_generate(prompt, **kwargs))

    async def a_generate(self, prompt: str, **kwargs) -> str:
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt, **kwargs)
        output = response.content

        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                json.loads(match.group(0))
                return match.group(0)
            except json.JSONDecodeError:
                print("invalid JSON response")

        print("prompt: ", str(prompt))
        print("response: ", str(output).replace("\n", "\\n"))
        return output

    def get_model_name(self) -> str:
        return self._model_name


class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()
        self.success: bool = False
        self.score: float = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        if test_case.actual_output is None or test_case.expected_output is None:
            raise ValueError("Both actual_output and expected_output must be strings.")

        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1",
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Rouge Metric"


class BERTScoreMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.85,
        lang: str = "en",
        model_type: str = "bert-base-uncased",
    ):
        self.threshold = threshold
        self.lang = lang
        self.model_type = model_type
        self.score: float = 0.0
        self.success: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        if not isinstance(test_case.actual_output, str) or not isinstance(
            test_case.expected_output, str
        ):
            self.score = 0.0
            self.success = False
            return self.score

        P, R, F1 = bert_score_function(
            [test_case.actual_output],
            [test_case.expected_output],
            lang=self.lang,
            model_type=self.model_type,
            rescale_with_baseline=True,
        )

        score_value = F1[0].item()
        self.score = score_value
        self.success = score_value >= self.threshold
        return score_value

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "BERT Score"


def extract_kb_records(event):
    flattened = []

    for meta in event.get("metadata", []):
        if meta.get("type") == "knowledgeBase":
            flattened.append(meta.get("data"))

    return flattened


def lambda_handler(event, context):
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-west-2")
    custom_model = ChatBedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="eu-west-2",
    )

    aws_bedrock = AWSBedrock(model=custom_model)

    metrics = [
        RougeMetric(),
        BERTScoreMetric(),
        BiasMetric(model=aws_bedrock),
        ToxicityMetric(model=aws_bedrock),
    ]

    if event.get("result_key", None) is not None:
        response = load_json_from_s3(event["result_key"])
        event["actual_answer"] = response.get("response")
        event["metadata"] = response.get("metadata")

    citations = extract_kb_records(event)

    if citations is not None:
        metrics.extend(
            [
                FaithfulnessMetric(model=aws_bedrock),
                AnswerRelevancyMetric(model=aws_bedrock),
                ContextualPrecisionMetric(model=aws_bedrock),
                ContextualRecallMetric(model=aws_bedrock),
                ContextualRelevancyMetric(model=aws_bedrock),
            ]
        )
        retrieval_context = [
            c.get("text_excerpt", "") for c in citations if isinstance(c, dict)
        ]
    else:
        retrieval_context = []

    if event.get("target").get("metric") is not None:
        match = next(metric for metric in metrics if metric.__name__ == event.get("target").get("metric"))
        if(match is not None):
            metrics = [match]

    test_case = LLMTestCase(
        input=event.get("question"),
        actual_output=event.get("actual_answer"),
        expected_output=event.get("expected_answer"),
        retrieval_context=retrieval_context,
    )

    metric_results = []
    for metric in metrics:
        try:
            metric.measure(test_case)
            metric_results.append(
                {
                    "metric": metric.__name__,
                    "score": metric.score,
                    "success": metric.success,
                    "reason": getattr(metric, "reason", None),
                }
            )
        except Exception as e:
            error_message = {
                "metric": metric.__name__,
                "score": None,
                "success": False,
                "error": True,
                "reason": f"Exception during measure(): {str(e)}",
                "traceback": traceback.format_exc(),
                "question": test_case.input,
                "expected_answer": test_case.expected_output,
                "actual_answer": test_case.actual_output,
            }
            print("Metric failed:", error_message)
            metric_results.append(error_message)

    event["metrics"] = metric_results
    event.pop("metadata", None)
    return event
