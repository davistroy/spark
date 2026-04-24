#!/usr/bin/env python3
"""Quality A/B test for model comparison on DGX Spark.

Runs identical prompts through two model configs and outputs structured results
for manual scoring. Designed for pipeline-specific and general capability testing.

Usage:
    python quality_test.py --url http://localhost:8000 --model spark-llm
    python quality_test.py --url http://localhost:8000 --model gemma4-26b --prompts prompts/pipeline.json
"""

import argparse
import json
import sys
import time
import urllib.request

# Default test prompts covering pipeline-specific and general capability
DEFAULT_PROMPTS = [
    # Pipeline-specific: Entity extraction
    {
        "category": "entity_extraction",
        "name": "Contact center entity extraction",
        "prompt": "Extract all named entities (people, organizations, products, locations) from this contact center transcript excerpt:\n\n\"Hello, this is Sarah from TechCorp calling about our Zendesk integration. We spoke with James in your Portland office last Tuesday about the Enterprise Plus plan. Our account number is AC-4892.\"\n\nReturn a JSON array of objects with fields: text, type, confidence (0-1).",
        "schema": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "type": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["text", "type", "confidence"]}},
    },
    # Pipeline-specific: JSON schema adherence
    {
        "category": "json_schema",
        "name": "Structured claim decomposition",
        "prompt": "Decompose this sentence into atomic claims. Each claim should be independently verifiable.\n\nSentence: \"The new refund policy requires manager approval for returns over $500 and was implemented on March 1st after customer complaints about the 30-day return window.\"\n\nReturn a JSON array of objects with fields: claim (string), type (one of: FACT, RULE, TEMPORAL, CAUSAL).",
        "schema": {"type": "array", "items": {"type": "object", "properties": {"claim": {"type": "string"}, "type": {"type": "string", "enum": ["FACT", "RULE", "TEMPORAL", "CAUSAL"]}}, "required": ["claim", "type"]}},
    },
    # Pipeline-specific: Rule formalization
    {
        "category": "rule_formalization",
        "name": "Policy rule extraction",
        "prompt": "Extract formal business rules from this policy text:\n\n\"Agents must escalate any call where the customer mentions legal action, a regulatory complaint, or media coverage. Escalation goes to the Retention Team Lead during business hours (8am-6pm EST) or the on-call supervisor outside business hours. All escalations must be logged in ServiceNow within 15 minutes.\"\n\nReturn a JSON array of rule objects with: condition (string), action (string), constraint (string or null), priority (HIGH/MEDIUM/LOW).",
        "schema": {"type": "array", "items": {"type": "object", "properties": {"condition": {"type": "string"}, "action": {"type": "string"}, "constraint": {"type": "string"}, "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}}, "required": ["condition", "action", "priority"]}},
    },
    # General: Reasoning
    {
        "category": "reasoning",
        "name": "Multi-step logical reasoning",
        "prompt": "A company has 4 teams: Alpha, Beta, Gamma, Delta. Each team has a different meeting day (Monday-Thursday) and a different project (Web, Mobile, Data, Security).\n\nClues:\n1. The team working on Security meets on Wednesday.\n2. Alpha does not work on Web or Mobile.\n3. Beta meets the day before the Data team.\n4. Gamma meets on Monday.\n5. Delta works on Mobile.\n\nDetermine each team's meeting day and project. Show your reasoning step by step, then give the final answer as a JSON object mapping team names to {day, project}.",
        "schema": None,
    },
    # General: Instruction following
    {
        "category": "instruction_following",
        "name": "Complex multi-constraint instruction",
        "prompt": "Write exactly 3 sentences about cloud computing. The first sentence must contain the word 'scalability'. The second sentence must be a question. The third sentence must be fewer than 10 words. Do not use the word 'data' in any sentence.",
        "schema": None,
    },
    # General: Code generation
    {
        "category": "code_generation",
        "name": "Python function with edge cases",
        "prompt": "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges overlapping intervals. Handle edge cases: empty list, single interval, already sorted, unsorted, touching intervals (e.g., [1,3] and [3,5] should merge). Include type hints and a docstring. No imports needed.",
        "schema": None,
    },
]


def run_prompt(url, model, prompt_data, max_tokens=500):
    """Run a single prompt and return the response with timing."""
    messages = [{"role": "user", "content": prompt_data["prompt"]}]
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    # Add guided JSON if schema provided
    if prompt_data.get("schema"):
        body["extra_body"] = {
            "guided_json": json.dumps(prompt_data["schema"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }
    else:
        body["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    payload = json.dumps(body).encode()
    start = time.monotonic()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        elapsed = time.monotonic() - start
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        usage = resp["usage"]
        return {
            "success": True,
            "content": content,
            "reasoning": msg.get("reasoning", ""),
            "completion_tokens": usage["completion_tokens"],
            "elapsed": elapsed,
            "tok_s": usage["completion_tokens"] / elapsed if elapsed > 0 else 0,
            "finish_reason": resp["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description="Quality A/B test")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--model", default="spark-llm", help="Model name")
    parser.add_argument("--prompts", help="Path to custom prompts JSON file")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per response")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)

    print(f"Model: {args.model}")
    print(f"URL: {args.url}")
    print(f"Prompts: {len(prompts)}")
    print("-" * 60)

    results = []
    for i, prompt_data in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt_data['category']}: {prompt_data['name']}")
        result = run_prompt(args.url, args.model, prompt_data, args.max_tokens)
        result["category"] = prompt_data["category"]
        result["name"] = prompt_data["name"]
        results.append(result)

        if result["success"]:
            print(f"  Tokens: {result['completion_tokens']}, Time: {result['elapsed']:.1f}s, Tok/s: {result['tok_s']:.1f}")
            # Show first 200 chars of content
            content_preview = result["content"][:200].replace("\n", " ")
            print(f"  Response: {content_preview}...")
            # Check if JSON is valid for schema prompts
            if prompt_data.get("schema"):
                try:
                    parsed = json.loads(result["content"])
                    print(f"  JSON valid: Yes ({len(parsed)} items)" if isinstance(parsed, list) else f"  JSON valid: Yes")
                except json.JSONDecodeError:
                    print(f"  JSON valid: NO — response is not valid JSON")
        else:
            print(f"  FAILED: {result['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("SCORING RUBRIC (manual):")
    print("  0 = Failed / wrong / unusable")
    print("  1 = Partially correct, significant issues")
    print("  2 = Good, minor issues")
    print("  3 = Excellent, fully correct")
    print("=" * 60)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
