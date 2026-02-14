"""External Benchmark Evaluation for BIOAgents Healthcare AI GYM.

Evaluates trained models on standard medical benchmarks:

Text QA (8 benchmarks):
- MedQA (USMLE-style MC questions)
- MedMCQA (Indian medical entrance MC)
- MMLU Medical x6 (clinical, professional, anatomy, genetics, biology, college_med)

Visual QA (6 benchmarks — via VQABenchmarkEvaluator):
- VQA-RAD, SLAKE, PathVQA, PMC-VQA, VQA-Med-2021, Quilt-VQA

EHR Benchmarks (2 databases — via EHRBenchmarkEvaluator):
- MIMIC-III Clinical Database v1.4 (50 ICU patients, 50 agent tasks)
- eICU Collaborative Research Database v2.0 (50 ICU patients, 50 agent tasks)

These are the official benchmarks that provide comparable results
with other published systems. For VQA benchmarks, use:
    from bioagents.evaluation.vqa_benchmark_eval import VQABenchmarkEvaluator
For EHR benchmarks, use:
    from bioagents.evaluation.ehr_benchmark_eval import EHRBenchmarkEvaluator
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class BenchmarkConfig:
    model_name_or_path: str
    model_name: str = "BIOAgent"
    backend: str = "transformers"
    benchmarks: list[str] = field(
        default_factory=lambda: ["medqa", "medmcqa", "mmlu_medical"]
    )
    max_samples: int = 0  # 0 = all
    batch_size: int = 8
    output_dir: str = "logs/benchmarks"
    few_shot: int = 0  # number of few-shot examples
    temperature: float = 0.0  # greedy for benchmarks
    max_new_tokens: int = 256


class BenchmarkEvaluator:
    """Evaluate models on standard medical benchmarks."""

    # Benchmark data paths
    BENCHMARK_PATHS = {
        "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
        "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
        "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
        "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
        "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
        "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
        "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
        "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
    }

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the model for evaluation."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.config.model_name_or_path}")

        # Auto-repair config for cross-version transformers compatibility
        from pathlib import Path
        if Path(self.config.model_name_or_path).is_dir():
            from bioagents.evaluation.agent_runner import repair_model_config
            repair_model_config(self.config.model_name_or_path)

        model_config = AutoConfig.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        model_type = getattr(model_config, "model_type", "")
        is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        if is_qwen_vl:
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name_or_path, **load_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path, **load_kwargs
            )

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded: {model_type}, params={sum(p.numel() for p in self.model.parameters())/1e6:.0f}M")

    def evaluate_all(self) -> dict:
        """Evaluate on all configured benchmarks."""
        if self.model is None:
            self.load_model()

        all_results = {}
        for benchmark in self.config.benchmarks:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Evaluating on: {benchmark}")
            logger.info(f"{'='*60}")

            try:
                results = self.evaluate_benchmark(benchmark)
                all_results[benchmark] = results
                logger.info(
                    f"  {benchmark}: accuracy={results['accuracy']:.3f} "
                    f"({results['correct']}/{results['total']})"
                )
            except Exception as e:
                logger.error(f"  Error evaluating {benchmark}: {e}")
                import traceback
                traceback.print_exc()
                all_results[benchmark] = {"error": str(e)}

        # Save results
        self._save_results(all_results)
        self._print_summary(all_results)

        return all_results

    def evaluate_benchmark(self, benchmark: str) -> dict:
        """Evaluate on a single benchmark."""
        # Load data
        data = self._load_benchmark_data(benchmark)
        if not data:
            return {"error": f"No data found for {benchmark}"}

        if self.config.max_samples > 0:
            data = data[: self.config.max_samples]

        correct = 0
        total = 0
        per_question = []

        for i, item in enumerate(data):
            question, options, correct_answer = self._parse_item(item, benchmark)
            if question is None:
                continue

            # Build prompt
            prompt = self._build_mc_prompt(question, options, benchmark)

            # Generate
            predicted = self._generate_answer(prompt)

            # Extract answer letter
            pred_letter = self._extract_answer_letter(predicted, list(options.keys()))
            is_correct = pred_letter == correct_answer

            if is_correct:
                correct += 1
            total += 1

            per_question.append(
                {
                    "question_id": item.get("id", i),
                    "predicted": pred_letter,
                    "correct": correct_answer,
                    "is_correct": is_correct,
                }
            )

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Progress: {i+1}/{len(data)}, "
                    f"accuracy={correct/total:.3f} ({correct}/{total})"
                )

        return {
            "benchmark": benchmark,
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "per_question": per_question,
        }

    def _load_benchmark_data(self, benchmark: str) -> list[dict]:
        """Load benchmark data."""
        rel_path = self.BENCHMARK_PATHS.get(benchmark)
        if rel_path is None:
            logger.error(f"Unknown benchmark: {benchmark}")
            return []

        data_path = PROJECT_ROOT / rel_path
        if not data_path.exists():
            logger.error(f"Benchmark data not found: {data_path}")
            return []

        data = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        logger.info(f"  Loaded {len(data)} items from {data_path}")
        return data

    def _parse_item(self, item: dict, benchmark: str):
        """Parse a benchmark item into (question, options, correct_answer).

        Self-BioRAG format:
        {
            "id": "seed_task_0",
            "name": "med_qa",
            "instruction": "Given four answer candidates...",
            "instances": {
                "input": "QUESTION: ... Option A: ... Option B: ...",
                "output": "text of correct answer"
            }
        }
        """
        # Self-BioRAG unified format
        instances = item.get("instances", {})
        raw_input = instances.get("input", "")
        raw_output = instances.get("output", "")

        if not raw_input:
            # Try direct format
            question = item.get("question", "")
            options = item.get("options", {})
            if isinstance(options, list):
                options = {chr(65 + i): opt for i, opt in enumerate(options)}
            correct = item.get("answer_idx", item.get("answer", ""))
            return question, options, correct

        # Parse Self-BioRAG format: "QUESTION: ... Option A: ... Option B: ..."
        question = ""
        options = {}

        # Extract question
        if "QUESTION:" in raw_input:
            parts = raw_input.split("QUESTION:", 1)
            rest = parts[1].strip()
        else:
            rest = raw_input.strip()

        # Extract options
        import re
        option_pattern = r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)'
        option_matches = re.findall(option_pattern, rest, re.DOTALL)

        if option_matches:
            # Question is everything before the first option
            first_opt_pos = rest.find("Option A:")
            if first_opt_pos == -1:
                first_opt_pos = rest.find("Option B:")
            if first_opt_pos > 0:
                question = rest[:first_opt_pos].strip().rstrip('\n')

            for letter, text in option_matches:
                options[letter.strip()] = text.strip()
        else:
            question = rest

        # Determine correct answer letter from output text
        correct_letter = ""
        if raw_output:
            raw_output_clean = raw_output.strip()
            # Match output text to option text
            for letter, text in options.items():
                if text.strip().lower() == raw_output_clean.lower():
                    correct_letter = letter
                    break
                # Partial match
                if raw_output_clean.lower() in text.lower() or text.lower() in raw_output_clean.lower():
                    correct_letter = letter
                    break

            # If still not found, check if output starts with a letter
            if not correct_letter and raw_output_clean and raw_output_clean[0].upper() in options:
                correct_letter = raw_output_clean[0].upper()

            if not correct_letter:
                correct_letter = "B"  # fallback

        return question, options, correct_letter

    def _build_mc_prompt(
        self, question: str, options: dict, benchmark: str
    ) -> str:
        """Build a multiple-choice prompt."""
        prompt = f"Question: {question}\n\nOptions:\n"
        for letter, text in sorted(options.items()):
            prompt += f"  {letter}) {text}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """Generate an answer from the model."""
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert taking a medical exam. Answer each question with only the correct option letter.",
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _extract_answer_letter(self, response: str, valid_letters: list[str]) -> str:
        """Extract the answer letter from model response."""
        response = response.strip()

        # Direct letter
        if response and response[0].upper() in valid_letters:
            return response[0].upper()

        # Pattern: "The answer is X" or "Answer: X"
        patterns = [
            r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?",
            r"\b([A-D])\b\s*(?:\)|\.|\:)",
            r"^\s*\(?([A-D])\)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Last resort: find any letter
        for letter in valid_letters:
            if letter in response.upper():
                return letter

        return "A"  # default

    def _save_results(self, all_results: dict):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(self.config.output_dir) / f"{self.config.model_name}_{timestamp}.json"
        )

        report = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_name_or_path,
            "benchmarks": {},
            "timestamp": datetime.now().isoformat(),
        }

        for bench, results in all_results.items():
            if "error" not in results:
                report["benchmarks"][bench] = {
                    "accuracy": results["accuracy"],
                    "correct": results["correct"],
                    "total": results["total"],
                }
            else:
                report["benchmarks"][bench] = {"error": results["error"]}

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def _print_summary(self, all_results: dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS: {self.config.model_name}")
        print(f"{'='*60}")

        for bench, results in all_results.items():
            if "error" in results:
                print(f"  {bench}: ERROR - {results['error'][:60]}")
            else:
                print(
                    f"  {bench}: {results['accuracy']:.1%} "
                    f"({results['correct']}/{results['total']})"
                )

        overall = [
            r["accuracy"] for r in all_results.values() if "accuracy" in r
        ]
        if overall:
            print(f"  {'─'*50}")
            print(f"  AVERAGE: {sum(overall)/len(overall):.1%}")
        print(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Medical Benchmark Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--model-name", default="BIOAgent", help="Model name for logging")
    # Text QA benchmarks
    TEXT_BENCHMARKS = [
        "medqa", "medmcqa", "mmlu_clinical", "mmlu_professional",
        "mmlu_anatomy", "mmlu_genetics", "mmlu_biology", "mmlu_college_med",
    ]
    # VQA benchmarks (handled by VQABenchmarkEvaluator)
    VQA_BENCHMARKS = [
        "vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa",
    ]
    # EHR benchmarks (handled by EHRBenchmarkEvaluator)
    EHR_BENCHMARKS = [
        "mimic_iii", "eicu",
    ]

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["medqa", "medmcqa", "mmlu_clinical", "mmlu_professional"],
        choices=TEXT_BENCHMARKS + VQA_BENCHMARKS + EHR_BENCHMARKS,
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--output-dir", default="logs/benchmarks")

    args = parser.parse_args()

    # Split benchmarks into text, VQA, and EHR
    text_benchmarks = [b for b in args.benchmarks if b in TEXT_BENCHMARKS]
    vqa_benchmarks = [b for b in args.benchmarks if b in VQA_BENCHMARKS]
    ehr_benchmarks = [b for b in args.benchmarks if b in EHR_BENCHMARKS]

    # Run text QA benchmarks
    if text_benchmarks:
        config = BenchmarkConfig(
            model_name_or_path=args.model,
            model_name=args.model_name,
            benchmarks=text_benchmarks,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
        evaluator = BenchmarkEvaluator(config)
        evaluator.evaluate_all()

    # Run VQA benchmarks
    if vqa_benchmarks:
        from bioagents.evaluation.vqa_benchmark_eval import (
            VQABenchmarkConfig,
            VQABenchmarkEvaluator,
        )

        vqa_config = VQABenchmarkConfig(
            model_name_or_path=args.model,
            model_name=args.model_name,
            benchmarks=vqa_benchmarks,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
        vqa_evaluator = VQABenchmarkEvaluator(vqa_config)
        vqa_evaluator.evaluate_all()

    # Run EHR benchmarks (MIMIC-III, eICU)
    if ehr_benchmarks:
        from bioagents.evaluation.ehr_benchmark_eval import (
            EHRBenchmarkConfig,
            EHRBenchmarkEvaluator,
        )

        ehr_config = EHRBenchmarkConfig(
            model_name_or_path=args.model,
            model_name=args.model_name,
            benchmarks=ehr_benchmarks,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
        ehr_evaluator = EHRBenchmarkEvaluator(ehr_config)
        ehr_evaluator.evaluate_all()


if __name__ == "__main__":
    main()
