import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
import os
import re
import json
from asknews_sdk import AskNewsSDK

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    MonetaryCostManager,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):
    """
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects
    """

    _max_concurrent_questions = (
        1
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                Your task is to compile a high‑signal news brief that maximizes accuracy and relevance for downstream forecasting models.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}

                Produce a summary of all relevant news articles for this question. For each item, include:
                - Headline
                - Source/Outlet
                - Publication date (YYYY-MM-DD)
                - URL
                - A 1–2 sentence summary of the key facts

                Curation guidelines:
                - Prioritize very recent items (last 30–90 days), but also include key background pieces that materially affect the answer, including important events that occurred after common LLM knowledge cutoffs.
                - Deduplicate near-identical reports; use credible sources; prefer primary reporting when possible.
                - Organize in reverse chronological order under two sections: "Recent" and "Background".
                - Tie summaries to the resolution criteria when relevant (e.g., thresholds, dates, definitions).
                - Do not produce forecasts; only report evidence. However, state explicitly if, based on current evidence and the criteria, the question would resolve Yes or No today, and why (2–3 sentences).
                - Keep the brief concise, factual, and citation-rich.
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                # Use AskNewsSDK streaming deep_news and collect chunks into a single string
                client_id = os.getenv("ASKNEWS_CLIENT_ID")
                client_secret = os.getenv("ASKNEWS_SECRET")
                if not client_id or not client_secret:
                    raise RuntimeError(
                        "ASKNEWS_CLIENT_ID/ASKNEWS_SECRET not set. Add them to your environment/.env."
                    )

                sdk = AskNewsSDK(
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=["chat", "news", "stories"],
                )

                try:
                    stream = sdk.chat.get_deep_news(
                        messages=[{"role": "user", "content": prompt}],
                        search_depth=2,
                        max_depth=2,
                        model="deepseek-basic",
                        return_sources=False,
                        filter_params=None,
                        stream=True,
                    )

                    chunks: list[str] = []
                    for message in stream:
                        try:
                            chunk = message.choices[0].delta.content
                        except Exception:
                            chunk = None
                        if chunk:
                            chunks.append(chunk)
                    research = "".join(chunks)
                except Exception as e:
                    # Handle AskNews usage/plan limits gracefully by falling back to summarizer LLM
                    es = str(e)
                    if (
                        "403012" in es  # usage limit exceeded for chat_tokens
                        or "403013" in es  # plan lacks access to 'news'
                        or "does not have access to 'news'" in es
                        or "ForbiddenError" in es
                    ):
                        logger.warning(
                            "AskNews error (%s). Falling back to summarizer LLM for research.",
                            es,
                        )
                        research = await self.get_llm("summarizer", "llm").invoke(prompt)
                    else:
                        raise
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=2,
                    max_depth=4,
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=4,
                    max_depth=6,
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Optional multi-forecaster ensemble path (opt-in via env var)
        try:
            ensemble_n = int(os.getenv("FORECAST_ENSEMBLE_SIZE", "0"))
        except ValueError:
            ensemble_n = 0
        if ensemble_n and ensemble_n >= 3:
            return await self._run_forecast_on_binary_with_ensemble(
                question, research, ensemble_n
            )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    @staticmethod
    def _last_number(text: str) -> float | None:
        m = re.search(r"(-?\d+(?:\.\d+)?)(?!.*\d)", text, flags=re.S)
        return float(m.group(1)) if m else None

    @staticmethod
    def _expert_to_forecaster(text: str) -> str | None:
        m = re.search(r"(?is)\b(expert\b.*?\bforecaster)\b", text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _str_to_int_list(string: str) -> list[int]:
        return [int(x) for x in re.findall(r"-?\d+", string)]

    @staticmethod
    def _compute_weights_from_ranks(ranks: list[int]) -> list[float]:
        """
        Implements weighting from the user's example:
        - Compute rank_pct = (rank-1)/(n-1)
        - Drop bottom third (keep rank_pct < 2/3)
        - Map kept items to RawWeight via buckets on rank_pct: <2/15->5, <4/15->4, <6/15->3, <8/15->2, else 1
        - Normalize weights to sum to 1 across kept items; set dropped items' weight to 0
        """
        n = len(ranks)
        if n <= 1:
            return [1.0]
        # Compute rank_pct
        rank_pcts = [((r - 1.0) / (n - 1.0)) for r in ranks]
        kept = [rp < (2.0 / 3.0) for rp in rank_pcts]

        raw_weights: list[float] = []
        for keep, rp in zip(kept, rank_pcts):
            if not keep:
                raw_weights.append(0.0)
                continue
            if rp < (2.0 / 15.0):
                raw_weights.append(5.0)
            elif rp < (4.0 / 15.0):
                raw_weights.append(4.0)
            elif rp < (6.0 / 15.0):
                raw_weights.append(3.0)
            elif rp < (8.0 / 15.0):
                raw_weights.append(2.0)
            else:
                raw_weights.append(1.0)

        total = sum(raw_weights)
        if total <= 0:
            # Fallback: uniform weighting if everything was dropped
            return [1.0 / n for _ in ranks]
        return [rw / total for rw in raw_weights]

    async def _run_forecast_on_binary_with_ensemble(
        self, question: BinaryQuestion, research: str, n_personas: int
    ) -> ReasonedPrediction[float]:
        """
        Generate multiple expert forecaster personas, have each produce a forecast,
        then LLM-rank personas by relevance and compute a weighted ensemble.
        Controlled via FORECAST_ENSEMBLE_SIZE env var.
        """
        output_limit = 200
        qtext = question.question_text
        today = datetime.now().strftime("%Y-%m-%d")

        # 1) Generate personas
        persona_prompt = clean_indents(
            f"""
            You are generating {n_personas} expert forecaster PERSONAS for the single forecasting QUESTION below.

            OBJECTIVE
            Return a Python list of {n_personas} strings. Each string must be formatted EXACTLY:
            "You an expert <domain> forecaster specialized in <narrow specialization tied to the QUESTION>. Consider: <10 concise, comma-separated considerations>, Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"

            INPUT
            QUESTION: <<{qtext}>>
            OUTPUT_LIMIT: {output_limit}

            RULES
            1) Infer exactly ONE concise <domain> phrase from the QUESTION and use it verbatim in EVERY item. Do not mix domains.
            2) Create {n_personas} DISTINCT experts by varying sub-specialization/background and which considerations they emphasize.
            3) The "Consider:" list must contain EXACTLY 10 items, comma-separated, short, domain-relevant, no trailing period.
            4) Each item MUST start exactly with "You an expert " and include the substrings "forecaster specialized in ", "Consider: ", and the trailing clause exactly as written above.
            5) Output ONLY the Python list literal with DOUBLE-QUOTED strings. No prose, no backticks, no numbering, no extra lines before/after.

            Now produce the list.
            """
        )
        persona_llm = self.get_llm("persona_generator", "llm") or self.get_llm(
            "default", "llm"
        )
        raw_personas = await persona_llm.invoke(persona_prompt)
        personas: list[str]
        try:
            personas = json.loads(raw_personas)
        except Exception:
            # Try to extract a JSON list literal substring
            m = re.search(r"\[[\s\S]*\]", raw_personas)
            if not m:
                raise RuntimeError("Failed to parse persona list from LLM output")
            personas = json.loads(m.group(0))

        # 2) Each persona produces a probability given research
        async def call_persona(p: str) -> tuple[str, str, float | None]:
            forecaster_short = self._expert_to_forecaster(p) or "expert forecaster"
            prompt = clean_indents(
                f"""
                {p}

                Your question is: {qtext}
                Today is {today}.
                Please consider the following recent research and evidence in your forecast:
                {research}

                Answer ONLY with your detailed reasoning followed by a final line as:
                Probability: ZZ%
                """
            )
            text = await self.get_llm("default", "llm").invoke(prompt)
            prob = self._last_number(text)
            return p, forecaster_short, prob

        results = await asyncio.gather(*[call_persona(p) for p in personas])
        persona_texts: list[str] = []
        forecaster_names: list[str] = []
        probs: list[float] = []
        for full_text, short_name, prob in results:
            persona_texts.append(full_text)
            forecaster_names.append(short_name)
            probs.append(prob if prob is not None else float("nan"))

        # Drop NaNs for weighting calc but keep alignment
        valid_indices = [i for i, v in enumerate(probs) if v == v]
        if not valid_indices:
            # Fallback to single-shot flow if no persona returned a prob
            return await self._run_forecast_on_binary(question, research)

        # 3) Rank forecasters by relevance using LLM
        enum_block = "\n".join(
            f"{i+1}. {forecaster_names[i]}" for i in range(len(forecaster_names))
        )
        k = len(forecaster_names)
        rank_prompt = clean_indents(
            f"""
            Rank the following forecasters by relevance to the question. Use ranks 1..{k}, where 1 = most relevant.
            Question: {qtext}

            Forecasters (in fixed order):
            {enum_block}

            OUTPUT FORMAT (MANDATORY):
            - Output ONLY a Python list of length {k}.
            - The i-th element is the rank (1..{k}) assigned to the i-th forecaster above.
            - Use each rank exactly once (no ties).
            - No text, no quotes, no backticks, no trailing commas.

            Example: [3, 1, 2]  (for k=3)
            """
        )
        rank_llm = self.get_llm("ranker", "llm") or self.get_llm("default", "llm")
        raw_ranks = await rank_llm.invoke(rank_prompt)
        ranks = self._str_to_int_list(raw_ranks)
        if len(ranks) != k:
            # If parser failed, assign uniform ranks by index
            ranks = list(range(1, k + 1))

        # 4) Compute weights per persona
        weights = self._compute_weights_from_ranks(ranks)

        # 5) Weighted probability (using only valid probs)
        # Align weights and probs; if a prob is NaN, set weight to 0 and renormalize
        adj_weights = [w if (probs[i] == probs[i]) else 0.0 for i, w in enumerate(weights)]
        total_w = sum(adj_weights) or 1.0
        adj_weights = [w / total_w for w in adj_weights]
        weighted_pct = sum((probs[i] if probs[i] == probs[i] else 0.0) * adj_weights[i] for i in range(k))

        # Bound and convert to decimal
        final_pct = round(max(0.0, min(100.0, weighted_pct)), 0)
        final_decimal = max(0.01, min(0.99, float(final_pct) / 100.0))

        # Build concise reasoning summary
        top_n_to_show = min(5, k)
        top_indices = sorted(range(k), key=lambda i: ranks[i])[:top_n_to_show]
        preview = "\n".join(
            f"- {forecaster_names[i]} (rank {ranks[i]}, w={adj_weights[i]:.2f}): {probs[i]:.1f}%"
            for i in top_indices
        )
        reasoning = clean_indents(
            f"""
            Ensemble of {k} expert forecasters considered, ranked by relevance, and weighted.
            Top forecasters preview:\n{preview}

            Probability: {int(final_pct)}%
            """
        )
        logger.info(
            f"Ensemble forecast for URL {question.page_url}: {final_decimal} (from {k} personas)"
        )
        return ReasonedPrediction(prediction_value=final_decimal, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/perplexity/sonar",
            "researcher": "asknews/news-summaries",
            "parser": "openrouter/openai/gpt-4o-mini",
        },
    )

    # Track cost/time and print supplemental metadata after the standard summary
    import time
    start_ts = time.perf_counter()
    with MonetaryCostManager() as _cost_manager:
        if run_mode == "tournament":
            seasonal_tournament_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
            minibench_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            )
            forecast_reports = seasonal_tournament_reports + minibench_reports
        elif run_mode == "metaculus_cup":
            # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
            # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                )
            )
        elif run_mode == "test_questions":
            # Example questions are a good way to test the bot's performance on a single question
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/"#,  # Human Extinction - Binary
                #"https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
                #"https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
                #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
            ]
            template_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )
    # Print the default summary
    template_bot.log_report_summary(forecast_reports)

    # Print supplemental metadata with real values (cost/time/LLMs/bot name)
    elapsed_minutes = (time.perf_counter() - start_ts) / 60.0
    def _model_name(val):
        return val.model if isinstance(val, GeneralLlm) else str(val)
    # Some ForecastBot implementations store LLMs as a private attribute
    _llms_map = getattr(template_bot, "_llms", {}) or {}
    llm_entries = [f"{k}: {_model_name(v)}" for k, v in _llms_map.items()] or ["n/a"]
    logger.info("\n---------------- Supplemental Metadata ----------------")
    logger.info(f"Total Cost: {_cost_manager.current_usage}")
    logger.info(f"Time Spent: {elapsed_minutes:.2f} minutes")
    logger.info(f"LLMs: {'; '.join(llm_entries)}")
    logger.info(f"Bot Name: {template_bot.__class__.__name__}")
