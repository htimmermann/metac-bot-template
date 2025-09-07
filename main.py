import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
import os
import ast
import re

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
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


# Persona definitions: a short `name` and a detailed `header_template`.
# `header_template` is formatted at runtime with `{output_limit}`.
# Fallback static personas in case LLM generation fails irrecoverably.
FALLBACK_PERSONA_HEADERS: list[str] = [
    (
        "You an expert leader-travel forecaster specialized in high-profile bilateral visits under legal, security, and political constraints. "
        "Consider: bilateral relations, election calendars, legal exposure/extradition risk, sanctions/visas, summit piggybacking, security assessments, public schedules/NOTAMs, prior travel patterns, logistics windows, media trial balloons, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
    (
        "You an expert summit-attendance forecaster specialized in leader participation at G7/NATO/UN events from diplomatic and logistical signals. "
        "Consider: official agendas, domestic constraints, security/legal exposure, aircraft routing, proxy/ministerial substitutes, sanctions/visa issues, bilateral side-meetings, past attendance, health factors, last-minute cancellations, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
    (
        "You an expert AI-benchmark forecaster specialized in leaderboard rank dynamics (e.g., Chatbot Arena) and #1 tenure. "
        "Consider: new model launches, evaluation settings, voting volume/brigading, rate limits, dataset familiarity, inference latency/cost, community sentiment, scoring rule changes, prior tenure, meta-updates, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
]


class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    """

    _max_concurrent_questions = (
        1 
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, output_limit: int | None = None, prompt_variants: list[dict[str, str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Limit for the persona reasoning instruction; can be overridden via env var or arg
        self.output_limit: int = (
            output_limit if output_limit is not None else int(os.getenv("FORECAST_OUTPUT_LIMIT", "250"))
        )
        # Dynamic personas will be generated per-question using an LLM.
        # Internal caches per-question
        self._personas_by_question: dict[str, list[tuple[str, str]]] = {}
        self._persona_cursor_by_question: dict[str, int] = {}

    def _get_question_key(self, question: MetaculusQuestion) -> str:
        return getattr(question, "page_url", None) or getattr(question, "question_text", None) or str(id(question))

    async def _ensure_personas_for_question(self, question: MetaculusQuestion) -> None:
        key = self._get_question_key(question)
        if key in self._personas_by_question:
            return
        n = getattr(self, "predictions_per_research_report", None)
        if not isinstance(n, int) or n <= 0:
            n = int(os.getenv("FORECAST_ENSEMBLE_SIZE", "3"))

        question_text = question.question_text
        output_limit = self.output_limit
        prompt = (
            f"""
You are generating {n} expert forecaster PERSONAS for the single forecasting QUESTION below.

OBJECTIVE
Return a Python list of {n} strings. Each string must be formatted EXACTLY:
"You an expert <domain> forecaster specialized in <narrow specialization tied to the QUESTION>. Consider: <10 concise, comma-separated considerations>, Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"

INPUT
QUESTION: <<{question_text}>>
OUTPUT_LIMIT: {output_limit}

RULES
1) Infer exactly ONE concise <domain> phrase from the QUESTION and use it verbatim in EVERY item. Do not mix domains.
   Examples of valid domain families (pick the closest one, or coin a concise equivalent): leader-travel; summit-attendance; AI-benchmark; podcast-chart; head-to-head podcast rank; Treasury-financing; US-macro releases nowcaster; state-GDP; spaceflight-operations; election-primaries; Euro-area sentiment; US consumer-sentiment; climate-anomaly; China inflation-cycle; rental-market; box-office; streaming-charts; geopolitical-bloc; diplomatic-recognition; West Africa border-policy; market-design; public-sector-efficiency; legal-outcome; IMF-program; fugitive-recapture; energy-program-policy; hydrology & drought; cross-country approval; candidacy-status; executive-clemency; party-alignment; web-availability; accords/treaty-signing; wealth-index composition; displacement-statistics; fuel-price nowcaster; legislative-throughput & confirmations; mortgage-rate; freight-transport; IPO-pipeline announcements; retail-platform policy; urban-complaints; UAP-reporting; air-service; asset-extremes; higher-ed metrics & compliance; social & media signal; live-events & awards; public-health surveillance; corporate-events.
2) Create {n} DISTINCT experts by varying sub-specialization/background and which considerations they emphasize (e.g., logistics vs legal risk, signals vs baselines, etc.).
3) The "Consider:" list must contain EXACTLY 10 items, comma-separated, short, domain-relevant, no trailing period.
4) Each item MUST start exactly with "You an expert " and include the substrings "forecaster specialized in ", "Consider: ", and the trailing clause exactly as written above.
5) Output ONLY the Python list literal with DOUBLE-QUOTED strings. No prose, no backticks, no numbering, no extra lines before/after.

Now produce the list.
"""
        )
        # Choose model for persona generation: prefer dedicated persona_generator, else default
        persona_llm = self.get_llm("persona_generator", "llm")
        if not persona_llm:
            persona_llm = self.get_llm("default", "llm")
        try:
            raw = await persona_llm.invoke(prompt)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(f"Persona generation failed with error: {e}. Falling back to static personas.")
            raw = None

        headers: list[str] = []
        if isinstance(raw, str):
            # Try strict Python list literal parsing
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    headers = [s.strip() for s in parsed if isinstance(s, str)]
            except Exception:
                pass

            if not headers:
                # Fallback: extract quoted strings
                matches = re.findall(r'"(You an expert .*?)"', raw, flags=re.DOTALL)
                if matches:
                    headers = [m.replace("\n", " ").strip() for m in matches]

        if not headers:
            # Final fallback: use static personas formatted with output_limit
            headers = [h.format(output_limit=output_limit) for h in FALLBACK_PERSONA_HEADERS][:n]

        # Normalize length
        if len(headers) < n:
            headers = headers + headers[: max(0, n - len(headers))]
        if len(headers) > n:
            headers = headers[:n]

        # Derive persona names from domain phrase
        personas: list[tuple[str, str]] = []
        for i, h in enumerate(headers):
            m = re.search(r"You an expert\s+(.*?)\s+forecaster\s+specialized\s+in", h, flags=re.IGNORECASE)
            domain = m.group(1) if m else "Forecaster"
            domain_slug = re.sub(r"[^A-Za-z0-9]+", "-", domain).strip("-")
            name = f"{domain_slug or 'Persona'}-{i+1}"
            personas.append((name, h))

        self._personas_by_question[key] = personas
        self._persona_cursor_by_question[key] = 0

    async def _get_persona(self, question: MetaculusQuestion) -> tuple[str, str]:
        key = self._get_question_key(question)
        await self._ensure_personas_for_question(question)
        personas = self._personas_by_question[key]
        idx = self._persona_cursor_by_question.get(key, 0) % max(1, len(personas))
        self._persona_cursor_by_question[key] = (idx + 1) % max(1, len(personas))
        return personas[idx]

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
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
            elif researcher == "perplexity" or (
                isinstance(researcher, str) and researcher.startswith("perplexity/")
            ):
                # Use OpenRouter to call Perplexity models, mirroring the provided snippet
                try:
                    from openai import OpenAI
                except Exception as e:
                    raise RuntimeError(
                        "openai package is required for Perplexity research via OpenRouter"
                    ) from e

                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

                client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

                def get_forecast(model_name: str, message: str) -> str:
                    completion = client.chat.completions.create(
                        extra_body={},
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": message,
                            }
                        ],
                    )
                    agent_response = completion.choices[0].message.content
                    print(
                        "Model Used: "
                        + model_name
                        + " \n Reasoning: "
                        + agent_response
                        + "\n \n \n"
                    )
                    return agent_response

                question_text = question.question_text
                px_prompt = f"""
                You are a research assistant. Find all materially relevant news for the QUESTION below, not just recent items—include seminal or high-impact older coverage if it adds context.

                REQUIREMENTS
                - Search broadly (multi-hop) and expand key terms, synonyms, and entities.
                - Cover the full timeline: earliest notable item → most recent updates.
                - De-duplicate near-duplicates; keep the best source per event.
                - Prefer reputable outlets; avoid paywalled summaries without original reporting.
                - Normalize dates to ISO 8601 (YYYY-MM-DD).
                - OUTPUT ONLY the following fields in JSON Lines (one JSON object per line), nothing else:
                  {{"headline": "...", "brief_content": "...", "date": "YYYY-MM-DD"}}

                QUESTION: {question_text}
                """

                model_name = (
                    researcher if isinstance(researcher, str) and researcher.startswith("perplexity/") else "perplexity/sonar-pro"
                )
                research = get_forecast(model_name=model_name, message=px_prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        persona_name, header = await self._get_persona(question)
        logger.info(f"Persona (binary) for {question.page_url}: {persona_name} | {header}")
        prompt = clean_indents(
            f"""
            {header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

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

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        persona_name, header = await self._get_persona(question)
        logger.info(f"Persona (multiple choice) for {question.page_url}: {persona_name} | {header}")
        prompt = clean_indents(
            f"""
            {header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

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
        persona_name, header = await self._get_persona(question)
        logger.info(f"Persona (numeric) for {question.page_url}: {persona_name} | {header}")
        prompt = clean_indents(
            f"""
            {header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

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
        predictions_per_research_report=int(os.getenv("FORECAST_ENSEMBLE_SIZE", "3")),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",  # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "perplexity",
            "parser": "openrouter/openai/gpt-4o-mini",
            # Use GPT-5 on OpenRouter to generate personas (override via env or code)
            "persona_generator": "openrouter/openai/gpt-5",
        },
        # Optional: override via env FORECAST_OUTPUT_LIMIT
        output_limit=int(os.getenv("FORECAST_OUTPUT_LIMIT", "250")),
        prompt_variants=None,
    )

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
    template_bot.log_report_summary(forecast_reports)
