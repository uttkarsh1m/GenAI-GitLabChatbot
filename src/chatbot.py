"""
Chatbot Core Module
Handles conversation logic, RAG pipeline, and Gemini API integration.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# System prompt — grounded but answer-permissive
SYSTEM_PROMPT = """You are GitBot, an AI assistant strictly grounded in GitLab's Handbook and Direction pages.

RULES:
1. Answer using ONLY the CONTEXT provided in each message. Do not use outside knowledge.
2. If the context contains relevant information — even partial — USE IT to answer. Do not refuse.
3. If the context does NOT contain the answer, say clearly:
   "The public GitLab Handbook doesn't cover [topic] in detail. Here's what I do know: [related info from context if any]. For more, check [relevant URL from context]."
4. NEVER include (Source: ...), URLs, or any citation text inside your response. Sources are shown separately.
5. Never invent facts. If context is partial, say what you found and note what's missing.
6. Be friendly, concise, and structured (use bullet points for lists).

You represent GitLab's "build in public" philosophy — be transparent about your sources and limitations.
"""

GUARDRAIL_TOPICS = [
    "personal information", "private data", "confidential", "password", "secret",
    "hack", "exploit", "illegal", "competitor secrets", "insider trading"
]

# Keywords that strongly indicate a GitLab-relevant question
GITLAB_RELEVANT_KEYWORDS = [
    # Company & culture
    "gitlab", "handbook", "direction", "values", "credit", "culture", "mission",
    "vision", "history", "all-remote", "remote", "async", "asynchronous",
    # People & HR
    "hire", "hiring", "onboard", "onboarding", "interview", "job", "career",
    "employee", "team member", "manager", "leadership", "performance", "review",
    "compensation", "salary", "benefits", "total rewards", "equity", "pto",
    "vacation", "leave", "diversity", "inclusion", "belonging",
    # Engineering & product
    "engineering", "product", "roadmap", "feature", "release", "deploy",
    "devops", "devsecops", "ci", "cd", "pipeline", "merge request", "issue",
    "milestone", "iteration", "okr", "kpi", "metric", "sprint",
    # Process & communication
    "process", "policy", "procedure", "communication", "meeting", "1:1",
    "feedback", "collaboration", "transparency", "efficiency", "results",
    "strategy", "planning", "okrs", "goals", "objective",
    # Departments
    "sales", "marketing", "finance", "legal", "security", "support",
    "customer", "partner", "alliance", "revenue", "growth",
]

# Topics clearly outside GitLab's scope
OFFTOPIC_PATTERNS = [
    # General knowledge
    ("capital of", "geography question"),
    ("how to cook", "cooking question"),
    ("recipe for", "cooking question"),
    ("weather in", "weather question"),
    ("stock price", "finance/stock question"),
    ("bitcoin", "cryptocurrency question"),
    ("cryptocurrency", "cryptocurrency question"),
    ("sports score", "sports question"),
    ("who won the", "sports/news question"),
    ("movie review", "entertainment question"),
    ("best restaurant", "lifestyle question"),
    ("how to lose weight", "health question"),
    ("celebrity", "entertainment question"),
    ("horoscope", "astrology question"),
    ("lottery", "gambling question"),
    ("write me a poem", "creative writing request"),
    ("write a story", "creative writing request"),
    ("tell me a joke", "entertainment request"),
    ("what is 2+2", "math question"),
    ("solve this equation", "math question"),
    ("translate", "translation request"),
    ("what language", "language question"),
    ("history of rome", "general history question"),
    ("world war", "general history question"),
    ("president of", "politics question"),
    ("prime minister", "politics question"),
    ("election", "politics question"),
]

SUGGESTED_TOPICS = [
    "GitLab's core values (CREDIT)",
    "Remote work culture and practices",
    "Engineering processes and workflows",
    "Product direction and roadmap",
    "Hiring and onboarding process",
    "Total rewards and compensation",
    "Leadership principles",
    "Communication guidelines",
    "Security practices",
    "OKRs and goal setting",
]

# Confidence threshold below which we flag low relevance
LOW_RELEVANCE_THRESHOLD = 35.0


class GitLabChatbot:
    """
    RAG-based chatbot for GitLab Handbook and Direction pages.
    Uses Gemini for generation and FAISS for retrieval.
    """

    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = None
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self._initialized = False

    def initialize(self, index, chunks: List[Dict]) -> bool:
        """Initialize the chatbot with a vector index and chunks."""
        try:
            from google import genai

            if not self.api_key or not self.api_key.strip():
                logger.error("GEMINI_API_KEY not set or empty")
                return False

            # Validate index and chunks
            if index is None or not chunks:
                logger.error("Cannot initialize: index or chunks are missing.")
                return False

            self._genai_client = genai.Client(api_key=self.api_key.strip())
            self._model_name = "gemini-2.5-flash"
            self.index = index
            self.chunks = chunks
            self._initialized = True
            logger.info("GitLabChatbot initialized successfully.")
            return True

        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            return False

    def check_guardrails(self, query: str) -> Tuple[bool, str]:
        """Layer 1: block harmful/sensitive topics and enforce length."""
        if not query or not isinstance(query, str):
            return False, "Please enter a valid question."

        # Strip and check whitespace-only
        stripped = query.strip()
        if not stripped:
            return False, "Please enter a question."
        if len(stripped) < 3:
            return False, "Please ask a more specific question."
        if len(stripped) > 1000:
            return False, "Your question is too long. Please keep it under 1000 characters."

        query_lower = stripped.lower()
        for topic in GUARDRAIL_TOPICS:
            if topic in query_lower:
                return False, (
                    f"I can't help with queries related to '{topic}'. "
                    "Please consult GitLab's security or legal team for sensitive matters."
                )

        return True, ""

    def is_gitlab_relevant(self, query: str) -> Tuple[bool, str]:
        """
        Layer 2 guardrail: detect off-topic questions using keyword scoring.

        Returns:
            (is_relevant, redirect_message)
            - is_relevant=True  → proceed normally
            - is_relevant=False → return a polite redirect
        """
        query_lower = query.lower()

        # Fast pass: if any GitLab keyword is present, it's relevant
        for kw in GITLAB_RELEVANT_KEYWORDS:
            if kw in query_lower:
                return True, ""

        # Check known off-topic patterns
        for pattern, topic_type in OFFTOPIC_PATTERNS:
            if pattern in query_lower:
                suggestions = "\n".join(f"• {t}" for t in SUGGESTED_TOPICS[:5])
                return False, (
                    f"🦊 I'm GitBot, specialized in GitLab's Handbook and Direction pages. "
                    f"That looks like a **{topic_type}** — outside my area of expertise.\n\n"
                    f"Here are some things I *can* help you with:\n{suggestions}\n\n"
                    f"Feel free to ask anything about GitLab's culture, processes, or product strategy!"
                )

        # Catch-all: longer queries with zero GitLab keywords are likely off-topic
        words = [w for w in query_lower.split() if len(w) > 3]
        if len(words) >= 4:
            # Check if any word even loosely relates to work/tech/company topics
            work_adjacent = {
                "company", "team", "work", "job", "role", "people", "org",
                "tool", "software", "platform", "system", "service", "app",
                "data", "code", "develop", "build", "deploy", "test", "review",
                "manage", "lead", "plan", "goal", "metric", "report", "document",
            }
            if not any(w in work_adjacent for w in words):
                suggestions = "\n".join(f"• {t}" for t in SUGGESTED_TOPICS[:5])
                return False, (
                    f"🦊 I'm GitBot — I only answer questions about GitLab's Handbook and Direction pages.\n\n"
                    f"I couldn't find a GitLab connection in your question. Here's what I can help with:\n"
                    f"{suggestions}\n\n"
                    f"Try asking something about GitLab's culture, values, engineering, or product strategy!"
                )

        return True, ""

    def check_relevance_by_confidence(self, avg_confidence: float,
                                       top_confidence: float,
                                       query: str) -> Tuple[bool, str]:
        """
        Layer 3 guardrail: post-retrieval confidence check.
        Uses BOTH avg and top-1 confidence — if either is high enough, allow it.
        A single highly relevant chunk is sufficient to answer the question.
        """
        # If the best single chunk is highly relevant, proceed regardless of avg
        if top_confidence >= 60.0:
            return True, ""
        # If avg is above threshold, proceed
        if avg_confidence >= LOW_RELEVANCE_THRESHOLD:
            return True, ""

        suggestions = "\n".join(f"• {t}" for t in SUGGESTED_TOPICS[:4])
        return False, (
            f"🦊 I couldn't find relevant information about that in GitLab's Handbook or Direction pages.\n\n"
            f"This might be outside my knowledge scope. I'm best at answering questions about:\n"
            f"{suggestions}\n\n"
            f"Could you rephrase, or ask something GitLab-related?"
        )

    def detect_hallucination(self, answer: str, context: str) -> Tuple[bool, str]:
        """
        Post-generation hallucination check.
        Flags answers that contain confident claims with no grounding signal.

        Returns:
            (is_grounded, warning_note)
            - is_grounded=True  → answer looks grounded
            - is_grounded=False → answer may contain hallucinated content
        """
        if not answer or not context:
            return True, ""

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Red flags: phrases that indicate the LLM is drawing on training knowledge
        hallucination_signals = [
            "as of my knowledge",
            "based on my training",
            "i believe",
            "i think",
            "typically",
            "generally speaking",
            "in most companies",
            "it is common",
            "usually",
            "as far as i know",
            "to my knowledge",
            "i'm not sure but",
            "i cannot confirm",
        ]

        found_signals = [s for s in hallucination_signals if s in answer_lower]
        if found_signals:
            return False, (
                f"\n\n---\n⚠️ **Transparency note:** This response may contain information "
                f"not directly sourced from the retrieved GitLab pages "
                f"(detected hedging language: *\"{found_signals[0]}\"*). "
                f"Please verify against the cited sources."
            )

        # Check: does the answer mention GitLab-specific claims that aren't in context?
        # Extract key noun phrases from answer that look like policy/fact claims
        import re
        # Look for sentences with strong assertion verbs not backed by context
        assertion_pattern = re.compile(
            r'gitlab\s+(?:requires?|mandates?|enforces?|guarantees?|promises?|ensures?)\s+\w+',
            re.IGNORECASE
        )
        assertions = assertion_pattern.findall(answer)
        ungrounded = []
        for assertion in assertions:
            # Check if the core claim appears anywhere in context
            key_words = assertion.lower().split()[2:]  # skip "gitlab requires/mandates"
            if key_words and not any(w in context_lower for w in key_words):
                ungrounded.append(assertion)

        if ungrounded:
            return False, (
                f"\n\n---\n⚠️ **Transparency note:** Some claims in this response "
                f"could not be directly verified in the retrieved context. "
                f"Please cross-check with the cited source URLs."
            )

        return True, ""

    def stream_chat(self, query: str, history: List[Dict] = None, show_sources: bool = True):
        """
        Streaming version of chat() — yields text tokens as they arrive.

        Usage in Streamlit:
            with st.empty():
                full_text = st.write_stream(bot.stream_chat(query))

        Yields:
            str tokens from the model as they stream in.
            On completion, stores result metadata in self._last_stream_meta
            so the caller can retrieve sources, confidence etc.
        """
        # Reset metadata
        self._last_stream_meta = {
            "sources": [], "confidence": 0,
            "category": None, "is_grounded": True,
            "verification_issues": [], "error": None
        }

        if not self._initialized:
            yield "Chatbot not initialized. Please check your API key."
            self._last_stream_meta["error"] = "Not initialized"
            return

        # Layer 1: guardrails
        is_safe, warning = self.check_guardrails(query)
        if not is_safe:
            yield f"⚠️ {warning}"
            self._last_stream_meta["error"] = "guardrail"
            return

        # Acknowledgment shortcut — no LLM needed
        if self.is_acknowledgment(query) and history:
            yield "Glad that helped! Feel free to ask another question about GitLab's handbook or direction. 🦊"
            return

        # Follow-up detection
        is_followup_query = self.is_followup(query, history=history)
        effective_query = self.expand_followup_query(query, history=history) if is_followup_query else query

        # Layer 2: topic relevance (skip for follow-ups)
        if not is_followup_query:
            is_relevant, redirect_msg = self.is_gitlab_relevant(effective_query)
            if not is_relevant:
                yield redirect_msg
                self._last_stream_meta["error"] = "off_topic"
                return

        # ── Vague query check — runs on ORIGINAL query, before any expansion ──
        # Must be after is_followup check (follow-ups are never vague)
        # Must be before retrieve_context (which expands with history)
        if not is_followup_query and self.is_vague_query(query):
            yield self.SAFE_DEFAULT_ANSWER
            self._last_stream_meta["error"] = "vague_query"
            return

        try:
            # Retrieve + rerank
            relevant_chunks = self.retrieve_context(effective_query, top_k=5, history=history)

            avg_confidence = 0.0
            top_confidence = 0.0
            if relevant_chunks:
                confidences = [c.get('confidence', 0) for c in relevant_chunks
                               if isinstance(c.get('confidence'), (int, float))]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                top_confidence = max(confidences) if confidences else 0.0

            # ── Post-retrieval coherence check ────────────────────────────
            # Low score spread = retriever pulled random chunks = vague query
            # Bypass this check if top_confidence is very high (meaning we found exact relevant chunks)
            coherence = self.check_chunk_coherence(relevant_chunks)
            if coherence < 0.15 and top_confidence < 40.0 and not is_followup_query and len(query.split()) <= 4:
                yield self.SAFE_DEFAULT_ANSWER
                self._last_stream_meta["error"] = "vague_query"
                return

            # Layer 3: confidence check
            if not is_followup_query:
                is_confident, low_conf_msg = self.check_relevance_by_confidence(
                    avg_confidence, top_confidence, query)
                if not is_confident:
                    yield low_conf_msg
                    self._last_stream_meta["error"] = "low_confidence"
                    self._last_stream_meta["confidence"] = round(avg_confidence, 1)
                    return
            else:
                if avg_confidence < 15.0 and top_confidence < 40.0:
                    yield ("🦊 I still couldn't find more specific information on this in the "
                           "retrieved GitLab pages. Try checking the source links from the previous response.")
                    self._last_stream_meta["error"] = "low_confidence_followup"
                    return

            context = self.format_context(relevant_chunks)
            prompt = self.build_prompt(query, effective_query, context, history=history, show_sources=show_sources)

            from google.genai import types as gtypes
            MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.0-flash",
                             "gemini-2.0-flash-lite", "gemini-2.5-flash-lite"]
            token_limit = 2048 if is_followup_query else 1024

            full_answer = ""
            last_error = None

            for model_name in MODELS_TO_TRY:
                try:
                    stream = self._genai_client.models.generate_content_stream(
                        model=model_name,
                        contents=prompt,
                        config=gtypes.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.3,
                            top_p=0.8,
                            max_output_tokens=token_limit,
                        )
                    )
                    for chunk in stream:
                        token = getattr(chunk, 'text', None)
                        if token:
                            full_answer += token
                            yield token
                    self._model_name = model_name
                    break

                except Exception as e:
                    err_str = str(e)
                    last_error = e
                    if any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED")):
                        logger.warning(f"Model {model_name} quota exceeded, trying next...")
                        continue
                    if "PERMISSION_DENIED" in err_str.upper() and "quota" not in err_str.lower():
                        raise e
                    logger.warning(f"Model {model_name} failed: {err_str[:100]}")
                    continue

            if not full_answer:
                raise last_error or RuntimeError("All models returned empty responses.")

            # Post-generation checks on the full assembled answer
            is_grounded, warning_note = self.detect_hallucination(full_answer, context)
            if not is_grounded:
                yield warning_note

            full_answer_final, verification_issues = self.verify_answer(
                full_answer + (warning_note if not is_grounded else ""), query, context
            )

            # Build sources
            sources = []
            seen_urls: set = set()
            for ch in relevant_chunks:
                url = ch.get('source_url') or ''
                if url and url not in seen_urls:
                    sources.append({
                        "title": ch.get('title') or 'GitLab',
                        "url": url,
                        "section": ch.get('section') or '',
                        "category": ch.get('category') or '',
                        "confidence": round(float(ch.get('confidence', 0)), 1)
                    })
                    seen_urls.add(url)

            # History is maintained by the UI.

            # Store metadata for caller to retrieve
            self._last_stream_meta = {
                "sources": sources[:3],
                "confidence": round(avg_confidence, 1),
                "category": self.detect_query_category(query),
                "is_grounded": is_grounded,
                "verification_issues": verification_issues,
                "error": None
            }

        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            error_msg = str(e)
            if any(k in error_msg for k in ("429", "RESOURCE_EXHAUSTED")):
                yield ("\n\n⏳ **API quota exceeded.**\n\n"
                       "Wait 1 minute and retry, or get a fresh API key at "
                       "[aistudio.google.com](https://aistudio.google.com/app/apikey).")
            elif any(k in error_msg.upper() for k in ("API_KEY", "UNAUTHENTICATED")):
                yield "❌ Invalid API key. Please check your Gemini API key in the sidebar."
            else:
                yield "❌ An error occurred. Please try again in a moment."
            self._last_stream_meta["error"] = error_msg

    def detect_query_category(self, query: str) -> Optional[str]:
        """Detect if query is more about handbook or direction."""
        direction_keywords = ["roadmap", "future", "plan", "vision", "strategy", "direction",
                               "upcoming", "next year", "product", "feature", "release"]
        handbook_keywords = ["policy", "process", "how to", "culture", "values", "team",
                              "hire", "onboard", "compensation", "benefit", "remote", "work"]
        query_lower = query.lower()
        direction_score = sum(1 for kw in direction_keywords if kw in query_lower)
        handbook_score = sum(1 for kw in handbook_keywords if kw in query_lower)
        if direction_score > handbook_score:
            return "direction"
        elif handbook_score > direction_score:
            return "handbook"
        return None

    # Short replies that only make sense in context of the previous message
    # Pure acknowledgments — user is just reacting, not asking for more
    ACKNOWLEDGMENT_TRIGGERS = {
        "okay great", "ok great", "great", "nice", "cool", "awesome",
        "perfect", "sounds good", "got it", "understood", "thanks",
        "thank you", "ok thanks", "okay thanks", "noted", "alright",
        "okay", "ok", "good", "nice one", "well done", "excellent",
        # Common typos
        "okay greate", "ok greate", "greate", "thnks", "thankyou",
        "thx", "ty", "np", "no problem", "sure thing", "cool i understood",
    }

    def is_acknowledgment(self, query: str) -> bool:
        """
        Detect if a short query is simply acknowledging a previous response.
        Handles exact matches and core acknowledgment words in short phrases.
        """
        q = query.strip().lower().rstrip("!?.")
        
        # Fast exact match
        if q in self.ACKNOWLEDGMENT_TRIGGERS:
            return True
            
        words = q.split()
        if len(words) > 5:
            return False
            
        # Core acknowledgment words
        core_ack_words = {"great", "cool", "understood", "awesome", "perfect", 
                          "thanks", "noted", "alright", "okay", "good", "excellent",
                          "understand", "gotcha", "makes sense"}
        
        # If phrase has an ack word but NO question words, treat as acknowledgment
        question_words = {"what", "why", "how", "where", "when", "who", "which"}
        word_set = set(words)
        
        if word_set.intersection(core_ack_words) and not word_set.intersection(question_words):
            return True
            
        # Typo fallback: if it ends in "work" and is short (e.g. "reat work", "gud work")
        if len(words) <= 3 and words[-1] == "work":
            return True
            
        return False

    # Exact short replies that are clearly follow-ups requesting more info
    FOLLOWUP_TRIGGERS = {
        "yes", "yes i am", "yes please", "yeah", "yep", "sure", "ok", "okay",
        "go ahead", "tell me more", "more", "continue", "and", "what else",
        "elaborate", "elaborate more", "elaborate it", "elaborate on that",
        "explain more", "explain it", "explain that", "explain each",
        "explain all", "explain them", "details", "please",
        "go on", "next", "sounds good", "great", "interesting", "cool",
        "nice", "good", "i am", "i do", "i will", "i would", "i'd like that",
        "definitely", "absolutely", "of course", "why not", "do it", "show me",
        "expand", "expand on that", "expand on it", "more details", "tell me",
        "keep going", "and then", "how so", "really", "what about that",
        "give me more", "give more", "more info", "more information",
        "can you elaborate", "can you explain", "please elaborate",
        "please explain", "tell me more about that", "what does that mean",
        "how does that work", "why is that", "what about it",
        "give me a list", "list them", "list it", "provide a list",
        "provide a specific list", "show the list", "what are they",
        "name them", "enumerate", "what specifically",
        # Short imperative follow-ups
        "explain each", "explain all", "explain them", "describe each",
        "describe them", "describe all", "tell me each", "list each",
        "break it down", "break them down", "one by one", "in detail",
        "with examples", "give examples", "show examples",
    }

    # Patterns that indicate a follow-up even with extra words
    FOLLOWUP_PATTERNS = [
        r'^elaborate',
        r'^explain\s+(it|that|more|this|each|all|them|every)',
        r'^expand\s+(on|it|that|this)',
        r'^describe\s+(each|all|them|every|it|that)',
        r'^tell me more',
        r'^more (about|on|details)',
        r'^can you (elaborate|explain|expand|describe)',
        r'^please (elaborate|explain|expand|continue|describe)',
        r'^what (does|do|is|are) that',
        r'^how (does|do|is|are) that',
        r'^why (is|are|does|do) that',
        r'^give me (more|details|examples)',
        r'^(provide|give|show)\s+(a|the|me)?\s*(specific\s+)?(list|details|examples)',
        r'^(list|name|enumerate|describe)\s+(them|it|the|each|all)',
        r'^what (specifically|exactly)',
        r'^break (it|them|this|that) down',
        r'^one by one',
        r'^in (detail|depth)',
        r'^(yes|yeah|yep|sure|ok|okay|go ahead|continue|keep going)',
        r'^(sounds good|great|interesting|cool|nice|good|perfect)',
        r'^(absolutely|definitely|of course|why not)',
    ]

    def is_followup(self, query: str, history: List[Dict] = None) -> bool:
        """
        Detect if the query DEPENDS ON previous context to make sense.
        Key principle: having history ≠ being a follow-up.
        A follow-up must contain explicit context-referencing markers.
        """
        import re
        history = history or []

        # No history = can't be a follow-up
        if not history:
            return False

        q = query.strip().lower().rstrip("!?.")

        # Explicit context-referencing words — these REQUIRE prior context
        CONTEXT_MARKERS = [
            "it", "this", "that", "they", "those", "them", "these",
            "the above", "the previous", "the last", "aforementioned",
        ]

        # Explicit continuation phrases
        CONTINUATION_PHRASES = {
            "elaborate", "elaborate more", "elaborate it", "elaborate on that",
            "elaborate on this", "explain more", "explain it", "explain that",
            "explain each", "explain all", "explain them", "describe each",
            "describe them", "describe all", "tell me more", "tell me more about that",
            "more details", "more info", "more information", "give me more",
            "keep going", "go on", "continue", "go ahead", "next",
            "expand on that", "expand on this", "expand on it",
            "break it down", "break them down", "one by one", "in detail",
            "with examples", "give examples", "yes", "yes please", "yeah",
            "yep", "sure", "ok", "okay", "sounds good", "great", "interesting",
            "cool", "nice", "good", "definitely", "absolutely", "of course",
            "why not", "why", "what", "how", "how so", "who", "where", "when", "which",
            "do it", "show me", "please", "provide a list",
            "list them", "name them", "what are they", "what specifically",
        }

        # Exact match against continuation phrases
        if q in CONTINUATION_PHRASES:
            return True

        # Contains explicit context marker (e.g. "explain it", "the last one")
        # Use regex to support multi-word markers like "the last" safely
        marker_pattern = r'\b(?:' + '|'.join(CONTEXT_MARKERS) + r')\b'
        if re.search(marker_pattern, q):
            return True

        # Regex patterns for context-referencing
        FOLLOWUP_PATTERNS = [
            r'^elaborate',
            r'^explain\s+(it|that|more|this|each|all|them|every)',
            r'^expand\s+(on|it|that|this)',
            r'^describe\s+(each|all|them|every|it|that)',
            r'^tell me more',
            r'^more (about|on|details)',
            r'^can you (elaborate|explain|expand|describe)',
            r'^please (elaborate|explain|expand|continue|describe)',
            r'^what (does|do|is|are) that',
            r'^how (does|do|is|are) that',
            r'^why (is|are|does|do) that',
            r'^give me (more|details|examples)',
            r'^(provide|give|show)\s+(a|the|me)?\s*(specific\s+)?(list|details|examples)',
            r'^(list|name|enumerate|describe)\s+(them|it|the|each|all)',
            r'^break (it|them|this|that) down',
            r'^one by one',
            r'^in (detail|depth)',
        ]

        if len(q.split()) <= 12:
            for pattern in FOLLOWUP_PATTERNS:
                if re.match(pattern, q, re.IGNORECASE):
                    return True

        return False

    def expand_followup_query(self, query: str, history: List[Dict] = None) -> str:
        """
        Reconstruct a meaningful search query from a follow-up reply.
        This is used EXCLUSIVELY for BM25/Vector retrieval, so it must be clean
        and devoid of conversational filler (no 'Previous question:' prefixes)
        which pollute embeddings and lexical search.
        """
        history = history or []
        if not history:
            return query

        last_user_q = ""
        for msg in reversed(history):
            if not last_user_q and msg.get('role') == 'user':
                last_user_q = (msg.get('content') or '').strip()
                break

        if last_user_q:
            # Clean combination of the topic context and the new follow-up
            expanded = f"{last_user_q} {query}"
            logger.info(f"Follow-up '{query}' expanded for retrieval: '{expanded}'")
            return expanded
            
        return query

    def verify_answer(self, answer: str, query: str, context: str) -> Tuple[str, List[str]]:
        """
        Answer verification step — checks the generated answer against
        multiple quality dimensions before it reaches the user.

        Checks performed:
        1. Non-empty
        2. Not a raw repetition of the question
        3. Not a model refusal
        4. Addresses the question (key query terms appear in answer)
        5. Cites at least one source
        6. Does not exceed a safe length

        Returns:
            (verified_answer, list_of_issues)
        """
        import re
        issues: List[str] = []

        # ── Check 1: Non-empty ────────────────────────────────────────────────
        if not answer or not answer.strip():
            return (
                "⚠️ GitBot received an empty response from the AI model. "
                "Please try rephrasing your question.",
                ["empty_response"]
            )

        stripped = answer.strip()
        answer_lower = stripped.lower()
        word_count = len(stripped.split())

        # ── Check 2: Minimum useful length — only flag truly trivial responses ─
        # Real answers are usually 40+ words; under 20 is suspicious
        if word_count < 20:
            issues.append("suspiciously_short")
            logger.warning(f"Answer too short ({word_count} words): {stripped[:80]}")

        # ── Check 3: Model refusal / error patterns ───────────────────────────
        refusal_patterns = [
            "i cannot", "i can't", "i am unable", "i'm unable",
            "i do not have access", "i don't have access",
            "as an ai", "as a language model",
            "i cannot provide", "i'm not able to",
            "i apologize, but i cannot",
        ]
        found_refusal = next((p for p in refusal_patterns if p in answer_lower), None)
        if found_refusal:
            issues.append("model_refusal")
            logger.warning(f"Model refusal detected: '{found_refusal}'")
            stripped = (
                stripped + "\n\n---\n"
                "💡 *If you think this should be answerable, try rephrasing "
                "or check the source pages directly.*"
            )

        # ── Check 4: Answer addresses the question ────────────────────────────
        stopwords = {"what", "when", "where", "which", "does", "gitlab", "about",
                     "tell", "explain", "describe", "with", "from", "that", "this",
                     "have", "their", "they", "there", "will", "would", "could",
                     "how", "why", "who", "the", "and", "for", "are", "is"}
        query_words = [
            w.lower().strip("?.,!") for w in query.split()
            if len(w) > 4 and w.lower().strip("?.,!") not in stopwords
        ]
        if query_words and word_count >= 20:
            matched = sum(1 for w in query_words if w in answer_lower)
            coverage = matched / len(query_words)
            if coverage < 0.2:
                issues.append("low_query_coverage")
                logger.warning(
                    f"Answer may not address the question. "
                    f"Coverage: {coverage:.0%} ({matched}/{len(query_words)} terms)"
                )

        # ── Check 5: Source citation present ─────────────────────────────────
        # Only enforce for substantive answers (40+ words)
        if word_count >= 40:
            context_urls = re.findall(r'https?://\S+', context)
            has_citation = (
                any(url in stripped for url in context_urls) or
                "handbook.gitlab.com" in stripped or
                "about.gitlab.com" in stripped or
                "according to" in answer_lower or
                "[source" in answer_lower or
                "source:" in answer_lower or
                re.search(r'\(https?://', stripped) is not None
            )
            if not has_citation:
                issues.append("no_citation")
                stripped = (
                    stripped + "\n\n---\n"
                    "📎 *Note: This response did not include explicit source citations. "
                    "Please verify against the sources shown below.*"
                )

        # ── Check 6: Safe length cap ──────────────────────────────────────────
        # Follow-ups get more room since they build on previous context
        MAX_WORDS = 800
        if word_count > MAX_WORDS:
            # Truncate at sentence boundary, not mid-word
            import re as _re
            sentences = _re.split(r'(?<=[.!?])\s+', stripped)
            truncated = []
            count = 0
            for sent in sentences:
                wc = len(sent.split())
                if count + wc > MAX_WORDS:
                    break
                truncated.append(sent)
                count += wc
            stripped = ' '.join(truncated) + (
                "\n\n*[Response truncated. See source pages for full details.]*"
            )
            issues.append("truncated")
            logger.info(f"Answer truncated from {word_count} to ~{count} words.")

        if issues:
            logger.info(f"Answer verification issues: {issues}")

        return stripped, issues

    # Vague queries that need clarification or a safe default answer
    VAGUE_QUERY_PATTERNS = [
        r'^gitlab\s+(uses?|used|using)$',
        r'^(uses?|used|using)\s+(of\s+)?gitlab$',
        r'^gitlab\s+(is|are|was|were)$',
        r'^what\s+is\s+gitlab$',
        r'^what\s+does\s+gitlab\s+do$',
        r'^tell\s+me\s+about\s+gitlab$',
        r'^gitlab\s*$',
        r'^about\s+gitlab$',
        r'^gitlab\s+(info|information|details?)$',
        r'^(what|how)\s+(is|are|does)\s+gitlab\??$',
        r'^gitlab\s+(overview|summary|intro|introduction)$',
    ]

    SAFE_DEFAULT_ANSWER = (
        "GitLab is an **AI-powered DevSecOps platform** used by organizations to:\n\n"
        "- 🔧 **Version control** — Git-based source code management\n"
        "- ⚡ **CI/CD** — Continuous integration and deployment pipelines\n"
        "- 🔒 **Security** — Built-in SAST, DAST, and vulnerability scanning\n"
        "- 🤝 **Collaboration** — Code review, merge requests, and issue tracking\n"
        "- 🤖 **AI (GitLab Duo)** — AI-assisted coding, code review, and security\n\n"
        "Could you be more specific? For example:\n"
        "- *\"What does GitLab use for AI?\"*\n"
        "- *\"What technologies does GitLab's engineering team use?\"*\n"
        "- *\"How does GitLab use OKRs?\"*"
    )

    def is_vague_query(self, query: str) -> bool:
        """Detect queries that are too vague to retrieve meaningful results."""
        import re
        q = query.strip().lower()

        # If it's a follow-up (contains context markers), it's NOT vague
        # — it has implicit meaning from history
        CONTEXT_MARKERS = {"it", "this", "that", "they", "those", "them", "these"}
        if set(q.split()) & CONTEXT_MARKERS:
            return False

        # Check explicit vague patterns
        for pattern in self.VAGUE_QUERY_PATTERNS:
            if re.match(pattern, q, re.IGNORECASE):
                return True

        # Very short queries with no specific intent
        # Exclude stopwords and context markers to count meaningful words
        stopwords = {"the", "and", "for", "are", "was", "what", "how", "why",
                     "who", "when", "where", "does", "did", "can", "will",
                     "is", "a", "an", "of", "in", "to", "at", "by", "do",
                     "be", "no", "if", "or", "it", "he", "we", "my", "so",
                     "up", "as", "on", "am", "me", "us", "go"}
        meaningful = [w for w in q.split()
                      if w not in stopwords and w not in CONTEXT_MARKERS]
        if len(meaningful) <= 1:
            return True

        return False

    def check_chunk_coherence(self, chunks: list) -> float:
        """
        Measure retrieval quality using score spread.
        Good retrieval: clear top match (high spread between best and worst).
        Bad retrieval: all scores similar (random chunks, no clear winner).
        Returns spread value — low spread = incoherent/vague retrieval.
        """
        if not chunks:
            return 0.0
        scores = [c.get('confidence', 0) for c in chunks]
        if len(scores) < 2:
            return 1.0
        return (max(scores) - min(scores)) / 100.0  # normalise to [0,1]

    def expand_query(self, query: str, history: List[Dict] = None) -> str:
        """
        Expand abbreviated or typo-ridden queries before retrieval.

        Three strategies:
        1. Normalize repeated characters (ccore → core, vvalues → values)
        2. Common GitLab term corrections (typos + abbreviations)
        3. Enrich very short queries with conversation history
        """
        import re

        # Step 0: Normalize typos with repeated characters
        # Rule 1: 3+ same chars anywhere → collapse to 1 (usses→uses, ccore→core)
        normalized = re.sub(r'(.)\1{2,}', r'\1', query.lower())
        # Rule 2: word-start doubled consonant → collapse (ccore→core, vvalues→values)
        normalized = re.sub(r'\b([bcdfghjklmnpqrstvwxyz])\1(?=[a-z])', r'\1', normalized)
        # Rule 3: doubled consonant at word END before 's' or 'es' (typo: usses→uses)
        # Pattern: vowel + doubled_consonant + (s|es|ed) at word boundary
        normalized = re.sub(r'([aeiou])([bcdfghjklmnpqrstvwxyz])\2(s|es|ed)\b', r'\1\2\3', normalized)
        if normalized != query.lower():
            logger.info(f"Typo normalized: '{query}' → '{normalized}'")
            query = normalized

        # Common abbreviations and typos for GitLab-specific terms
        EXPANSIONS = {
            r'\bcor\b': 'core',
            r'\bval\b': 'values',
            r'\bcvals\b': 'core values',
            r'\bcvalues\b': 'core values',
            r'\beng\b': 'engineering',
            r'\bprod\b': 'product',
            r'\bdir\b': 'direction',
            r'\bhand\b': 'handbook',
            r'\bcomm\b': 'communication',
            r'\bcollab\b': 'collaboration',
            r'\btranspar\b': 'transparency',
            r'\biter\b': 'iteration',
            r'\bdiv\b': 'diversity',
            r'\beff\b': 'efficiency',
            r'\bres\b': 'results',
            r'\bhir\b': 'hiring',
            r'\bonboard\b': 'onboarding',
            r'\bokrs?\b': 'OKRs objectives key results',
            r'\bdevsecops\b': 'DevSecOps',
            r'\bci\b': 'continuous integration',
            r'\bcd\b': 'continuous deployment',
        }

        expanded = query.lower()
        changed = False
        for pattern, replacement in EXPANSIONS.items():
            new = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
            if new != expanded:
                expanded = new
                changed = True

        if changed:
            logger.info(f"Query expanded: '{query}' → '{expanded}'")

        # If still very short and we have history, append last topic
        words = expanded.split()
        if len(words) <= 3 and history:
            for msg in reversed(history):
                if msg.get('role') == 'user':
                    last_q = (msg.get('content') or '').strip()
                    if last_q and last_q.lower() != query.lower():
                        expanded = f"{expanded} {last_q}"
                        logger.info(f"Short query enriched with history: '{expanded}'")
                        break

        return expanded

    def retrieve_context(self, query: str, top_k: int = 5, history: List[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search (BM25 + FAISS + reranker).
        Falls back to FAISS-only if hybrid search fails.
        """
        from src.embeddings import hybrid_search, search

        if not self.index or not self.chunks:
            logger.warning("retrieve_context called before index/chunks loaded.")
            return []

        # Expand abbreviated/typo queries before retrieval
        expanded_query = self.expand_query(query, history=history)
        category = self.detect_query_category(expanded_query)

        try:
            results = hybrid_search(
                expanded_query, self.index, self.chunks,
                top_k=top_k,
                category_filter=category
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed ({e}), falling back to FAISS.")
            results = search(expanded_query, self.index, self.chunks,
                             top_k=top_k, category_filter=category)

        # Supplement if sparse
        if len(results) < 3:
            fallback = search(expanded_query, self.index, self.chunks, top_k=top_k)
            seen_ids = {r.get('chunk_id') for r in results}
            for r in fallback:
                if r.get('chunk_id') not in seen_ids:
                    results.append(r)
                if len(results) >= top_k:
                    break

        # Intent boost: if query is clearly about core values, promote CREDIT chunk
        q_lower = expanded_query.lower()
        if 'values' in q_lower and any(w in q_lower for w in ['core', 'credit', 'what are']):
            for chunk in results:
                if chunk.get('section', '').upper() == 'CREDIT' or \
                   'CREDIT' in chunk.get('text', '')[:50]:
                    chunk['confidence'] = min(chunk['confidence'] + 10, 100)
            results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return results

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks for the LLM.
        Uses clean internal labels that the LLM won't copy verbatim into responses.
        The LLM is instructed to cite using (Title, URL) format, not [Source N] labels.
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            if not isinstance(chunk, dict):
                continue
            source = chunk.get('title') or 'GitLab Handbook'
            section = chunk.get('section') or ''
            url = chunk.get('source_url') or ''
            text = chunk.get('text') or ''

            if not text:
                continue

            # Use clean labels — no [Source N - CATEGORY] that LLM might copy
            label = f"--- Passage {i}: {source}"
            if section:
                label += f" / {section}"
            context_parts.append(f"{label}\nSource URL: {url}\n{text}")

        return "\n\n".join(context_parts) if context_parts else "No relevant context found."

    def build_prompt(self, original_query: str, effective_query: str,
                     context: str, history: List[Dict] = None, show_sources: bool = True) -> str:
        """
        Build the fully grounded prompt.
        - original_query: what the user literally typed
        - effective_query: expanded version for follow-ups (may differ)
        - context: retrieved GitLab content
        - show_sources: if False, tell LLM not to include inline citations
        """
        history_parts = []
        if history:
            recent = history[-6:]
            for msg in recent:
                if not isinstance(msg, dict):
                    continue
                role = "User" if msg.get('role') == 'user' else "GitBot"
                content = (msg.get('content') or '').strip()
                # Give the last bot message a much larger length limit so follow-ups on the end of a list work
                max_len = 2000 if (msg == recent[-1] and role == "GitBot") else 400
                history_parts.append(f"{role}: {content[:max_len]}")

        history_block = (
            "=== CONVERSATION HISTORY (for context only) ===\n"
            + "\n".join(history_parts)
            + "\n=== END OF HISTORY ===\n\n"
        ) if history_parts else ""

        followup_note = (
            f"NOTE: The user's message \"{original_query}\" is a short follow-up reply. "
            f"Interpret it in the context of the conversation history above and continue "
            f"on the same topic.\n\n"
        ) if original_query != effective_query else ""

        # Citation instruction — always no inline citations, sources shown separately
        citation_instruction = (
            "- NEVER include (Source: ...) or any URLs inside your response text. "
            "Sources are displayed separately — keep your answer clean and citation-free.\n"
            if show_sources else
            "- Do NOT include any source citations, URLs, or references anywhere in your response.\n"
        )

        return (
            f"=== RETRIEVED CONTEXT FROM GITLAB HANDBOOK & DIRECTION PAGES ===\n\n"
            f"{context}\n\n"
            f"=== END OF CONTEXT ===\n\n"
            f"{history_block}"
            f"{followup_note}"
            f"User Message: {original_query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer using the CONTEXT above. If it contains relevant info, use it — do not refuse.\n"
            f"- Only say you don't know if the context has ZERO relevant content.\n"
            f"{citation_instruction}"
            f"- Do not use knowledge outside the provided context.\n"
            f"- Format clearly with bullet points or headers where helpful."
        )

    def chat(self, query: str, history: List[Dict] = None) -> Dict:
        """
        Process a user query and return a response with metadata.
        
        Returns:
            Dict with keys: response, sources, confidence, category, error
        """
        if not self._initialized:
            return {
                "response": "Chatbot not initialized. Please check your API key and data setup.",
                "sources": [],
                "confidence": 0,
                "category": None,
                "error": "Not initialized"
            }

        # Layer 1: Harmful/sensitive content check
        is_safe, warning = self.check_guardrails(query)
        if not is_safe:
            return {
                "response": f"⚠️ {warning}",
                "sources": [],
                "confidence": 0,
                "category": None,
                "error": "guardrail"
            }

        # Detect pure acknowledgments — brief reply, no LLM call needed
        if self.is_acknowledgment(query) and history:
            return {
                "response": "Glad that helped! Feel free to ask another question about GitLab's handbook or direction. 🦊",
                "sources": [],
                "confidence": 0,
                "category": None,
                "is_grounded": True,
                "verification_issues": [],
                "error": None
            }

        # Detect and expand follow-up replies ("yes", "tell me more", etc.)
        effective_query = query
        is_followup_query = self.is_followup(query, history=history)
        if is_followup_query:
            effective_query = self.expand_followup_query(query, history=history)
            logger.info(f"Follow-up detected: '{query}' expanded with conversation context")

        # Layer 2: Topic relevance check — skip for follow-ups (already in context)
        if not is_followup_query:
            is_relevant, redirect_msg = self.is_gitlab_relevant(effective_query)
            if not is_relevant:
                return {
                    "response": redirect_msg,
                    "sources": [],
                    "confidence": 0,
                    "category": None,
                    "error": "off_topic"
                }

        # ── Vague query check — on ORIGINAL query, before any expansion ───────
        if not is_followup_query and self.is_vague_query(query):
            return {
                "response": self.SAFE_DEFAULT_ANSWER,
                "sources": [], "confidence": 0,
                "category": None, "is_grounded": True,
                "verification_issues": [], "error": "vague_query"
            }

        try:
            # Retrieve using effective_query (context-expanded for follow-ups)
            relevant_chunks = self.retrieve_context(effective_query, top_k=5, history=history)

            # Calculate overall confidence safely
            avg_confidence = 0.0
            top_confidence = 0.0
            if relevant_chunks:
                confidences = [c.get('confidence', 0) for c in relevant_chunks
                               if isinstance(c.get('confidence'), (int, float))]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                top_confidence = max(confidences) if confidences else 0.0

            # ── Post-retrieval coherence check ────────────────────────────
            # Bypass if top_confidence is high (meaning all chunks were equally excellent)
            coherence = self.check_chunk_coherence(relevant_chunks)
            if coherence < 0.15 and top_confidence < 40.0 and not is_followup_query and len(query.split()) <= 4:
                return {
                    "response": self.SAFE_DEFAULT_ANSWER,
                    "sources": [], "confidence": 0,
                    "category": None, "is_grounded": True,
                    "verification_issues": [], "error": "vague_query"
                }

            # Layer 3: Post-retrieval confidence check — skip for follow-ups
            if not is_followup_query:
                is_confident, low_conf_msg = self.check_relevance_by_confidence(
                    avg_confidence, top_confidence, query)
                if not is_confident:
                    return {
                        "response": low_conf_msg,
                        "sources": [],
                        "confidence": round(avg_confidence, 1),
                        "category": None,
                        "error": "low_confidence"
                    }
            else:
                # For follow-ups, only block if confidence is extremely low (near zero)
                if avg_confidence < 15.0:
                    return {
                        "response": (
                            "🦊 I still couldn't find more specific information on this in the "
                            "retrieved GitLab pages. The handbook may not cover this in detail.\n\n"
                            "You can check the source pages directly using the links in the "
                            "previous response, or try asking about a related topic."
                        ),
                        "sources": [],
                        "confidence": round(avg_confidence, 1),
                        "category": None,
                        "error": "low_confidence_followup"
                    }

            context = self.format_context(relevant_chunks)
            prompt = self.build_prompt(query, effective_query, context)

            # Try models in order — import types once outside loop
            from google.genai import types as gtypes
            MODELS_TO_TRY = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-flash-lite",
            ]
            answer = None
            last_error: Optional[Exception] = None

            for model_name in MODELS_TO_TRY:
                try:
                    # Follow-ups need more tokens since they build on previous context
                    token_limit = 2048 if is_followup_query else 1024
                    response = self._genai_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=gtypes.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.3,
                            top_p=0.8,
                            max_output_tokens=token_limit,
                        )
                    )
                    # Safely extract text
                    answer = getattr(response, 'text', None)
                    if answer and answer.strip():
                        self._model_name = model_name
                        break
                    else:
                        logger.warning(f"Model {model_name} returned empty response.")
                        answer = None
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # Check quota first — 429 should try next model, not raise
                    if any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED")):
                        logger.warning(f"Model {model_name} quota exceeded, trying next model...")
                        continue
                    # Only stop retrying on hard auth failures
                    if "PERMISSION_DENIED" in err_str.upper() and "quota" not in err_str.lower():
                        raise e
                    logger.warning(f"Model {model_name} failed: {err_str[:100]}")
                    continue

            if not answer:
                final_error = last_error or RuntimeError("All models returned empty responses.")
                logger.error(f"All models failed. Last error: {str(final_error)[:200]}")
                raise final_error

            # Post-generation hallucination check
            is_grounded, warning_note = self.detect_hallucination(answer, context)
            if not is_grounded:
                answer = answer + warning_note
                logger.warning("Potential hallucination detected in response.")

            # Answer verification — quality gate before returning to user
            answer, verification_issues = self.verify_answer(answer, query, context)
            if verification_issues:
                logger.info(f"Verification issues for query '{query[:50]}': {verification_issues}")

            # Build source citations
            sources = []
            seen_urls: set = set()
            for chunk in relevant_chunks:
                if not isinstance(chunk, dict):
                    continue
                url = chunk.get('source_url') or ''
                if url and url not in seen_urls:
                    sources.append({
                        "title": chunk.get('title') or 'GitLab',
                        "url": url,
                        "section": chunk.get('section') or '',
                        "category": chunk.get('category') or '',
                        "confidence": round(float(chunk.get('confidence', 0)), 1)
                    })
                    seen_urls.add(url)

            return {
                "response": answer,
                "sources": sources[:3],
                "confidence": round(avg_confidence, 1),
                "category": self.detect_query_category(query),
                "is_grounded": is_grounded,
                "verification_issues": verification_issues,
                "error": None
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = str(e)

            # Check quota/rate limit FIRST — 429 errors can contain "PERMISSION_DENIED"
            # so must be matched before the auth check
            if any(k in error_msg for k in ("429", "RESOURCE_EXHAUSTED")):
                user_msg = (
                    "⏳ **API quota exceeded.**\n\n"
                    "**Options:**\n"
                    "1. Wait 1 minute and retry (per-minute limit)\n"
                    "2. Get a fresh API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
                    "3. Enable billing on your Google Cloud project"
                )
            elif any(k in error_msg.upper() for k in ("API_KEY", "UNAUTHENTICATED")) or (
                "PERMISSION_DENIED" in error_msg.upper() and "quota" not in error_msg.lower()
            ):
                user_msg = "❌ Invalid API key. Please check your Gemini API key in the sidebar."
            elif "NETWORK" in error_msg.upper() or "ConnectionError" in error_msg:
                user_msg = "❌ Network error. Please check your internet connection and try again."
            else:
                user_msg = "❌ An error occurred. Please try again in a moment."

            return {
                "response": user_msg,
                "sources": [],
                "confidence": 0,
                "category": None,
                "error": error_msg
            }

    def get_suggested_questions(self) -> List[str]:
        """Return suggested questions for the UI."""
        return [
            "What are GitLab's core values?",
            "How does GitLab approach remote work?",
            "What is GitLab's product direction for AI/ML?",
            "How does the hiring process work at GitLab?",
            "What are GitLab's engineering principles?",
            "How does GitLab handle performance reviews?",
            "What is GitLab's approach to transparency?",
            "How are OKRs used at GitLab?",
        ]
