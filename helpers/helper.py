_BOTH_SAME_SYSTEM = """You are a strict semantic comparator for QA.
Given a QUESTION and two candidate answers (A and B), decide:
1) Do A and B convey the *same substantive answer*? Judge by meaning, not wording.
   - Consider paraphrases, different length, formatting, extra justification, partial overlap that still yields the same final conclusion.
   - Treat numerical equivalence (e.g., 3.14 vs 3.140) as the same.
   - If one is a specific value and the other is a range that clearly includes that value, they are NOT the same unless the question expects a range.
2) Are BOTH answers abstentions/unknowns ("I don't know", "not in knowledge base", refuses to answer, similar)?

Return STRICT JSON only:
{"same": true|false, "both_idk": true|false}
"""

_IDK_SYSTEM = """You are an abstention detector.
Given one ANSWER string, set idk=true iff the answer abstains or expresses unknown, e.g.:
"I don't know", "not in knowledge base", "cannot answer", "insufficient context", "unknown", refusals to answer, or empty.
Allow variants/typos/casing. If the answer gives a guess or hedged but substantive content, idk=false.

Return STRICT JSON only:
{"idk": true|false}
"""


def _parse_bool(dct, key, default=False):
    try:
        return bool(dct.get(key))
    except Exception:
        return default
    
def both_same(question: str, a_text: str, b_text: str, model: str = "gpt-4o") -> tuple[bool, bool]:
    """
    Returns (same, both_idk)
    same=True if A and B convey the same substantive answer (semantic).
    both_idk=True if BOTH are abstentions/unknowns.
    """
    user = f"QUESTION:\n{question}\n\nANSWER A:\n{a_text}\n\nANSWER B:\n{b_text}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _BOTH_SAME_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    return _parse_bool(out, "same"), _parse_bool(out, "both_idk")

def is_idk(answer_text: str, model: str = "gpt-4o") -> bool:
    """
    Returns True if the answer is an abstention/unknown according to the LLM classifier.
    """
    user = f"ANSWER:\n{answer_text}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _IDK_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    return _parse_bool(out, "idk")

_PICK_BEST_SYSTEM = """You are a judge. Given a QUESTION and two different substantive answers (KG vs TRUSTRAG),
pick the one that is more likely **correct** from your own general knowledge and typical references.
Ignore style, pick based on factual plausibility and alignment with the question.

Return STRICT JSON only:
{"winner":"kg"|"trustrag"}
"""

def ask_self_llm(question: str, kg_answer_text: str, trustrag_answer_text: str, model: str = "gpt-4o") -> str:
    """
    When both answers are substantive but different, ask LLM to pick which is more likely correct.
    Returns the chosen answer text (not just the tag).
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"KG_ANSWER:\n{kg_answer_text}\n\n"
        f"TRUSTRAG_ANSWER:\n{trustrag_answer_text}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _Pick_Best_SYSTEM if ( _Pick_Best_SYSTEM := _PICK_BEST_SYSTEM ) else _PICK_BEST_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    winner = (out.get("winner") or "kg").lower()
    return trustrag_answer_text if winner == "trustrag" else kg_answer_text

# --- Your evaluation function (fixed & LLM-only gating) ---

def adjudicate_label_llm_new(
    kg_answer_text: str,
    rag_answer_text: str,  # TrustRAG answer
    question: str,
    model: str = "gpt-4o"
) -> tuple[str, str]:
    """
    Returns (final_answer, validation_flag)
    Validation rules:
      - Same (semantic) -> "Validated"
      - Both IDK -> "Validated" with "I don't know"
      - Fallback to the non-IDK one -> "Non-Validated"
      - Both substantive but different -> LLM picks one -> "Non-Validated"
    """
    try:
        same, both_idk_flag = both_same(question, kg_answer_text, rag_answer_text, model=model)

        if both_idk_flag:
            print("!!!!!both")
            return "I don't know", "Validated"

        if same:
            print("!!!!!same")
            # Either one; they mean the same
            # Prefer the TrustRAG text if you like, but it's arbitrary
            return rag_answer_text if rag_answer_text else kg_answer_text, "Validated"

        # Not the same -> check abstentions via LLM
        kg_idk = is_idk(kg_answer_text, model=model)
        rt_idk = is_idk(rag_answer_text, model=model)

        if kg_idk and not rt_idk:
            print("@@@@")
            return rag_answer_text, "Non-Validated"
        if rt_idk and not kg_idk:
            print("#####")
            return kg_answer_text, "Non-Validated"
        if kg_idk and rt_idk:
            print("&&&&&&&")
            # Redundant (both_idk would have caught), but guard anyway
            return "I don't know", "Validated"
        if not kg_idk and not rt_idk:
            print("(((((((())))))))")
            return ask_self_llm(question,kg_answer_text, rag_answer_text, model= model), "Non-Validated"

    except Exception as e:
        print("adjudicator failed, defaulting to 'I don't know' validated:", e)
    return "I don't know", "Validated"