# "Done when" test for the output checks node. Spec Section 6, Stage 17. Issue #22.
"""Stage 17 output checks: the early stop, the fixed checks in order, the judge, sampling, the
shared repair budget, the three edge values, and the spans. Fake chat models: no key, no network.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import ContextBlock
from flp_rag.graph import build, markers
from flp_rag.graph.nodes import check_output as co
from flp_rag.graph.nodes import generate as gen
from flp_rag.graph.state import initial_state
from flp_rag.models import Completion, Price
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings
from tests.test_generate import FakeChat, fast
from tests.test_graph_build import stub_nodes

ROOT = Path(__file__).resolve().parents[1]

CODE = (
    "class CustomerDTO\n"
    "{\n"
    "    public function __construct(\n"
    "        public string $name,\n"
    "        public string $email,\n"
    "    ) {}\n"
    "}"
)
BLOCKS = [
    ContextBlock(
        "S1",
        ["d:05:0038"],
        "[S1] Chapter 05: Property Promotion, p. 64-65\nPromotion removes boilerplate.",
        5,
        "Property Promotion",
        64,
        65,
        20,
    ),
    ContextBlock(
        "S2",
        ["d:05:0039"],
        "[S2] Chapter 05: Property Promotion > The DTO, p. 64\nHere is the class:\n\n"
        f"```php\n{CODE}\n```\n\nThe rest is left out:\n\n```php\n// …\n```",
        5,
        "Property Promotion",
        64,
        64,
        80,
    ),
    ContextBlock(
        "S3",
        ["d:06:0046"],
        "[S3] Chapter 06: Readonly Properties, p. 77\nReadonly is set once.",
        6,
        "Readonly Properties",
        77,
        77,
        15,
    ),
]
LABEL_MAP = {
    b.label: {
        "chunk_ids": b.source_ids,
        "parent_id": f"p{i}",
        "kind": "chunk",
        "chapter_no": b.chapter_no,
        "chapter_title": b.chapter_title,
        "section_title": b.chapter_title,
        "page_printed_start": b.page_printed_start,
        "page_printed_end": b.page_printed_end,
    }
    for i, b in enumerate(BLOCKS)
}
CLEAN = "Promotion removes boilerplate [S1]. Readonly is set once [S3]."
FAKE_S9 = "Promotion removes boilerplate [S9]."


class FakeJudge(FakeChat):
    """A scripted judge: each output is a verdict dict, a non-JSON string, or an exception."""

    def __init__(self, *outputs: Any) -> None:
        super().__init__(*outputs, model_name="fake:judge")
        self.price_per_million_tokens = Price(input=0.1, output=0.4)

    def complete(
        self, system, messages, json_schema=None, temperature=0, max_tokens=None
    ) -> Completion:  # type: ignore[no-untyped-def]
        self.calls.append((system, list(messages), temperature))
        self.max_tokens, self.json_schema = max_tokens, json_schema
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        text = json.dumps(out) if isinstance(out, dict) else out
        parsed = out if isinstance(out, dict) else None
        return Completion(text, 6400, 100, "end_turn", 5, self.model_name, json=parsed)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return fast(load_settings())


def with_rate(settings: Settings, rate: float) -> Settings:
    return settings.model_copy(
        update={"output": settings.output.model_copy(update={"faithfulness_sample_rate": rate})}
    )


def check(
    answer: str,
    settings: Settings,
    *,
    judge: FakeJudge | None = None,
    borderline: bool = False,
    repair_calls: int = 0,
    request_id: str = "r1",
    judge_all: bool = False,
    context: list[ContextBlock] = BLOCKS,
    label_map: dict[str, Any] = LABEL_MAP,
) -> co.Outcome:
    judge = judge or FakeJudge()
    return co.check_output(
        answer,
        label_map=label_map,
        context=context,
        borderline=borderline,
        request_id=request_id,
        repair_calls=repair_calls,
        settings=settings,
        judge=lambda: judge,
        prompts=co.load_check_prompts(settings),
        judge_all=judge_all,
    )


def spans_named(exporter: InMemorySpanExporter, name: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


# --------------------------------------------------------------------------- marker parser


def test_one_marker_parser_reads_every_accepted_form() -> None:
    assert markers.find_markers("[S1]") == ["S1"]
    assert markers.find_markers("[S1][S3]") == ["S1", "S3"]
    assert markers.find_markers("[S1, S3]") == ["S1", "S3"]
    assert markers.find_markers("[S1,S3] and [S12]") == ["S1", "S3", "S12"]
    assert markers.find_markers("(S1), Source 1, S1 alone, [S], [s1]") == []
    assert markers.find_malformed("[S1-S9] [s9] [S 9] [S9 ] [ S9] [S1; S9] [S1] [see S1]") == [
        "[S1-S9]",
        "[s9]",
        "[S 9]",
        "[S1; S9]",
        "[see S1]",
    ]
    assert markers.find_markers("[ S1 ] and [S2 ] and [ S3, S4 ]") == ["S1", "S2", "S3", "S4"]
    assert markers.find_markers("boilerplate【S1】 and 【S2, S3】") == ["S1", "S2", "S3"]
    assert markers.find_malformed("【S1-S2】 【s1】 【S1】") == ["【S1-S2】", "【s1】"]
    assert markers.find_malformed("[S1][S2, S3] [1] [S] [note]") == []
    # Stage 16 reads markers through the same parser.
    assert gen.classify("Promotion removes boilerplate [S1, S3].") == "answer"
    assert gen.classify("Promotion removes boilerplate (S1).") == "invalid"


# --------------------------------------------------------------------------- early stop


def test_exact_abstain_stops_early_without_any_check(settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    judge = FakeJudge({"supported": False, "unsupported_claims": ["x"]})
    outcome = check(ABSTAIN_TEXT, with_rate(settings, 1.0), judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.early_stop
    assert outcome.answer == ABSTAIN_TEXT and outcome.abstain_reason is None  # Stage 16's stays
    assert judge.calls == [] and outcome.citations is None and outcome.code is None
    assert not [s for s in exporter.get_finished_spans() if s.name.startswith("check.")]


def test_tolerated_abstain_forms_also_stop_early(settings) -> None:
    for text in (f' "{ABSTAIN_TEXT}." ', f"**{ABSTAIN_TEXT}**"):
        outcome = check(text, settings)
        assert outcome.result == "abstain" and outcome.early_stop and outcome.answer == ABSTAIN_TEXT


# --------------------------------------------------------------------------- fixed checks


@pytest.mark.parametrize(
    "answer",
    [
        f"{ABSTAIN_TEXT}. However, [S1] says that promotion removes boilerplate.",
        f"{ABSTAIN_TEXT.upper()}, but promotion removes boilerplate [S1].",
        f"Promotion removes boilerplate [S1]. Otherwise, {ABSTAIN_TEXT.lower()}.",
    ],
)
def test_mixed_abstain_text_in_any_case_asks_for_a_repair(settings, answer) -> None:
    outcome = check(answer, settings)
    assert outcome.result == "repair" and outcome.failure == "mixed_abstain"
    assert outcome.mixed_abstain and outcome.citations is None  # the first check stops the rest
    assert outcome.repair_message and ABSTAIN_TEXT in outcome.repair_message
    assert outcome.repair_message.rstrip().endswith(ABSTAIN_TEXT)  # the rewrite line closes it


def test_unknown_marker_fails_even_next_to_valid_ones(settings) -> None:
    outcome = check("Promotion [S1][S9] removes boilerplate [S2, S3].", settings)
    assert outcome.result == "repair" and outcome.failure == "citations"
    assert outcome.citations is not None
    assert outcome.citations.markers_found == ["S1", "S9", "S2", "S3"]
    assert outcome.citations.markers_unknown == ["S9"] and not outcome.citations.valid
    assert "[S9]" in (outcome.repair_message or "")
    assert outcome.code is None  # the code check never ran


@pytest.mark.parametrize(
    "answer",
    [
        "Promotion removes boilerplate [S1][S3].",
        "Promotion removes boilerplate [S1, S3].",
        "Promotion removes boilerplate (S1) [S1].",  # (S1) is no marker
    ],
)
def test_valid_markers_pass(settings, answer) -> None:
    outcome = check(answer, with_rate(settings, 0.0))
    assert outcome.result == "ok" and outcome.citations is not None and outcome.citations.valid
    assert outcome.answer == answer


def test_malformed_citation_groups_fail_the_check(settings) -> None:
    outcome = check("Promotion removes boilerplate [S1][S1-S9].", settings)
    assert outcome.result == "repair" and outcome.failure == "citations"
    assert outcome.citations is not None and outcome.citations.markers_unknown == ["[S1-S9]"]
    assert "[S1-S9]" in (outcome.repair_message or "")


def test_an_answer_without_markers_has_no_unknown_marker(settings) -> None:
    c = co.check_citations("Promotion removes boilerplate (S1).", LABEL_MAP)
    assert c.markers_found == [] and c.markers_unknown == [] and c.valid


def test_code_normalization_rules() -> None:
    raw = "  class A  {\n\n\t  public   int $x; // …  \n   }  \n"
    assert co.normalize_code(raw) == "class A {\npublic int $x; // ...\n}"


def test_fence_scanner_reads_every_fence_form() -> None:
    assert co.fenced_code("a\n```php\nx = 1;\n```\nb") == ["x = 1;"]
    assert co.fenced_code("~~~\nx = 1;\n~~~") == ["x = 1;"]
    assert co.fenced_code("````php\n```\nnested\n```\n````") == ["```\nnested\n```"]
    assert co.fenced_code("```php echo 1;```") == ["php echo 1;"]  # one line: no info string
    assert co.fenced_code("A [S1]\n```php\necho 'evil';\n") == ["echo 'evil';"]  # unclosed
    assert co.fenced_code("```\n\n```") == []  # empty: no code to check
    assert co.fenced_code("no fences here") == []


def test_unclosed_and_tilde_fences_with_invented_code_fail(settings) -> None:
    for answer in (
        "Promotion [S1]:\n```php\nclass Invented {}\n",
        "Promotion [S1]:\n~~~\nclass Invented {}\n~~~",
        "Promotion [S1]:\n```class Invented {}```",  # a one-line block at the line start
    ):
        outcome = check(answer, settings)
        assert outcome.result == "repair" and outcome.failure == "code", answer


def test_code_excerpt_with_other_indentation_passes(settings) -> None:
    excerpt = "public function __construct(\n  public string $name,\n  public string $email,\n) {}"
    answer = f"The DTO takes two strings [S2]:\n\n```\n{excerpt}\n```"
    outcome = check(answer, with_rate(settings, 0.0))
    assert outcome.result == "ok" and outcome.code is not None
    assert outcome.code.code_blocks == 1 and outcome.code.code_blocks_unmatched == 0
    assert outcome.code.ok and not outcome.code.plain_text


def test_code_with_an_extra_method_fails(settings) -> None:
    invented = CODE[:-1] + "\n    public function reset(): void {}\n}"
    answer = f"The class [S2]:\n\n```php\n{invented}\n```"
    outcome = check(answer, settings)
    assert outcome.result == "repair" and outcome.failure == "code"
    assert outcome.code is not None and outcome.code.code_blocks_unmatched == 1
    assert "code block" in (outcome.repair_message or "")


def test_three_dots_match_the_books_ellipsis(settings) -> None:
    outcome = check("The rest is left out [S2]:\n\n```php\n// ...\n```", with_rate(settings, 0.0))
    assert outcome.result == "ok" and outcome.code is not None and outcome.code.ok


def test_unfenced_code_is_not_checked_and_code_verbatim_false_skips_the_check(settings) -> None:
    answer = "Call `$dto->reset()` to start over [S2]. public function reset(): void {}"
    outcome = check(answer, with_rate(settings, 0.0))
    assert outcome.result == "ok" and outcome.code is not None and outcome.code.code_blocks == 0
    off = settings.model_copy(
        update={"output": settings.output.model_copy(update={"code_verbatim": False})}
    )
    answer = "Invented [S1]:\n\n```php\nclass Invented {}\n```"
    outcome = check(answer, with_rate(off, 0.0))
    assert outcome.result == "ok" and outcome.code is None


def test_plain_text_context_compares_against_whole_blocks(settings) -> None:
    """An index built before the blocks table serves unfenced context (Stage 15 fallback)."""
    plain = [
        ContextBlock("S1", ["d:05:0039"], f"[S1] Chapter 05, p. 64\n{CODE}", 5, "P", 64, 64, 60)
    ]
    answer = f"The class [S1]:\n\n```php\n{CODE}\n```"
    outcome = check(answer, with_rate(settings, 0.0), context=plain, label_map={"S1": {}})
    assert outcome.result == "ok" and outcome.code is not None
    assert outcome.code.ok and outcome.code.plain_text


# --------------------------------------------------------------------------- order and sampling


def test_a_failing_fixed_check_means_no_judge_call(settings) -> None:
    judge = FakeJudge({"supported": True, "unsupported_claims": []})
    outcome = check(FAKE_S9, with_rate(settings, 1.0), judge=judge, borderline=True)
    assert outcome.result == "repair" and judge.calls == [] and not outcome.sampled


def test_sampling_is_a_deterministic_function_of_the_request_id() -> None:
    assert all(co.is_sampled("r1", 0.2) == co.is_sampled("r1", 0.2) for _ in range(5))
    assert not co.is_sampled("r1", 0.0) and co.is_sampled("r1", 1.0)
    share = sum(co.is_sampled(f"req-{i}", 0.2) for i in range(2000)) / 2000
    assert 0.15 < share < 0.25
    assert co.should_judge(borderline=True, judge_all=False, request_id="r1", rate=0.0)
    assert co.should_judge(borderline=False, judge_all=True, request_id="r1", rate=0.0)
    assert not co.should_judge(borderline=False, judge_all=False, request_id="r1", rate=0.0)


def test_sampled_requests_are_judged_and_others_are_not(settings) -> None:
    judge = FakeJudge({"supported": True, "unsupported_claims": []})
    outcome = check(CLEAN, with_rate(settings, 1.0), judge=judge)
    assert outcome.result == "ok" and outcome.sampled and len(judge.calls) == 1
    assert outcome.verdict is not None and outcome.verdict.supported
    assert outcome.cost_usd == pytest.approx((6400 * 0.1 + 100 * 0.4) / 1e6)
    assert (
        judge.json_schema is not None
        and judge.max_tokens == settings.output.judge_max_output_tokens
    )
    system, messages, _ = judge.calls[0]
    assert system.startswith("You check whether an answer is supported")
    assert messages[0][1].startswith("Context blocks:") and CLEAN in messages[0][1]
    assert "[S2] Chapter 05" in messages[0][1] and "{answer}" not in messages[0][1]

    judge = FakeJudge()
    outcome = check(CLEAN, with_rate(settings, 0.0), judge=judge)
    assert outcome.result == "ok" and not outcome.sampled and judge.calls == []


# --------------------------------------------------------------------------- judge


def test_unfaithful_answer_abstains_without_repair(settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    judge = FakeJudge(
        {"supported": False, "unsupported_claims": ["readonly resets via reflection"]}
    )
    answer = "A readonly property can be reset with reflection [S3]."
    outcome = check(answer, settings, judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.unfaithful"
    assert outcome.answer == ABSTAIN_TEXT and outcome.repair_message is None
    (span,) = spans_named(exporter, "check.faithfulness")
    a = span.attributes
    assert a["openinference.span.kind"] == "LLM" and a["rag.faithful"] is False
    assert json.loads(a["rag.unsupported_claims"]) == ["readonly resets via reflection"]
    assert a["rag.faithfulness_sampled"] is True and a["gen_ai.request.model"] == "fake:judge"
    assert a["gen_ai.usage.input_tokens"] == 6400 and a["gen_ai.usage.output_tokens"] == 100
    assert a["rag.cost_usd"] == pytest.approx(outcome.cost_usd)
    assert a["rag.prompt_version"] == co.load_check_prompts(settings).judge.version


def test_judge_failure_abstains_for_borderline_and_delivers_for_sampled(settings) -> None:
    judge = FakeJudge(TimeoutError("slow"), TimeoutError("slow"))
    outcome = check(CLEAN, settings, judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.judge_unavailable"
    assert len(judge.calls) == 2 and "slow" in (outcome.judge_error or "")

    judge = FakeJudge(TimeoutError("slow"), TimeoutError("slow"))
    outcome = check(CLEAN, with_rate(settings, 1.0), judge=judge)
    assert outcome.result == "ok" and outcome.answer == CLEAN and outcome.sampled
    assert len(judge.calls) == 2 and "slow" in (outcome.judge_error or "")

    judge = FakeJudge("not json at all")  # not a transport error: no retry
    outcome = check(CLEAN, with_rate(settings, 1.0), judge=judge)
    assert outcome.result == "ok" and len(judge.calls) == 1 and outcome.judge_error


def test_judge_verdict_with_the_wrong_shape_is_a_judge_failure(settings) -> None:
    judge = FakeJudge({"supported": "yes", "unsupported_claims": []})
    outcome = check(CLEAN, settings, judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.judge_unavailable"


def test_supported_with_listed_claims_counts_as_unfaithful(settings) -> None:
    judge = FakeJudge({"supported": True, "unsupported_claims": ["reset with reflection"]})
    outcome = check(CLEAN, settings, judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.unfaithful"


def test_any_provider_error_follows_the_judge_failure_policy_but_config_errors_propagate(
    settings,
) -> None:
    from flp_rag.models import ModelConfigError

    class Unauthorized(Exception):
        status_code = 401

    judge = FakeJudge(Unauthorized("bad key"))
    outcome = check(CLEAN, with_rate(settings, 1.0), judge=judge)
    assert (
        outcome.result == "ok"
        and len(judge.calls) == 1
        and "bad key" in (outcome.judge_error or "")
    )
    judge = FakeJudge(Unauthorized("bad key"))
    outcome = check(CLEAN, settings, judge=judge, borderline=True)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.judge_unavailable"

    def placeholder() -> FakeJudge:
        raise ModelConfigError("output.judge_model: placeholder")

    for borderline in (False, True):
        with pytest.raises(ModelConfigError):
            co.check_output(
                CLEAN,
                label_map=LABEL_MAP,
                context=BLOCKS,
                borderline=borderline,
                request_id="r1",
                repair_calls=0,
                settings=with_rate(settings, 1.0),
                judge=placeholder,
                prompts=co.load_check_prompts(settings),
            )


# --------------------------------------------------------------------------- budget


def test_a_used_budget_abstains_on_the_first_failure_here(settings) -> None:
    outcome = check(FAKE_S9, settings, repair_calls=settings.gen.max_repair_calls)
    assert outcome.result == "abstain" and outcome.abstain_reason == "output.invalid_after_repair"
    assert outcome.failure == "citations" and outcome.repair_message is None
    assert outcome.answer == ABSTAIN_TEXT


# --------------------------------------------------------------------------- prompts


def test_repair_messages_come_from_a_versioned_prompt_file(settings) -> None:
    prompts = co.load_check_prompts(settings)
    path = ROOT / settings.output.repair_prompt_file
    assert prompts.repair.version == gen.prompt_version(path) and len(prompts.repair.version) == 8
    assert set(prompts.repair.sections) >= set(co.REPAIR_SECTIONS)
    message = prompts.repair.message("citations", markers="[S9]")
    assert message.startswith(prompts.repair.sections["citations"].replace("{markers}", "[S9]"))
    assert message.endswith(prompts.repair.sections["rewrite"])
    assert prompts.judge.version == gen.prompt_version(ROOT / settings.output.judge_prompt_file)
    with pytest.raises(ValueError, match="rewrite"):
        co.parse_repair_sections("## citations\nonly one section\n")


# --------------------------------------------------------------------------- node and spans


def node_state(answer: str, **kw: Any) -> dict[str, Any]:
    state = initial_state("What is promotion?", request_id="r1", debug=True, **kw)
    return {
        **state,
        "answer": answer,
        "status": "answered",
        "context": BLOCKS,
        "label_map": LABEL_MAP,
    }


def test_node_writes_the_parent_span_attributes_and_the_three_edge_values(settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    judge = FakeJudge({"supported": True, "unsupported_claims": []})
    node = build._traced(
        "check_output", co.make_check_output(with_rate(settings, 0.0), lambda: judge)
    )

    out = node(node_state(CLEAN))
    assert out["check_result"] == "ok" and out["status"] == "answered" and out["answer"] == CLEAN
    assert out["repair_failure"] is None
    (parent,) = spans_named(exporter, "guardrails.output")
    a = parent.attributes
    assert a["rag.check_result"] == "ok" and a["rag.citations_valid"] is True
    assert a["rag.code_verbatim_ok"] is True and a["rag.faithfulness_sampled"] is False
    assert "rag.faithful" not in a and "rag.rejected_answer" not in a
    assert [s.name for s in exporter.get_finished_spans()] == [
        "check.citations",
        "check.code",
        "guardrails.output",
    ]
    assert out["debug_info"]["check_output"]["result"] == "ok"

    exporter.clear()
    out = node(node_state(FAKE_S9))
    assert out["check_result"] == "repair" and out["rejected_answer"] == FAKE_S9
    assert "[S9]" in out["repair_failure"] and "status" not in out
    (parent,) = spans_named(exporter, "guardrails.output")
    a = parent.attributes
    assert a["rag.check_result"] == "repair" and a["rag.citations_valid"] is False
    assert a["rag.rejected_answer"] == FAKE_S9 and "rag.code_verbatim_ok" not in a
    (cit,) = spans_named(exporter, "check.citations")
    assert cit.attributes["openinference.span.kind"] == "TOOL"
    assert json.loads(cit.attributes["rag.markers_unknown"]) == ["S9"]
    assert json.loads(cit.attributes["rag.markers_found"]) == ["S9"]
    assert cit.parent.span_id == parent.context.span_id

    exporter.clear()
    out = node(node_state(FAKE_S9) | {"repair_calls": 1})
    assert out["check_result"] == "abstain" and out["status"] == "no_information"
    assert out["answer"] == ABSTAIN_TEXT and out["sources"] == []
    assert out["abstain_reason"] == "output.invalid_after_repair"
    (parent,) = spans_named(exporter, "guardrails.output")
    assert parent.attributes["rag.abstain_reason"] == "output.invalid_after_repair"


def test_node_early_stop_keeps_the_stage_16_reason(settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    node = build._traced("check_output", co.make_check_output(settings, FakeJudge))
    state = node_state(ABSTAIN_TEXT) | {
        "status": "no_information",
        "abstain_reason": "gen.model_abstained",
        "sources": [],
    }
    out = node(state)
    assert out["check_result"] == "abstain" and out["status"] == "no_information"
    assert "abstain_reason" not in out  # the reason in the state is kept
    (parent,) = spans_named(exporter, "guardrails.output")
    assert parent.attributes["rag.abstain_reason"] == "gen.model_abstained"
    assert parent.attributes["rag.early_stop"] is True


def test_rejected_answer_is_not_recorded_when_content_capture_is_off(settings) -> None:
    quiet = with_rate(settings, 0.0).model_copy(
        update={"trace": settings.trace.model_copy(update={"capture_content": False})}
    )
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(quiet, exporter=exporter, force=True)
    node = build._traced("check_output", co.make_check_output(quiet, FakeJudge))
    node(node_state(FAKE_S9))
    (parent,) = spans_named(exporter, "guardrails.output")
    assert "rag.rejected_answer" not in parent.attributes
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)


# --------------------------------------------------------------------------- graph loop


def loop_nodes(settings: Settings, chat: FakeChat, judge: FakeJudge) -> build.GraphNodes:
    """Stub retrieval and gate; real Stage 16 and Stage 17; the real Stage 18 stub from build."""
    stubs = stub_nodes().as_dict()
    real = build.default_nodes(build.Deps(settings=settings, chat=chat, judge=judge))

    def assemble(state: Any) -> dict[str, Any]:
        return {"context": BLOCKS, "label_map": LABEL_MAP}

    return build.GraphNodes(
        **{
            **stubs,
            "assemble": assemble,
            "generate": real.generate,
            "check_output": real.check_output,
            "respond": real.respond,
        }
    )


def test_graph_loop_repairs_a_fabricated_marker_once(settings) -> None:
    """The Done when: `[S9]` never reaches the client; one `check.citations` span with
    `rag.citations_valid = false`, one repair call, a clean final answer."""
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    chat = FakeChat(FAKE_S9, "Promotion removes boilerplate [S1].")
    judge = FakeJudge()
    app = build.build_graph(loop_nodes(with_rate(settings, 0.0), chat, judge))
    state = app.invoke(initial_state("What is promotion?", request_id="r1", debug=True))

    assert state["visited"][-5:] == [
        "generate",
        "check_output",
        "generate",
        "check_output",
        "respond",
    ]
    assert (
        state["status"] == "answered" and state["answer"] == "Promotion removes boilerplate [S1]."
    )
    assert state["repair_calls"] == 1 and state["check_result"] == "ok" and judge.calls == []
    assert "[S9]" not in state["response"].answer
    assert len(chat.calls) == 2
    system, messages, _ = chat.calls[1]
    prompts = gen.load_prompts(settings)
    assert system == prompts.strict.system  # the repair uses the strict prompt
    assert messages[1] == ("assistant", FAKE_S9) and "[S9]" in messages[2][1]
    assert messages[2][1].rstrip().endswith(ABSTAIN_TEXT)
    assert state["prompt_version"] == prompts.strict.version

    cits = spans_named(exporter, "check.citations")
    assert [
        s.attributes["rag.citations_valid"] for s in spans_named(exporter, "guardrails.output")
    ] == [False, True]
    assert len(cits) == 2 and json.loads(cits[0].attributes["rag.markers_unknown"]) == ["S9"]
    gens = spans_named(exporter, "llm.generate")
    assert [s.attributes["rag.repair_calls"] for s in gens] == [0, 1]
    assert state["response"].debug["check_output"]["result"] == "ok"


def test_graph_abstains_when_stage_16_spent_the_repair_and_the_answer_still_fails(settings) -> None:
    chat = FakeChat("Promotion removes boilerplate.", FAKE_S9)  # no marker, then a fake one
    app = build.build_graph(loop_nodes(with_rate(settings, 0.0), chat, FakeJudge()))
    state = app.invoke(initial_state("What is promotion?", request_id="r1"))
    assert state["visited"][-3:] == ["generate", "check_output", "respond"]
    assert state["status"] == "no_information" and state["answer"] == ABSTAIN_TEXT
    assert state["abstain_reason"] == "output.invalid_after_repair" and len(chat.calls) == 2
    assert state["repair_calls"] == 1 and state["response"].sources == []


def test_graph_borderline_request_is_judged_and_an_unfaithful_one_abstains(settings) -> None:
    chat = FakeChat("Readonly can be reset with reflection [S3].")
    judge = FakeJudge({"supported": False, "unsupported_claims": ["reset with reflection"]})
    nodes = loop_nodes(with_rate(settings, 0.0), chat, judge)
    borderline = {
        **stub_nodes(gate="borderline").as_dict(),
        **{k: getattr(nodes, k) for k in ("assemble", "generate", "check_output", "respond")},
    }
    state = build.build_graph(build.GraphNodes(**borderline)).invoke(
        initial_state("Can readonly be reset?", request_id="r1")
    )
    assert state["borderline"] and len(judge.calls) == 1
    assert state["status"] == "no_information" and state["abstain_reason"] == "output.unfaithful"
    assert state["response"].answer == ABSTAIN_TEXT and state["cost_usd"] > 0


def test_routing_after_the_output_check() -> None:
    assert build.route_after_check({"check_result": "ok"}) == "respond"
    assert build.route_after_check({"check_result": "abstain"}) == "respond"
    assert build.route_after_check({"check_result": "repair"}) == "generate"
    with pytest.raises(KeyError):
        build.route_after_check({})
