import re
import json
import logging
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import jsonschema
from jsonschema.exceptions import ValidationError, SchemaError

logger = logging.getLogger("crossverify.worker.evaluator.structural")


class StructuralComparisonResult(BaseModel):
    """Result of JSON syntax and schema validation between Champion and Candidate."""
    is_structured: bool = Field(
        ...,
        description="Whether the output was expected to be structured JSON"
    )
    syntax_valid: bool = Field(
        ...,
        description="Whether the Candidate output contains syntactically valid JSON"
    )
    schema_matched: bool = Field(
        ...,
        description="Whether the Candidate output adheres to the expected or inferred JSON schema"
    )
    structural_valid: bool = Field(
        ...,
        description="Overall structural validity verdict"
    )
    candidate_parsed: Optional[Any] = Field(
        None,
        description="Parsed JSON payload from Candidate if valid"
    )
    error_message: Optional[str] = Field(
        None,
        description="Validation error description if structural defects detected"
    )


class StructuralEvaluator:
    """
    Evaluator for function calling, tool use, and structured JSON outputs.
    Ensures that Candidate models adhere strictly to JSON syntax and required schemas,
    preventing runtime deserialization crashes in downstream agentic systems.
    """

    JSON_BLOCK_REGEX = re.compile(
        r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```",
        re.DOTALL | re.IGNORECASE
    )

    @classmethod
    def extract_json(cls, text: str) -> Tuple[Optional[Any], bool]:
        """
        Extract and parse JSON from raw text or markdown code fence blocks.
        Returns (parsed_data, is_valid_json).
        """
        if not text or not text.strip():
            return None, False

        stripped = text.strip()

        # 1. Direct JSON parse attempt
        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try:
                return json.loads(stripped), True
            except json.JSONDecodeError:
                pass

        # 2. Markdown fenced block attempt (e.g. ```json { ... } ```)
        matches = cls.JSON_BLOCK_REGEX.findall(stripped)
        for candidate_snippet in matches:
            try:
                return json.loads(candidate_snippet.strip()), True
            except json.JSONDecodeError:
                continue

        # 3. Best-effort substring extraction
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(stripped[first_brace:last_brace + 1]), True
            except json.JSONDecodeError:
                pass

        first_bracket = stripped.find("[")
        last_bracket = stripped.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            try:
                return json.loads(stripped[first_bracket:last_bracket + 1]), True
            except json.JSONDecodeError:
                pass

        return None, False

    def validate_against_schema(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate parsed data against JSON schema specification."""
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            return False, f"Schema validation error: {e.message} at path: '{e.json_path}'"
        except SchemaError as e:
            logger.error(f"Malformed JSON schema provided for validation: {e}")
            return False, f"Invalid JSON schema: {e.message}"

    def evaluate(
        self,
        champion_text: str,
        candidate_text: str,
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> StructuralComparisonResult:
        """
        Evaluate structural validity:
        - If expected_schema is provided, enforces it on Candidate output.
        - If Champion produced JSON, Candidate MUST also produce valid JSON.
        - If neither is structured, passes as structurally valid prose.
        """
        champion_data, champion_is_json = self.extract_json(champion_text)
        candidate_data, candidate_is_json = self.extract_json(candidate_text)

        # Case A: Explicit schema provided
        if expected_schema:
            if not candidate_is_json:
                return StructuralComparisonResult(
                    is_structured=True,
                    syntax_valid=False,
                    schema_matched=False,
                    structural_valid=False,
                    candidate_parsed=None,
                    error_message="Candidate failed to produce parseable JSON required by schema"
                )

            schema_matched, schema_err = self.validate_against_schema(
                candidate_data, expected_schema
            )
            return StructuralComparisonResult(
                is_structured=True,
                syntax_valid=True,
                schema_matched=schema_matched,
                structural_valid=schema_matched,
                candidate_parsed=candidate_data,
                error_message=schema_err
            )

        # Case B: Implicit structured mode inferred from Champion output
        if champion_is_json:
            if not candidate_is_json:
                return StructuralComparisonResult(
                    is_structured=True,
                    syntax_valid=False,
                    schema_matched=False,
                    structural_valid=False,
                    candidate_parsed=None,
                    error_message="Champion output structured JSON but Candidate returned unparseable prose"
                )

            # Check matching root data types (e.g. both dicts or both lists)
            if type(champion_data) is not type(candidate_data):
                err = f"JSON type mismatch: Champion produced {type(champion_data).__name__}, Candidate produced {type(candidate_data).__name__}"
                return StructuralComparisonResult(
                    is_structured=True,
                    syntax_valid=True,
                    schema_matched=False,
                    structural_valid=False,
                    candidate_parsed=candidate_data,
                    error_message=err
                )

            # If both are dicts, check for required keys coverage
            if isinstance(champion_data, dict) and isinstance(candidate_data, dict):
                champion_keys = set(champion_data.keys())
                candidate_keys = set(candidate_data.keys())
                missing_keys = champion_keys - candidate_keys
                if missing_keys:
                    err = f"Candidate JSON missing expected keys: {sorted(list(missing_keys))}"
                    return StructuralComparisonResult(
                        is_structured=True,
                        syntax_valid=True,
                        schema_matched=False,
                        structural_valid=False,
                        candidate_parsed=candidate_data,
                        error_message=err
                    )

            return StructuralComparisonResult(
                is_structured=True,
                syntax_valid=True,
                schema_matched=True,
                structural_valid=True,
                candidate_parsed=candidate_data,
                error_message=None
            )

        # Case C: Freeform natural language prose (unstructured)
        return StructuralComparisonResult(
            is_structured=False,
            syntax_valid=True,
            schema_matched=True,
            structural_valid=True,
            candidate_parsed=None,
            error_message=None
        )
