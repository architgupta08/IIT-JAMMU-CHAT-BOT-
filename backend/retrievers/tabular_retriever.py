"""
backend/retrievers/tabular_retriever.py — Tabular Retriever for structured data
=============================================================================
Intercepts queries about placements, fees, cutoffs, and department faculty lists.
Loads data from the structured JSON schemas and returns formatted markdown tables
as retrieval context.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Resolve paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(backend_dir).parent.parent
data_dir = project_root / "data" / "processed"


# ── Department detection map ───────────────────────────────────────────
# Maps canonical faculty_data.json department string → list of query keywords
DEPT_KEYWORD_MAP: Dict[str, List[str]] = {
    "Computer Science & Engineering": [
        "computer science", "computer engineering", "cse", "comp sci", "cs department",
    ],
    "Electrical Engineering": [
        "electrical engineering", "electrical", "ee department", "ee dept",
    ],
    "Mechanical Engineering": [
        "mechanical engineering", "mechanical", "me department", "me dept",
    ],
    "Civil Engineering": [
        "civil engineering", "civil", "ce department", "ce dept",
    ],
    "Chemical Engineering": [
        "chemical engineering", "chemical", "che department", "che dept",
    ],
    "Materials Engineering": [
        "materials engineering", "materials", "mme", "metallurgical",
    ],
    "Bioscience and Bioengineering": [
        "bioscience", "bioengineering", "bsbe", "biosciences",
    ],
    "Mathematics": ["mathematics", "maths", "math department"],
    "Physics": ["physics", "physics department"],
    "Chemistry": ["chemistry", "chemistry department"],
    "Humanities & Social Sciences": [
        "humanities", "social sciences", "hss", "humanities department",
    ],
}

# Designation sort order (lower index = higher priority)
_DESIG_ORDER = [
    "director",
    "professor (director)",
    "dean",
    "head of department",
    "hod",
    "professor",
    "associate professor",
    "assistant professor",
]


def _desig_rank(designation: str) -> int:
    """Return sort key for designation (lower = appears first)."""
    d = designation.lower().strip()
    for rank, key in enumerate(_DESIG_ORDER):
        if key in d:
            return rank
    return len(_DESIG_ORDER)


# Department slug map for source URL construction
_DEPT_SLUG: Dict[str, str] = {
    "Computer Science & Engineering": "computer_science_engineering",
    "Electrical Engineering": "electrical_engineering",
    "Mechanical Engineering": "mechanical_engineering",
    "Civil Engineering": "civil_engineering",
    "Chemical Engineering": "chemical_engineering",
    "Materials Engineering": "materials_engineering",
    "Bioscience and Bioengineering": "bioscience_and_bioengineering",
    "Mathematics": "mathematics",
    "Physics": "physics",
    "Chemistry": "chemistry",
    "Humanities & Social Sciences": "humanities_social_sciences",
}

# Keywords that together signal a "list all faculty" query
_FACULTY_LIST_KEYWORDS = [
    "who are the faculty",
    "list of faculty",
    "faculty members",
    "faculty member",
    "all faculty",
    "faculty of",
    "faculty in",
    "professors in",
    "professors of",
    "who teaches in",
    "who teaches",
    "staff in",
    "staff of",
    "faculty list",
    "show faculty",
    "list faculty",
    "name the faculty",
    "names of faculty",
    "which faculty",
    "which professors",
    "faculty members in",
    "show me the faculty",
]


class TabularRetriever:
    """Rules-based retriever that serves structured tables for placements, fees, cutoffs, and faculty lists."""

    def __init__(self):
        self.placements_path = data_dir / "placements.json"
        self.fees_path = data_dir / "fees.json"
        self.cutoffs_path = data_dir / "cutoffs.json"
        self.faculty_path = data_dir / "faculty_data.json"

    def _load_json(self, path: Path) -> Optional[Any]:
        if not path.exists():
            logger.warning(f"Structured JSON file not found at: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading JSON from {path}: {e}")
            return None

    def retrieve_context(self, query: str) -> Optional[str]:
        """
        Scan query for keywords and return relevant structured context if matched.
        Returns None if query is not targeting placements, fees, cutoffs, or faculty lists.
        """
        q_lower = query.lower()
        contexts = []

        # 1. FACULTY LIST KEYWORDS — check first so department faculty queries get full list
        if any(kw in q_lower for kw in _FACULTY_LIST_KEYWORDS):
            dept = self._detect_department(q_lower)
            if dept:
                faculty_ctx = self._get_faculty_context(dept)
                if faculty_ctx:
                    contexts.append(faculty_ctx)

        # 2. PLACEMENT KEYWORDS
        if any(kw in q_lower for kw in ["placement", "package", "salary", "ctc", "lpa", "recruit", "higher studies"]):
            contexts.append(self._get_placement_context(q_lower))

        # 3. FEE KEYWORDS
        if any(kw in q_lower for kw in ["fee", "tuition", "hostel charge", "mess fee", "mess circular"]):
            contexts.append(self._get_fee_context(q_lower))

        # 4. CUTOFF KEYWORDS
        if any(kw in q_lower for kw in ["cutoff", "cut-off", "cut off", "gate score", "closing rank", "jee rank", "jee cutoff"]):
            contexts.append(self._get_cutoff_context(q_lower))

        valid_contexts = [c for c in contexts if c]
        if valid_contexts:
            return "\n\n".join(valid_contexts)
        return None

    def _detect_department(self, q_lower: str) -> Optional[str]:
        """Detect the target department from a lowercased query string."""
        for dept_name, keywords in DEPT_KEYWORD_MAP.items():
            if any(kw in q_lower for kw in keywords):
                return dept_name
        return None

    def _get_faculty_context(self, dept_name: str) -> str:
        """Load faculty from faculty_data.json, filter by department, sort, and format as Markdown table."""
        data = self._load_json(self.faculty_path)
        if not data or not isinstance(data, list):
            return ""

        # Filter faculty whose department list contains the target department
        faculty_list = []
        for fac in data:
            depts = fac.get("department", [])
            if not isinstance(depts, list):
                depts = [str(depts)]
            if any(dept_name.lower() in d.lower() for d in depts):
                faculty_list.append(fac)

        if not faculty_list:
            return ""

        # Sort: by designation hierarchy, then alphabetically by name
        faculty_list.sort(key=lambda f: (
            _desig_rank(f.get("designation", "")),
            (f.get("faculty_name") or "").lower()
        ))

        # Build Markdown table
        dept_slug = _DEPT_SLUG.get(dept_name, "")
        source_url = (
            f"https://www.iitjammu.ac.in/{dept_slug}/faculty-list"
            if dept_slug else "https://www.iitjammu.ac.in"
        )

        lines = [
            f"### OFFICIAL IIT JAMMU FACULTY DIRECTORY — {dept_name.upper()}",
            f"Source: {source_url}",
            f"Total Faculty: {len(faculty_list)}",
            "",
            "| # | Name | Designation | Email |",
            "|---|------|-------------|-------|",
        ]
        for idx, fac in enumerate(faculty_list, start=1):
            salutation = fac.get("salutation", "").strip()
            name = fac.get("faculty_name", "Unknown").strip()
            full_name = f"{salutation} {name}".strip() if salutation else name
            designation = fac.get("designation", "Faculty").strip()
            email = fac.get("email", "").strip()
            email_str = email if email else "—"
            lines.append(f"| {idx} | {full_name} | {designation} | {email_str} |")

        return "\n".join(lines)

    def _get_placement_context(self, q_lower: str) -> str:
        data = self._load_json(self.placements_path)
        if not data:
            return ""

        context_parts = []
        context_parts.append("### OFFICIAL IIT JAMMU PLACEMENT & RECRUITMENT STATISTICS (LPA)")

        btech_data = data.get("btech_placements", {})
        if btech_data:
            for year in sorted(btech_data.keys(), reverse=True):
                context_parts.append(f"\n**B.Tech Placements {year}:**")
                context_parts.append("| Branch | Placement % | Avg Salary (LPA) | Max Salary (LPA) | Min Salary (LPA) |")
                context_parts.append("|---|---|---|---|---|")
                for item in btech_data[year]:
                    context_parts.append(f"| {item['branch']} | {item['placed_pct']}% | {item['avg']} | {item['max']} | {item['min']} |")

        mtech_data = data.get("mtech_placements", {})
        if mtech_data:
            for year in sorted(mtech_data.keys(), reverse=True):
                context_parts.append(f"\n**M.Tech Placements {year}:**")
                context_parts.append("| Specialization | Placement % | Avg Salary (LPA) | Max Salary (LPA) | Min Salary (LPA) |")
                context_parts.append("|---|---|---|---|---|")
                for item in mtech_data[year]:
                    context_parts.append(f"| {item['branch']} | {item['placed_pct']}% | {item['avg']} | {item['max']} | {item['min']} |")

        return "\n".join(context_parts)

    def _get_fee_context(self, q_lower: str) -> str:
        data = self._load_json(self.fees_path)
        if not data:
            return ""

        context_parts = []
        context_parts.append("### OFFICIAL IIT JAMMU FEE STRUCTURE (ACADEMIC YEAR 2025-2026)")
        context_parts.append("Source: https://www.iitjammu.ac.in/fee-structure")

        btech = data.get("btech_2025_26", {})
        
        # Tuition Fees
        context_parts.append("\n**B.Tech 2025 Batch Tuition Fee Waiver Rules (Per Semester):**")
        context_parts.append(
            "| Student Category / Income Bracket | Semester Tuition Fee | Waiver / Subsidy Details |\n"
            "| --- | --- | --- |"
        )
        tuition = btech.get("tuition_fee_per_semester", {})
        for cat, details in tuition.items():
            cat_display = cat.replace("_", " ")
            context_parts.append(
                f"| {cat_display} | Rs. {details['amount_rs']:,} | {details['notes']} |"
            )

        # Hostel Fees
        context_parts.append("\n**B.Tech 2025 Batch Hostel Charges (Per Semester):**")
        hostel = btech.get("hostel_fee_per_semester", {})
        context_parts.append(
            f"- **Single Occupancy Room:** Rs. {hostel.get('single_occupancy_rs', 0):,} per semester.\n"
            f"- **Shared Occupancy Room:** Rs. {hostel.get('shared_occupancy_rs', 0):,} per semester.\n"
            f"- **Details:** {hostel.get('notes', '')}"
        )

        # Mess fee
        context_parts.append("\n**Mess and Laundry Fee Rules:**")
        context_parts.append(
            f"- {btech.get('mess_fee', {}).get('notes', '')}\n"
            f"- Students should refer to the latest official Mess Fee Circular AY 2025-26 for exact payment amounts."
        )

        return "\n".join(context_parts)

    def _get_cutoff_context(self, q_lower: str) -> str:
        data = self._load_json(self.cutoffs_path)
        if not data:
            return ""

        context_parts = []
        context_parts.append("### OFFICIAL IIT JAMMU ADMISSION CUTOFF GUIDELINES")
        
        btech = data.get("btech_jee_advanced", {})
        context_parts.append("\n**B.Tech JEE Advanced Cutoffs & Closing Ranks (General Category):**")
        context_parts.append("Source: https://www.iitjammu.ac.in/Programme/ugadmissions")
        context_parts.append(
            f"- **Computer Science and Engineering (CSE):** Typically closes within the **{btech.get('cse', {}).get('closing_rank_estimate', '')}** ranks in JEE Advanced.\n"
            f"- **Other Branches (Civil, Chemical, Materials):** Typically close around **{btech.get('other_branches', {}).get('closing_rank_estimate', '')}** in JEE Advanced."
        )

        btech_previous = btech.get("previous_years", {})
        if btech_previous:
            context_parts.append("\n**B.Tech JEE Advanced Closing Ranks (General Category - Gender Neutral) for Previous Years:**")
            for year in sorted(btech_previous.keys(), reverse=True):
                context_parts.append(f"\n### JEE Advanced Cutoff Ranks - Year {year}")
                context_parts.append(
                    "| Branch / Program | Closing Rank (Round 6) |\n"
                    "| --- | ---: |"
                )
                for item in btech_previous[year]:
                    branch = item.get("branch", "")
                    rank = item.get("closing_rank", "Nil")
                    context_parts.append(f"| {branch} | {rank:,} |" if isinstance(rank, int) else f"| {branch} | {rank} |")


        mtech = data.get("mtech_gate", {}).get("general_rules", {})
        context_parts.append("\n**M.Tech GATE Requirements & Admissions Guidelines:**")
        context_parts.append("Source: https://www.iitjammu.ac.in/Programme/pgadmissions")
        context_parts.append(
            f"- **GATE Cutoff Scores:** {mtech.get('numerical_cutoffs', '')}\n"
            f"- **Selection Process:** {mtech.get('selection_criteria', '')}\n"
            f"- **CRITICAL CLARIFICATION:** {mtech.get('clarification', '')}\n"
            f"- **IMPORTANT INSTRUCTION FOR AI ASSISTANT:** If the user asks for GATE cutoffs but does not specify a year, you MUST first state the general guidelines (including that numerical cutoffs are not fixed and vary depending on applicant pool size, vacancies, and department) and then provide the specific numerical scores from the most recent year's table below (e.g., Year 2025)."
        )


        previous_years = data.get("mtech_gate", {}).get("previous_years", {})
        if previous_years:
            context_parts.append("\n**M.Tech GATE Cutoff Scores (Closing Scores of Admitted Candidates) for Previous Years:**")
            for year in sorted(previous_years.keys(), reverse=True):
                context_parts.append(f"\n### GATE Cutoff Scores - Year {year}")
                context_parts.append(
                    "| Specialization | GEN | OBC-NCL | EWS | SC | ST | PwD |\n"
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
                )
                for item in previous_years[year]:
                    spec = item.get("specialization", "")
                    gen = item.get("GEN", "Nil")
                    obc = item.get("OBC_NCL", "Nil")
                    ews = item.get("EWS", "Nil")
                    sc = item.get("SC", "Nil")
                    st = item.get("ST", "Nil")
                    pwd = item.get("PwD", "Nil")
                    context_parts.append(
                        f"| {spec} | {gen} | {obc} | {ews} | {sc} | {st} | {pwd} |"
                    )

        return "\n".join(context_parts)
