"""
autocomplete/service.py — Autocomplete Service
===============================================
Combines trie-based prefix search with fuzzy matching (Levenshtein)
to provide smart typeahead suggestions.

Suggestions come from:
  - Faculty names
  - Department names
  - Program names (B.Tech, M.Tech, Ph.D, etc.)
  - Common queries / FAQs
  - Research labs
  - Event types

The index is built from ChromaDB metadata and rebuilt
when the scraper completes a cycle.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from autocomplete.trie import Trie

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A single autocomplete suggestion."""
    text: str
    category: str
    score: float


# ── Comprehensive Master Autocomplete Index ────────────────────────
STATIC_SUGGESTIONS = {

    # ── Programs ──────────────────────────────────────────────────
    "B.Tech Admission": ("program", 100),
    "M.Tech Admission": ("program", 95),
    "Ph.D Admission": ("program", 90),
    "M.Sc Program": ("program", 85),
    "MBA Program": ("program", 80),
    "Dual Degree Program": ("program", 78),
    "B.Tech in Computer Science and Engineering": ("program", 90),
    "B.Tech in Electrical Engineering": ("program", 88),
    "B.Tech in Mechanical Engineering": ("program", 86),
    "B.Tech in Civil Engineering": ("program", 84),
    "B.Tech in Chemical Engineering": ("program", 82),
    "B.Tech in Mathematics and Computing": ("program", 80),
    "B.Tech in Biosciences and Bioengineering": ("program", 78),
    "M.Tech in CSE": ("program", 85),
    "M.Tech in Artificial Intelligence and Machine Learning": ("program", 87),
    "M.Tech in Data Science": ("program", 83),
    "M.Tech in Power Systems": ("program", 80),
    "M.Tech in Structural Engineering": ("program", 78),
    "M.Tech in Thermal Engineering": ("program", 78),
    "Post Graduate Diploma in Cyber Security": ("program", 82),
    "Executive M.Tech in GenAI and Data Science": ("program", 85),
    "Ph.D in Computer Science and Engineering": ("program", 88),
    "Ph.D in Electrical Engineering": ("program", 85),
    "Ph.D in Mechanical Engineering": ("program", 83),
    "Ph.D in Physics": ("program", 80),
    "Ph.D in Mathematics": ("program", 80),
    "Ph.D in Chemistry": ("program", 78),

    # ── Departments ───────────────────────────────────────────────
    "Computer Science and Engineering": ("department", 90),
    "Electrical Engineering": ("department", 88),
    "Mechanical Engineering": ("department", 86),
    "Civil Engineering": ("department", 84),
    "Chemical Engineering": ("department", 82),
    "Mathematics and Computing": ("department", 80),
    "Physics Department": ("department", 78),
    "Chemistry Department": ("department", 76),
    "Humanities and Social Sciences": ("department", 74),
    "Materials Engineering": ("department", 72),
    "Biosciences and Bioengineering": ("department", 70),
    "Interdisciplinary Program": ("department", 68),

    # ── Faculty — Computer Science & Engineering ──────────────────
    "Manoj Singh Gaur": ("faculty", 98),
    "Karan Nathwani": ("faculty", 90),
    "Vinit Jakhetiya": ("faculty", 88),
    "Gaurav Varshney": ("faculty", 88),
    "Pankaj Chauhan": ("faculty", 86),
    "Amit Kumar Misra": ("faculty", 85),
    "Deepak Kumar Sharma": ("faculty", 85),
    "Rahul Gupta": ("faculty", 84),
    "Sumit Srivastava": ("faculty", 84),
    "Sanjeev Sofat": ("faculty", 83),
    "Saurabh Bilgaiyan": ("faculty", 83),
    "Subhash Chander Gupta": ("faculty", 82),
    "Arvind Selwal": ("faculty", 82),
    "Rohit Beniwal": ("faculty", 82),
    "Pradeep Kumar Gupta": ("faculty", 81),
    "Vishal Goyal": ("faculty", 81),

    # ── Faculty — Electrical Engineering ─────────────────────────
    "Badri Narayan Subudhi": ("faculty", 95),
    "Shikha Baghel": ("faculty", 90),
    "Anup Shukla": ("faculty", 88),
    "Kushmanda Saurav": ("faculty", 87),
    "Brajesh Kumar": ("faculty", 85),
    "Arun Kumar Verma": ("faculty", 85),
    "Neha Gupta": ("faculty", 84),
    "Ravi Shankar": ("faculty", 83),
    "Sarita Sarangal": ("faculty", 83),
    "Akash Saxena": ("faculty", 82),
    "Dhruv Mahajan": ("faculty", 82),

    # ── Faculty — Mechanical Engineering ─────────────────────────
    "Vinod Kushvaha": ("faculty", 90),
    "RT Durai Prabhakaran": ("faculty", 87),
    "Ravi Kumar": ("faculty", 85),
    "Chetan Singh Thakur": ("faculty", 84),
    "Rajesh Kumar Singh": ("faculty", 83),
    "Anil Kumar Sharma": ("faculty", 83),
    "Amit Rai Dixit": ("faculty", 82),
    "Rahul Vaish": ("faculty", 82),
    "Anoop Kumar Shukla": ("faculty", 81),

    # ── Faculty — Physics ─────────────────────────────────────────
    "Meenakshi Rajeev": ("faculty", 87),
    "Ambika Prasad Shah": ("faculty", 86),
    "Partha Pratim Hazarika": ("faculty", 84),
    "Bhavesh Joshi": ("faculty", 83),
    "Himanshu Agarwal": ("faculty", 82),
    "Pranab Mandal": ("faculty", 82),

    # ── Faculty — Chemistry ───────────────────────────────────────
    "Pankaj Chauhan Chemistry": ("faculty", 85),
    "Ravi Kumar Arun": ("faculty", 85),
    "Shiva S": ("faculty", 83),
    "Amarjeet Singh": ("faculty", 82),
    "Mohit Saraf": ("faculty", 82),

    # ── Faculty — Mathematics ─────────────────────────────────────
    "Sayantan Mandal": ("faculty", 85),
    "Aditi Gupta": ("faculty", 84),
    "Deepak Mishra": ("faculty", 83),
    "Rajesh Kumar Mathematics": ("faculty", 82),
    "Amit Kumar Mathematics": ("faculty", 82),
    "Poonam Rani": ("faculty", 81),

    # ── Faculty — HSS ─────────────────────────────────────────────
    "Subhash Chandra HSS": ("faculty", 82),
    "Rekha Mahajan": ("faculty", 81),
    "Sanjeev Kumar HSS": ("faculty", 80),

    # ── Faculty — Research Questions ─────────────────────────────
    "Who is Professor Manoj Singh Gaur?": ("faculty", 95),
    "Who is Professor Badri Narayan Subudhi?": ("faculty", 92),
    "Who is Karan Nathwani?": ("faculty", 90),
    "Who is Professor Vinit Jakhetiya?": ("faculty", 88),
    "Who is Professor Gaurav Varshney?": ("faculty", 88),
    "Who is Shikha Baghel?": ("faculty", 87),
    "Who is Professor Kushmanda Saurav?": ("faculty", 86),
    "Who is Vinod Kushvaha?": ("faculty", 86),
    "Who is Professor Anup Shukla?": ("faculty", 85),
    "Who is Aditi Gupta?": ("faculty", 84),
    "Who is Sayantan Mandal?": ("faculty", 83),
    "What are the research interests of Professor Badri Narayan Subudhi?": ("faculty", 88),
    "What are the research interests of Professor Vinit Jakhetiya?": ("faculty", 86),
    "What are the research interests of Professor Gaurav Varshney?": ("faculty", 85),

    # ── Admissions ────────────────────────────────────────────────
    "What courses are offered at Indian Institute of Technology Jammu?": ("faq", 98),
    "How can I apply for BTech admission?": ("faq", 97),
    "What is the eligibility for MTech admission?": ("faq", 96),
    "What is the admission procedure for PhD?": ("faq", 95),
    "What is the fee structure for BTech?": ("faq", 99),
    "What is the fee structure for MTech?": ("faq", 95),
    "What is the hostel fee at Indian Institute of Technology Jammu?": ("faq", 94),
    "Is hostel accommodation compulsory for first-year students?": ("faq", 90),
    "What are the cutoff marks for BTech admission?": ("faq", 92),
    "What are the GATE cutoff requirements for MTech?": ("faq", 91),
    "Can I apply for direct PhD after BTech?": ("faq", 90),
    "What scholarships are available for students?": ("faq", 89),
    "Does IIT Jammu provide financial assistance?": ("faq", 88),
    "How can international students apply?": ("faq", 87),
    "What documents are required for admission?": ("faq", 88),
    "Is there any entrance exam for PhD admission?": ("faq", 86),
    "What is the last date for admission applications?": ("faq", 85),
    "How can I prepare for MTech admission at IIT Jammu?": ("faq", 84),
    "Can I apply for multiple programs at IIT Jammu?": ("faq", 83),

    # ── Placements ────────────────────────────────────────────────
    "What is the placement record of IIT Jammu?": ("placement", 94),
    "Which companies visit IIT Jammu for placements?": ("placement", 93),
    "What is the highest package offered at IIT Jammu?": ("placement", 92),
    "What is the average package for CSE students?": ("placement", 91),
    "Does IIT Jammu provide internship opportunities?": ("placement", 90),
    "How can I apply for internships through IIT Jammu?": ("placement", 88),
    "How can I access previous years placement statistics and recruiter data?": ("placement", 85),
    "Which departments have the highest placement rates in recent years?": ("placement", 84),

    # ── Academic ──────────────────────────────────────────────────
    "When does the academic session start?": ("faq", 85),
    "What is the academic calendar for this semester?": ("faq", 84),
    "When are semester exams conducted?": ("faq", 83),
    "How does semester registration work?": ("faq", 82),
    "What is the grading system at IIT Jammu?": ("faq", 81),
    "What is the minimum CPI requirement?": ("faq", 80),
    "How can I get my transcript?": ("faq", 79),
    "What is the attendance policy?": ("faq", 78),
    "How can I change my branch?": ("faq", 77),
    "What are the eligibility criteria for branch change?": ("faq", 76),
    "Can students participate in exchange programs?": ("faq", 75),
    "Does IIT Jammu offer international collaborations?": ("faq", 74),

    # ── Campus & Facilities ───────────────────────────────────────
    "What are the hostel facilities available?": ("faq", 88),
    "Are single rooms available in hostels?": ("faq", 85),
    "What are the mess charges?": ("faq", 84),
    "Does the campus have WiFi facilities?": ("faq", 83),
    "Are sports facilities available on campus?": ("faq", 82),
    "Is there a gym on campus?": ("faq", 81),
    "What medical facilities are available?": ("faq", 80),
    "Does IIT Jammu provide transport facilities?": ("faq", 79),
    "Is there a library on campus?": ("faq", 78),
    "What are the library timings?": ("faq", 77),
    "How can I access online journals and research papers?": ("faq", 76),
    "What is the anti-ragging policy?": ("faq", 75),
    "Is there a student counseling facility available?": ("faq", 74),
    "What are the hostel rules and regulations?": ("faq", 73),
    "How can I apply for hostel leave?": ("faq", 72),

    # ── General Info ──────────────────────────────────────────────
    "Who is the director of IIT Jammu?": ("faq", 96),
    "Who is the Head of Department for CSE?": ("faq", 95),
    "Who is the HoD of Electrical Engineering?": ("faq", 94),
    "Who is the Head of Mechanical Engineering?": ("faq", 93),
    "Where is IIT Jammu located?": ("faq", 92),
    "What is the official contact email of IIT Jammu?": ("faq", 88),
    "How can I contact the admissions office?": ("faq", 87),
    "Which departments are available at IIT Jammu?": ("faq", 86),
    "What are the specializations in Computer Science?": ("faq", 85),
    "Is Artificial Intelligence available as a specialization?": ("faq", 84),
    "Does IIT Jammu offer Data Science courses?": ("faq", 83),
    "What research areas are available in CSE?": ("faq", 82),
    "Why should I choose Indian Institute of Technology Jammu for higher studies?": ("faq", 80),
    "How can I stay updated with IIT Jammu announcements?": ("faq", 79),
    "Does IIT Jammu provide alumni networking opportunities?": ("faq", 78),
    "What career opportunities are available after graduation?": ("faq", 77),

    # ── Research Topics ───────────────────────────────────────────
    "Machine Learning Research": ("research", 82),
    "Artificial Intelligence Research": ("research", 80),
    "Computer Vision Research": ("research", 78),
    "Cybersecurity Research": ("research", 76),
    "Robotics Research": ("research", 75),
    "Data Science Research": ("research", 74),
    "Signal Processing Research": ("research", 73),
    "Natural Language Processing": ("research", 72),
    "Deep Learning": ("research", 72),
    "Reinforcement Learning": ("research", 70),
    "Edge Computing": ("research", 68),
    "IoT and Smart City Research": ("research", 68),
    "Drone Technology Research": ("research", 70),
    "Autonomous Systems Research": ("research", 70),
    "Multimodal AI Research": ("research", 68),
    "Explainable AI Research": ("research", 67),
    "Medical Imaging Research": ("research", 67),
    "Speech Recognition Research": ("research", 66),
    "Embedded AI Research": ("research", 66),
    "Autonomous Vehicles Research": ("research", 65),
    "High Performance Computing": ("research", 65),
    "Generative AI Research": ("research", 68),
    "Quantum Computing Research": ("research", 62),
    "Digital Forensics Research": ("research", 64),
    "Smart Materials Research": ("research", 63),
    "Energy Storage Research": ("research", 63),

    # ── Research Questions ────────────────────────────────────────
    "Which professors work on Machine Learning?": ("research", 87),
    "Which faculty members work on Artificial Intelligence?": ("research", 86),
    "Which professors work on robotics?": ("research", 85),
    "Which faculty members work on cybersecurity?": ("research", 84),
    "Which professors are working on NLP research?": ("research", 83),
    "Which professors are working on computer vision?": ("research", 82),
    "Which professors are accepting PhD students?": ("research", 81),
    "How can I contact a professor for research collaboration?": ("research", 80),
    "Which professor should I contact for AI research?": ("research", 79),
    "What labs are available in the CSE department?": ("research", 78),
    "Which labs are working on drone technology?": ("research", 77),
    "Are there any ongoing AI projects at IIT Jammu?": ("research", 76),
    "What funded research projects are currently active?": ("research", 75),
    "Does IIT Jammu collaborate with industries for research?": ("research", 74),
    "What are the latest research publications from IIT Jammu?": ("research", 73),
    "Are there any startup incubation facilities?": ("faq", 72),
    "Does IIT Jammu support entrepreneurship?": ("faq", 71),
    "Which labs at IIT Jammu are actively working on funded AI research projects?": ("research", 78),
    "How can I join an ongoing ML project as an undergraduate student?": ("research", 77),
    "What are the current research trends at IIT Jammu in Artificial Intelligence?": ("research", 76),
    "Which professors have published papers recently in NLP or Generative AI?": ("research", 75),
    "I want to work on drone-based surveillance systems. Which research groups should I contact?": ("research", 74),
    "Which professors are currently accepting PhD scholars in AI-related fields?": ("research", 73),
    "How does IIT Jammu support startup incubation for AI-based projects?": ("faq", 72),
    "Which department is best for pursuing research in autonomous systems?": ("research", 71),
    "Which faculty members collaborate with industries like DRDO, ISRO, or private AI companies?": ("research", 70),
    "Which labs provide GPU or high-performance computing facilities for deep learning research?": ("research", 69),
    "Which research areas at IIT Jammu are expected to grow in the next 5 years?": ("research", 68),
    "How can I prepare for research internships under IIT Jammu professors?": ("research", 67),
    "How does IIT Jammu support students interested in entrepreneurship and startups?": ("faq", 65),
    "Which professors are working on embedded AI and edge computing systems?": ("research", 64),
    "Does IIT Jammu have collaborations with foreign universities for research exchange?": ("faq", 63),
    "Which professors are working on IoT and smart city technologies?": ("research", 62),
    "How can I contribute to published research papers as a student?": ("research", 61),
    "Which department offers the strongest curriculum for Machine Learning specialization?": ("research", 60),
    "Which professors work on ethical AI or explainable AI?": ("research", 59),
    "Which faculty members have received major research grants recently?": ("research", 58),
    "Which professors are working on autonomous vehicles or smart transportation systems?": ("research", 57),
    "Which faculty members specialize in data science and big data analytics?": ("research", 56),
    "Which professors are involved in cybersecurity and digital forensics research?": ("research", 55),
    "Which labs are working on real-world industrial AI applications?": ("research", 55),
    "How can I identify which professor aligns best with my research interests?": ("research", 54),
    "Which current research areas have the highest funding opportunities at IIT Jammu?": ("research", 54),
    "I am interested in robotics and computer vision. Which department and faculty should I approach?": ("research", 82),
    "Which faculty members are most suitable for research in sound detection and signal processing?": ("research", 80),
    "I want to pursue research in cybersecurity but my background is Mechanical Engineering. Am I eligible?": ("faq", 79),
    "I have completed BTech in Electronics Engineering. Can I apply for direct PhD in AI or Machine Learning at Indian Institute of Technology Jammu?": ("faq", 85),

    # ── Student Life & Clubs ──────────────────────────────────────
    "What clubs are available for students?": ("faq", 72),
    "Is there a coding club at IIT Jammu?": ("faq", 71),
    "What technical festivals are organized?": ("faq", 70),
    "When is the annual tech fest conducted?": ("faq", 69),
    "Are hackathons organized on campus?": ("faq", 68),
    "What are the latest events happening at IIT Jammu?": ("faq", 67),
    "Are there any upcoming AI workshops?": ("faq", 66),
    "How can I participate in seminars and conferences?": ("faq", 65),
    "Are there any internships currently open?": ("faq", 64),
    "What are the latest notices released by IIT Jammu?": ("faq", 63),
    "Where can I find the latest circulars?": ("faq", 62),
    "How can I report technical issues on campus?": ("faq", 61),
    "What facilities are available for students building robotics or AI hardware projects?": ("faq", 60),
    "I want to build a startup in AI during college. What support systems are available?": ("faq", 60),
    "How does IIT Jammu support students preparing for higher studies abroad?": ("faq", 60),
    "What research opportunities are available for first-year BTech students?": ("research", 62),
    "Which labs are open for undergraduate participation in research?": ("research", 62),
    "Which department is best for Machine Learning research?": ("research", 75),
    "I completed BTech in Electronics. Can I apply for AI research?": ("research", 74),
    "I want to pursue robotics research. Which professor should I contact?": ("research", 73),
    "Can I pursue AI research without strong coding experience initially?": ("research", 65),
    "Are there any interdisciplinary programs combining AI with Mechanical or Civil Engineering?": ("faq", 65),
    "What are the most active student technical clubs related to AI and robotics?": ("faq", 63),
    "Which faculty members focus on applied AI rather than theoretical AI?": ("research", 63),
    "How does IIT Jammu evaluate students for research assistantships?": ("research", 62),
    "Are there any live industry-sponsored projects currently running on campus?": ("research", 62),
    "What are the best pathways to enter AI research after completing BTech?": ("research", 62),
    "Can students from IIT Jammu participate in international hackathons and competitions?": ("faq", 61),
    "Can students from non-CS backgrounds pursue MTech in CSE-related specializations?": ("faq", 61),
    "Can I pursue AI research without strong coding experience initially?": ("research", 60),
    "What is the process for converting from MTech to PhD at IIT Jammu?": ("faq", 72),
    "I want to pursue higher studies abroad after MTech. Which research groups would strengthen my profile?": ("research", 63),
    "How can I join a research project as an undergraduate student?": ("research", 68),
}


class AutocompleteService:
    """
    Smart autocomplete with trie + fuzzy matching.

    Provides:
      1. Prefix-based suggestions (fast, from trie)
      2. Fuzzy matching (handles typos)
      3. Category-aware results
      4. Dynamically updated from database content
    """

    def __init__(self):
        self._trie = Trie()
        self._all_terms: List[Dict] = []  # For fuzzy matching
        self._is_built = False
        self._build_static_index()

    def _build_static_index(self):
        """Build trie from static seed suggestions."""
        for text, (category, freq) in STATIC_SUGGESTIONS.items():
            self._trie.insert(text, frequency=freq, category=category)
            self._all_terms.append({"text": text, "category": category, "frequency": freq})

    def build_from_database(self, chroma_store=None):
        """
        Rebuild autocomplete index from ChromaDB metadata.

        Extracts:
          - Faculty names from document titles
          - Department names
          - Program names
          - Unique topics
        """
        if not chroma_store:
            self._is_built = True
            return

        try:
            metadata_list = chroma_store.get_all_metadata()
            seen_titles = set()

            for meta in metadata_list:
                if not meta:
                    continue

                title = meta.get("title", "").strip()
                clean_title = re.sub(r"^(Auto FAQ|Curated FAQ|FAQ)\s*\d*\s*:\s*", "", title, flags=re.I).strip()
                if clean_title and len(clean_title) > 3 and clean_title not in seen_titles:
                    seen_titles.add(clean_title)

                    # Extract faculty names
                    if any(kw in clean_title.lower() for kw in ["prof", "dr.", "professor"]):
                        name = re.sub(r"^(Prof\.?|Dr\.?|Professor)\s+", "", clean_title, flags=re.I).strip()
                        if name and len(name) > 3:
                            self._trie.insert(name, frequency=50, category="faculty")
                            self._all_terms.append({"text": name, "category": "faculty", "frequency": 50})

                    # Extract by doc_type
                    doc_type = meta.get("doc_type", "")
                    if doc_type and doc_type != "General":
                        category = doc_type.lower()
                        if len(clean_title) <= 80:
                            self._trie.insert(clean_title, frequency=30, category=category)
                            self._all_terms.append({"text": clean_title, "category": category, "frequency": 30})

            logger.info(f"Autocomplete index built: {self._trie.count()} entries")
            self._is_built = True

        except Exception as e:
            logger.error(f"Failed to build autocomplete index: {e}")
            self._is_built = True

    def search(self, prefix: str, limit: int = 8) -> List[Dict]:
        """
        Get autocomplete suggestions for a prefix.

        Combines:
          1. Trie prefix matches (fast, exact prefix)
          2. Fuzzy matches (handles typos, Levenshtein ≤ 2)

        Returns list of {text, category, score} dicts.
        """
        if not prefix or len(prefix.strip()) < 1:
            return []

        prefix = prefix.strip()
        results = []
        seen = set()

        # 1. Trie prefix search
        trie_results = self._trie.search_prefix(prefix, limit=limit)
        for text, category, freq in trie_results:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                results.append({
                    "text": text,
                    "category": category,
                    "score": freq / 100.0,
                })

        # 2. Fuzzy matching for remaining slots
        if len(results) < limit:
            fuzzy = self._fuzzy_search(prefix, limit=limit - len(results), seen=seen)
            results.extend(fuzzy)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _fuzzy_search(
        self, query: str, limit: int = 5, seen: set = None
    ) -> List[Dict]:
        """Simple fuzzy matching using substring + edit distance."""
        if seen is None:
            seen = set()

        query_lower = query.lower()
        results = []

        for term in self._all_terms:
            text = term["text"]
            text_lower = text.lower()
            key = text_lower

            if key in seen:
                continue

            # Substring match
            if query_lower in text_lower:
                score = term["frequency"] / 100.0 * 0.8
                results.append({"text": text, "category": term["category"], "score": score})
                seen.add(key)
                continue

            # Word-start match ("b" matches "BTech", "Badrinarayan")
            words = text_lower.split()
            if any(w.startswith(query_lower) for w in words):
                score = term["frequency"] / 100.0 * 0.6
                results.append({"text": text, "category": term["category"], "score": score})
                seen.add(key)
                continue

            # Simple edit distance for short queries (typo tolerance)
            if len(query) >= 3 and len(query) <= 15:
                dist = self._levenshtein(query_lower, text_lower[:len(query_lower) + 2])
                if dist <= 2:
                    score = term["frequency"] / 100.0 * (1.0 - dist * 0.2)
                    results.append({"text": text, "category": term["category"], "score": score})
                    seen.add(key)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return AutocompleteService._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


# ── Singleton ─────────────────────────────────────────────────────
_service: Optional[AutocompleteService] = None


def get_autocomplete_service() -> AutocompleteService:
    """Get or create the singleton AutocompleteService."""
    global _service
    if _service is None:
        _service = AutocompleteService()
    return _service
