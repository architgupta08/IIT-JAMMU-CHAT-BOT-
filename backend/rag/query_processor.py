"""
rag/query_processor.py — Smart Query Processing
=================================================
Handles query understanding, intent classification, NER,
abbreviation expansion, and fuzzy matching.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Abbreviation expansion map ─────────────────────────────────────
ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "dl": "deep learning",
    "dnn": "deep neural network",
    "cse": "Computer Science and Engineering",
    "ee": "Electrical Engineering",
    "me": "Mechanical Engineering",
    "ce": "Civil Engineering",
    "che": "Chemical Engineering",
    "hss": "Humanities and Social Sciences",
    "ece": "Electronics and Communication Engineering",
    "btech": "B.Tech",
    "mtech": "M.Tech",
    "phd": "Ph.D",
    "msc": "M.Sc",
    "ug": "undergraduate",
    "pg": "postgraduate",
    "hod": "Head of Department",
    "tnp": "Training and Placement",
    "mcm": "Merit-cum-Means scholarship",
    "pmrf": "Prime Minister's Research Fellowship",
    "cpi": "Cumulative Performance Index",
    "sgpa": "Semester Grade Point Average",
    "jrf": "Junior Research Fellow",
    "srf": "Senior Research Fellow",
    "ra": "Research Assistant",
    "hpc": "High Performance Computing",
}

# ── Intent patterns ────────────────────────────────────────────────
INTENT_PATTERNS = {
    "faculty_info": [
        r"who is\s+(.+?)[\?!.]?$",
        r"tell me about\s+(professor|prof|dr)\s+(.+?)[\?!.]?$",
        r"(?:which|what)\s+professors?\s+(?:work|research)\s+(?:on|in)\s+(.+?)[\?!.]?$",
        r"faculty\s+(?:in|of|from)\s+(.+?)$",
        r"show\s+faculty\s+(?:from|in|of)\s+(.+?)$",
        # Publications, research, email, designation queries about a person
        r"(?:what\s+(?:is|are)\s+the\s+)?(?:publications?|research\s+interests?|research\s+areas?)\s+(?:of|for|by)\s+(.+?)[\?!.]?$",
        r"(?:what\s+is\s+the\s+)?(?:email|contact|designation|profile)\s+(?:of|for)\s+(.+?)[\?!.]?$",
    ],
    "admission": [
        r"(?:how to|how can i)\s+(?:apply|get admission|join)\s+(?:for|to|in)?\s*(.+?)[\?!.]?$",
        r"admission\s+(?:process|procedure|criteria|eligibility)\s+(?:for)?\s*(.+?)[\?!.]?$",
        r"(?:what is|what are)\s+(?:the)?\s*(?:cutoff|cut-off|eligibility)\s+(?:for)?\s*(.+?)[\?!.]?$",
    ],
    "placement": [
        r"(?:placement|package|salary|ctc|lpa|recruit)",
        r"which\s+companies?\s+(?:visit|come|recruit)",
        r"(?:highest|average|median)\s+(?:package|salary|ctc)",
    ],
    "research": [
        r"(?:research|project|lab)\s+(?:in|on|about)\s+(.+?)[\?!.]?$",
        r"which\s+(?:faculty|professor|prof)\s+(?:works?|research)\s+(?:on|in)\s+(.+?)[\?!.]?$",
        r"(?:who|which|what)\s+(?:is|are|faculty|professor|prof)?\s*(?:working|works?|researching|researches?)\s+(?:on|in)\s+(.+?)[\?!.]?$",
    ],
    "campus": [
        r"(?:hostel|mess|library|medical|sports|gym|wifi|transport|campus|canteen|cafeteria|facility|facilities)",
        r"(?:fees?|charges?)\s+(?:for)?\s*(?:hostel|mess)",
    ],
    "course": [
        r"(?:course|curriculum|syllabus|subject|elective)\s+(?:for|in|of)\s*(.+?)[\?!.]?$",
        r"(?:what|which)\s+(?:courses?|subjects?)\s+(?:are)?\s*(?:offered|taught|available)",
    ],
    "general": [
        r"(?:where|location|address|contact|email|phone|director|about)",
        r"(?:vision|mission|history|established)",
    ],
}


@dataclass
class QueryIntent:
    """Processed query with intent and extracted entities."""
    original_query: str
    corrected_query: str
    processed_query: str
    intent: str
    entities: Dict[str, str]
    expanded_terms: List[str]
    is_followup: bool = False


# ── Canonical Domain Targets & Values (for fuzzy typo correction) ─────
CANONICAL_TARGETS = {
    "cutoff": ["cutoff", "cutoffs", "cut-off", "cut-offs", "cut off", "cut offs"],
    "admission": ["admission", "admissions"],
    "placement": ["placement", "placements"],
    "statistics": ["statistics", "stats"],
    "package": ["package", "packages"],
    "salary": ["salary", "salaries"],
    "record": ["record", "records"],
    "hostel": ["hostel", "hostels", "hostal", "hostals"],
    "mess": ["mess", "messes", "mes"],
    "fee": ["fee", "fees"],
    "tuition": ["tuition"],
    "syllabus": ["syllabus"],
    "curriculum": ["curriculum"],
    "department": ["department", "departments"],
    "faculty": ["faculty"],
    "professor": ["professor", "professors"],
    "research": ["research"],
    "publications": ["publications"],
    "interest": ["interest", "interests"],
    "eligibility": ["eligibility"],
    "canteen": ["canteen", "canteens", "cafeteria", "cafeterias"],
    "block": ["block", "blocks", "building", "buildings"],
    "facility": ["facility", "facilities"],
    "paloura": ["paloura", "ploura", "palora", "palour"],
    "campus": ["campus", "camopus", "campuses", "campuse"]
}

ALL_CANONICAL_VALUES = set(CANONICAL_TARGETS.keys())

# Protect all abbreviations (e.g. cse, ug, pg, btech, mtech, phd)
ALL_CANONICAL_VALUES.update(ABBREVIATIONS.keys())
# Protect specific key terms in the IIT Jammu domain
ALL_CANONICAL_VALUES.update([
    "jee", "iit", "jammu", "gate", "gates", "josaa", "course", "courses", 
    "program", "programs", "admission", "admissions", "msc", "mba", "phd", 
    "academic", "academics", "roll", "rolls", "rank", "ranks", "score", "scores", 
    "seat", "seats", "matrix", "canteen", "canteens", "cafeteria", "cafeterias",
    "hostel", "hostels", "accommodation", "lodging", "boarding", "canary", "braeg",
    "fulgar", "dedhar", "egret", "mess", "messes", "dining", "annapurna",
    "block", "blocks", "building", "buildings", "pushkar", "mansar", "trikuta",
    "facility", "facilities", "amenity", "amenities", "working", "works",
    "learning", "deep", "drones", "quantum", "computing", "wireless", "communication",
    "audio", "speech", "processing", "iot", "ccd", "nescafe",
    "paloura", "campus", "campuses", "rate", "rates", "date", "dates", "state", "states", "free"
])


# ── Common Typos Map ───────────────────────────────────────────────
COMMON_TYPOS = {
    "admissin": "admission",
    "admision": "admission",
    "admisson": "admission",
    "amdisson": "admission",
    "btec": "btech",
    "mtec": "mtech",
    "facuty": "faculty",
    "faculties": "faculty",
    "plaement": "placement",
    "placment": "placement",
    "rearch": "research",
    "reserch": "research",
    "reseach": "research",
    "resdeasrch": "research",
    "hostal": "hostel",
    "libary": "library",
    "sylabuss": "syllabus",
    "syalbus": "syllabus",
    "sylabus": "syllabus",
    "fees": "fee", # standardize to singular for better matching
    "hied": "head",
    "depaetement": "department",
    "od": "of",
    "ot": "to",
    "priocedure": "procedure",
    "engneering": "engineering",
    "engg": "engineering",
    "ijtrsdt": "interest",
    "publicastionjs": "publications",
    "publucations": "publications",
    "proff": "prof",
}

# ── Known Faculty Names (for fuzzy matching) ──────────────────────
FACULTY_NAMES = [
    "Abhay Sharma",
    "Abhishek Kumar",
    "Ajay Singh",
    "Akash Subhash Awale",
    "Alok Kumar Saxena",
    "Ambika Prasad Shah",
    "Angshuman Kapil",
    "Ankit Dubey",
    "Ankit Kathuria",
    "Ankur Bansal",
    "Anup Shukla",
    "Anurag Misra",
    "Archana Rajput",
    "Aroof Aimen",
    "Arpita Paul",
    "Arun Kumar Verma",
    "Arvind Kumar Rajput",
    "Ashutosh Bijalwan",
    "Badri Narayan Subudhi",
    "Badrinarayan Subuddhi",
    "Biswanath Chakraborty",
    "Chaitanya Indukuri",
    "Chandan Yadav",
    "Chembolu Vinay",
    "Deepak Yadav",
    "Dhanendra Kumar",
    "Divyesh Varade",
    "Gaurav Varshney",
    "Goutam Dutta",
    "Harkeerat Kaur",
    "Ibhanchand Rath",
    "Kankat Ghosh",
    "Karan Nathwani",
    "Manoj Singh Gaur",
    "Mir Faizan Ul Haq",
    "Mrinmoy Bhattacharjee",
    "Nalin Kumar Sharma",
    "Navneet Kumar",
    "Nitin Joshi",
    "Nityananda Sahu",
    "Padmini Singh",
    "Parveen Kumar",
    "Pervaiz Fathima Khatoon M",
    "Pothukuchi Harish",
    "Prasun Halder",
    "Pratik Kumar",
    "Priyanka Mishra",
    "Priyatosh Jena",
    "Prof. Manoj Singh Gaur",
    "Rajendra Kumar Varma",
    "Rajkumar V",
    "Ravikant Saini",
    "Rimen Jamatia",
    "Riya Bhowmik",
    "Rohit Buddhiram Chaurasiya",
    "Rohit Chaurasiya",
    "Roshan Udaram Patil",
    "S. R. K. Chaitanya Indukuri",
    "Sahil Kalra",
    "Samaresh Bera",
    "Sameer Kumar Sarma Pachalla",
    "Samrat Rao",
    "Sanat Kumar Tiwari",
    "Sarada Prasad Gochhayat",
    "Sarah Mariam Abraham",
    "Satya Sekhar Bhogilla",
    "Satyadev Ahlawat",
    "Saurabh Biswas",
    "Sayantan Mukherjee",
    "Shaifu Gupta",
    "Shanmugadas K.P",
    "Shikha Baghel",
    "Shirsendu Ghosh",
    "Shiva S",
    "Sidharth Maheshwari",
    "Sivakumar G",
    "Soma S Dhavala",
    "Srishti Singh",
    "Subhasis Bhattacharjee",
    "Sudhakar Modem",
    "Suman Banerjee",
    "Sumit Kumar Pandey",
    "Sumit Pandey",
    "Surendra Beniwal",
    "Ved Prakash Ranjan",
    "Venkata Sathish Akella",
    "Vijay Kumar Pal",
    "Vinit Jakhetiya",
    "Yamuna Prasad",
    "Aditi Gupta",
    "Kushmanda Saurav",
    "Ashutosh Yadav",
    "Aditya Shankar Sandupatla",
    "Ankit Tyagi",
    "Dharitri Rath",
    "Dnyaneshwar Bhawangirkar",
    "Gaurav Ashok Bhaduri",
    "Krishna Mohan Gupta",
    "Ravi Kumar Arun",
    "Durai Prabhakaran R T",
    "Sampat Rao",
    "Shanmugadas K P",
    "Ajay Gautam",
    "Avinash Raulo",
    "Devi Lal",
    "Kancharla Hari Krishna",
    "Nitesh Kumar",
    "Rani Rohini",
    "Srinivasan N",
    "Srishilan C",
    "Sujaya Chakraborty",
    "Suman Sarkar",
    "Sunil Kumar Kashyap",
    "Shantanu Vijay Madge",
    "Uma Satya Ranjan"
]

class QueryProcessor:
    """
    Smart query processing pipeline:
    1. Correct common typos and fuzzy-match domain terms using Levenshtein distance
    2. Fuzzy-match faculty names using segment-based analysis
    3. Expand abbreviations (ml → machine learning)
    4. Extract named entities (professor names, departments)
    5. Classify intent (admission, faculty, placement, etc.)
    6. Generate expanded search query
    """

    def __init__(self):
        # Build faculty name variants map dynamically
        self.faculty_variants = {}
        ambiguous_variants = set()

        for name in FACULTY_NAMES:
            name_lower = name.lower()
            parts = name_lower.split()
            # Remove title prefixes from parts for clean lookup keys
            parts = [p for p in parts if p not in {"prof.", "prof", "dr.", "dr", "professor"}]
            if not parts:
                continue
                
            full_clean = " ".join(parts)
            variants = [full_clean]
            
            # Individual name parts (first name, last name, etc.)
            for part in parts:
                if len(part) > 2:  # Ignore initials
                    variants.append(part)
                    
            # 2-word combinations if name has 3 or more parts
            if len(parts) >= 3:
                for i in range(len(parts) - 1):
                    variants.append(f"{parts[i]} {parts[i+1]}")
                variants.append(f"{parts[0]} {parts[-1]}")
                
            # Populate faculty variants map, flagging ambiguous (shared) parts
            for v in variants:
                if v in self.faculty_variants:
                    if self.faculty_variants[v] != name:
                        ambiguous_variants.add(v)
                else:
                    self.faculty_variants[v] = name
                    
        # Remove ambiguous variants to avoid incorrect mapping (e.g. "Ankit" matches two professors)
        for v in ambiguous_variants:
            self.faculty_variants.pop(v, None)

    def process(self, query: str) -> QueryIntent:
        """Process a query through the full pipeline."""
        original = query.strip()
        corrected = self._correct_typos(original)
        corrected = self._fuzzy_match_faculty(corrected)
        processed = self._expand_abbreviations(corrected)
        intent = self._classify_intent(processed)
        entities = self._extract_entities(processed)
        expanded_terms = self._generate_expanded_terms(processed, intent, entities)

        return QueryIntent(
            original_query=original,
            corrected_query=corrected,
            processed_query=processed,
            intent=intent,
            entities=entities,
            expanded_terms=expanded_terms,
        )

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    def _fuzzy_match_faculty(self, query: str) -> str:
        """
        Identify misspelled/abbreviated faculty names and replace them with canonical names.
        Uses a segment-based approach to avoid consuming verbs/stopwords.
        """
        import difflib

        title_prefixes = {"dr", "dr.", "prof", "prof.", "professor"}
        name_ignore_words = {
            "who", "is", "about", "tell", "me", "what", "are", "the", "of", "for", 
            "email", "publications", "research", "interests", "qualification", 
            "designation", "profile", "contact", "and", "hod", "in", "at", "on", 
            "with", "by", "from", "to", "how", "many", "does", "do", "did", "has", 
            "have", "had", "can", "could", "would", "should", "will", "shall", "a", 
            "an", "any", "there", "their", "here", "its", "it", "they", "them", 
            "we", "us", "our", "you", "your", "under", "work", "working", "researching",
            "teaching", "professor", "professors", "faculty", "faculties", "member", "members",
            "he", "she", "his", "her", "i", "me", "my", "myself", "campus", "iit", "jammu",
            "hostel", "mess", "canteen", "block", "blocks", "building", "buildings",
            "trikuta", "pushkar", "mansar", "facility", "facilities",
            "where", "room", "rooms", "office", "offices", "location", "locations", "sits", 
            "sit", "floor", "floors", "cabin", "cabins", "number", "numbers", "find", "locate", 
            "show", "search", "get", "address", "details", "info", "information", "phone", 
            "contacts", "telephone", "mobile", "seat", "seating", "siting", "whois", "whatis", 
            "whereis"
        }

        words = query.split()
        segments = []
        current_segment = []

        # Group contiguous words that are potential name components
        for idx, word in enumerate(words):
            clean_word = re.sub(r"[.,!?']", "", word).lower()
            if clean_word in title_prefixes:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            elif clean_word in name_ignore_words or clean_word in ALL_CANONICAL_VALUES or len(clean_word) <= 1:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append((idx, word))
        if current_segment:
            segments.append(current_segment)

        keys = list(self.faculty_variants.keys())

        # Process segments from right to left to prevent index shift issues upon replacement
        for segment in reversed(segments):
            idx_start = segment[0][0]
            idx_end = segment[-1][0]
            segment_str = " ".join([re.sub(r"[.,!?']", "", w).lower() for _, w in segment])
            
            # Look up closest variant match
            close = difflib.get_close_matches(segment_str, keys, n=1, cutoff=0.72)
            if close:
                canonical_name = self.faculty_variants[close[0]]
                last_orig_word = words[idx_end]
                punct_match = re.search(r"([.,!?]+)$", last_orig_word)
                punct = punct_match.group(1) if punct_match else ""
                
                # Replace the entire segment span with the canonical name
                words[idx_start:idx_end+1] = [canonical_name + punct]
                logger.info(f"Faculty fuzzy match: '{segment_str}' -> '{canonical_name}'")

        return " ".join(words)

    def _correct_typos(self, query: str) -> str:
        """Correct very common typos and split compound words before processing."""
        # Standardize conversational queries asking if data was inserted/added/loaded/present
        query = re.sub(r"\bdid\s+(?:u|you)\s+(?:insert|insertyed|add|update|load|put|put-in)\b", "what is", query, flags=re.IGNORECASE)
        query = re.sub(r"\bdo\s+(?:u|you)\s+(?:have|possess|know|knows|have\s+inserted)\b", "what is", query, flags=re.IGNORECASE)
        
        # Clean prefix typos
        query = re.sub(r"\bwhati\b", "what is", query, flags=re.IGNORECASE)
        query = re.sub(r"\bsthe\b", "the", query, flags=re.IGNORECASE)
        query = re.sub(r"\bwhatis\b", "what is", query, flags=re.IGNORECASE)
        query = re.sub(r"\bwhatare\b", "what are", query, flags=re.IGNORECASE)
        query = re.sub(r"\bwhois\b", "who is", query, flags=re.IGNORECASE)
        query = re.sub(r"\bwhereis\b", "where is", query, flags=re.IGNORECASE)
        query = re.sub(r"\bhowto\b", "how to", query, flags=re.IGNORECASE)
        query = re.sub(r"\bhowcan\b", "how can", query, flags=re.IGNORECASE)
        query = re.sub(r"\bhowmany\b", "how many", query, flags=re.IGNORECASE)
        
        # Split compound words like csehod, hodcivil, mtechadmission, etc.
        query = re.sub(
            r"\b(cse|civil|ee|me|che|ece|btech|mtech|phd|msc|ug|pg)(hod|admission|admissions|cutoff|cutoffs|admisson|admissin|amdisson)\b",
            r"\1 \2",
            query,
            flags=re.IGNORECASE
        )
        query = re.sub(
            r"\b(hod|admission|admissions|cutoff|cutoffs|admisson|admissin|amdisson)(cse|civil|ee|me|che|ece|btech|mtech|phd|msc|ug|pg)\b",
            r"\1 \2",
            query,
            flags=re.IGNORECASE
        )

        words = query.split()
        corrected = []
        for word in words:
            # Strip punctuation but keep it to append back
            match = re.match(r"^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$", word)
            if match:
                prefix, core, suffix = match.groups()
                if not core:
                    corrected.append(word)
                    continue
                lower_core = core.lower()
                
                # 1. Exact match in COMMON_TYPOS
                if lower_core in COMMON_TYPOS:
                    corrected.append(f"{prefix}{COMMON_TYPOS[lower_core]}{suffix}")
                # 2. Exact match in canonical values
                elif lower_core in ALL_CANONICAL_VALUES:
                    corrected.append(word)
                # 3. Fuzzy match against canonical targets using Levenshtein distance
                else:
                    best_target = None
                    min_dist = 999
                    for canonical, variations in CANONICAL_TARGETS.items():
                        for val in variations:
                            if len(val) <= 4:
                                limit = 1
                            elif len(val) <= 7:
                                limit = 2
                            else:
                                limit = 3
                                
                            # Safe edit distance caps based on input word length
                            if len(lower_core) <= 4 and limit > 1:
                                limit = 1
                            elif len(lower_core) <= 6 and limit > 2:
                                limit = 2
                                
                            dist = self._edit_distance(lower_core, val)
                            if dist <= limit and dist < min_dist:
                                min_dist = dist
                                best_target = canonical
                                
                    if best_target:
                        corrected.append(f"{prefix}{best_target}{suffix}")
                    else:
                        corrected.append(word)
            else:
                corrected.append(word)
        query = " ".join(corrected)

        # Standardize colloquial department names & degrees
        query = re.sub(r"\bcomputer\s+engineering\b", "Computer Science and Engineering", query, flags=re.IGNORECASE)
        query = re.sub(r"\bcomputer\s+science\b(?!\s+and\s+engineering)", "Computer Science and Engineering", query, flags=re.IGNORECASE)
        query = re.sub(r"\badmissions\b", "admission", query, flags=re.IGNORECASE)
        
        # Standardize research interests synonyms
        query = re.sub(r"\barea(s)?\s+of\s+interest(s)?\b", "research interests", query, flags=re.IGNORECASE)
        query = re.sub(r"\bresearch\s+area(s)?\b", "research interests", query, flags=re.IGNORECASE)
        query = re.sub(r"\bresearch\s+interest(s)?\b", "research interests", query, flags=re.IGNORECASE)
        return query

    def _expand_abbreviations(self, query: str) -> str:
        """Expand known abbreviations in the query."""
        words = query.split()
        expanded = []
        for word in words:
            lower = word.lower().strip(".,!?")
            if lower in ABBREVIATIONS and len(lower) <= 4:
                expanded.append(f"{word} ({ABBREVIATIONS[lower]})")
            else:
                expanded.append(word)
        return " ".join(expanded)

    def _classify_intent(self, query: str) -> str:
        """Classify query intent based on pattern matching."""
        q = query.lower()
        
        # Override intent to general for job-related queries to prevent incorrect admission classification
        job_keywords = ["job", "vacancy", "vacancies", "career", "careers", "recruitment", "employment", "hiring", "project associate", "project staff", "research associate", "postdoc", "jrf", "fellowship"]
        if any(w in q for w in job_keywords):
            return "general"
            
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q, re.I):
                    return intent
        return "general"

    def _extract_entities(self, query: str) -> Dict[str, str]:
        """Extract named entities from the query."""
        entities = {}

        # Direct lookup of known faculty names in the query text
        q_lower = query.lower()
        for name in FACULTY_NAMES:
            if name.lower() in q_lower:
                entities["person"] = name
                break

        invalid_name_words = {
            "working", "works", "researching", "researches", "teaching", "teaches",
            "doing", "in", "on", "at", "about", "project", "lab", "with", "using", 
            "for", "course", "subject", "placement", "hostel", "mess", "canteen", "fees"
        }

        # Extract person names ("who is X", "about Professor X")
        person_match = re.search(
            r"(?:who is|about|tell me about)\s+(?:professor|prof\.?|dr\.?)?\s*(.+?)[\?!.]?$",
            query, re.I
        )
        if person_match:
            name = person_match.group(1).strip()
            name_words = set(name.lower().split())
            if name and len(name.split()) <= 5 and not (name_words & invalid_name_words):
                entities["person"] = name

        # Also extract person from "what are the publications of X?" style queries
        if "person" not in entities:
            attr_person_match = re.search(
                r"(?:what\s+(?:is|are)\s+the\s+)?(?:publications?|research\s+interests?|research\s+areas?|"
                r"email|contact|designation|qualification|profile|department)\s+"
                r"(?:of|for|by)\s+(?:professor|prof\.?|dr\.?)?[\s]*(.+?)[\?!.]?$",
                query, re.I
            )
            if attr_person_match:
                name = attr_person_match.group(1).strip()
                name_words = set(name.lower().split())
                if name and 1 <= len(name.split()) <= 5 and not (name_words & invalid_name_words):
                    entities["person"] = name

        # Also extract person from "X's publications" style queries
        if "person" not in entities:
            possessive_match = re.search(
                r"(?:professor|prof\.?|dr\.?)?[\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})(?:'s|'s)\s+"
                r"(?:publications?|research|email|contact|designation|profile)",
                query
            )
            if possessive_match:
                name = possessive_match.group(1).strip()
                name_words = set(name.lower().split())
                if name and 1 <= len(name.split()) <= 5 and not (name_words & invalid_name_words):
                    entities["person"] = name

        # Extract student/scholar names from supervisor/scholar queries
        # 1. "supervisor of/for <name>"
        sup_match = re.search(
            r"\bsupervisor\s+(?:of|for)\s+(?:the\s+phd\s+student\s+named\s+|the\s+student\s+named\s+)?([A-Za-z\s]+)",
            query, re.I
        )
        if sup_match:
            name = sup_match.group(1).strip()
            name_words = [w for w in name.split() if w.lower() not in {"the", "a", "an", "phd", "student", "scholar"}]
            if name_words and len(name_words) <= 4:
                entities["person"] = " ".join(name_words)

        # 2. "<name>'s supervisor" or "<name> supervisor"
        poss_sup_match = re.search(
            r"\b([A-Za-z\s]+?)(?:'s)?\s+supervisor\b",
            query, re.I
        )
        if poss_sup_match:
            name = poss_sup_match.group(1).strip()
            name_words = [w for w in name.split() if w.lower() not in {"who", "is", "the", "a", "an", "phd", "student", "scholar", "for", "of"}]
            if name_words and len(name_words) <= 4:
                entities["person"] = " ".join(name_words)

        # 3. "who is supervising <name>"
        supervising_match = re.search(
            r"\bsupervising\s+([A-Za-z\s]+)",
            query, re.I
        )
        if supervising_match:
            name = supervising_match.group(1).strip()
            name_words = [w for w in name.split() if w.lower() not in {"the", "a", "an", "phd", "student", "scholar"}]
            if name_words and len(name_words) <= 4:
                entities["person"] = " ".join(name_words)

        # Extract department (robust keyword scanning mapping to canonical name)
        q_lower = query.lower()
        dept_keywords = {
            "Computer Science and Engineering": ["computer science", "computer engineering", "cse"],
            "Electrical Engineering": ["electrical engineering", "electrical", "ee"],
            "Mechanical Engineering": ["mechanical engineering", "mechanical", "me"],
            "Civil Engineering": ["civil engineering", "civil", "ce"],
            "Chemical Engineering": ["chemical engineering", "chemical", "che"],
            "Materials Engineering": ["materials engineering", "materials", "mme"],
            "Biosciences and Bioengineering": ["biosciences", "bioengineering", "bsbe"],
            "Mathematics": ["mathematics", "maths", "math"],
            "Physics": ["physics"],
            "Chemistry": ["chemistry"],
            "Humanities and Social Sciences": ["humanities", "social sciences", "hss"],
        }
        for canonical_name, aliases in dept_keywords.items():
            if any(re.search(rf"\b{alias}\b", q_lower) for alias in aliases):
                entities["department"] = canonical_name
                break

        # Extract program
        prog_match = re.search(r"\b(b\.?tech|m\.?tech|ph\.?d|m\.?sc|mba)\b", query, re.I)
        if prog_match:
            prog_raw = prog_match.group(1).lower().replace(".", "")
            prog_map = {
                "btech": "B.Tech",
                "mtech": "M.Tech",
                "phd": "Ph.D",
                "msc": "M.Sc",
                "mba": "MBA"
            }
            entities["program"] = prog_map.get(prog_raw, prog_match.group(1))

        # Extract research area
        research_match = re.search(
            r"(?:research|work|interest|working|works|researching|researches)\s+(?:in|on)\s+(.+?)[\?!.]?$",
            query, re.I
        )
        if research_match:
            entities["research_area"] = research_match.group(1).strip()

        return entities

    def _generate_expanded_terms(
        self, query: str, intent: str, entities: Dict
    ) -> List[str]:
        """Generate additional search terms based on intent and entities."""
        terms = []

        if intent == "faculty_info" and "person" in entities:
            terms.extend([
                f"{entities['person']} IIT Jammu",
                f"{entities['person']} research interests",
                f"{entities['person']} professor department",
            ])

        if intent == "admission":
            program = entities.get("program", "")
            terms.extend([
                f"IIT Jammu {program} admission process",
                f"IIT Jammu {program} eligibility criteria",
            ])

        if intent == "research" and "research_area" in entities:
            terms.append(f"IIT Jammu {entities['research_area']} research faculty lab")

        return terms

    def build_web_search_query(self, query_intent: QueryIntent) -> str:
        """Build an optimized web search query for DuckDuckGo using typo-corrected queries."""
        q = query_intent.corrected_query.lower()

        if query_intent.intent == "faculty_info" and "person" in query_intent.entities:
            return f"{query_intent.entities['person']} IIT Jammu faculty profile research"

        if query_intent.intent == "admission":
            program = query_intent.entities.get("program", "")
            return f"IIT Jammu {program} admission eligibility process"

        if query_intent.intent == "placement":
            return "IIT Jammu placement statistics companies package CTC"

        return query_intent.corrected_query

