"""
knowledge_graph.py — NetworkX Knowledge Graph for IIT Jammu Chatbot
====================================================================
A directed graph capturing entities and relationships from IIT Jammu data.

Entity types: Institution, Department, Program, Person, Facility, Fee,
              Placement, Scholarship, Contact

Design:
  - Append-only: never deletes existing nodes/edges
  - Persisted to GraphML file on disk
  - Auto-extracts entities from raw text via regex patterns
  - Queryable: find related info via graph traversal
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

KG_FILE = os.getenv("KG_FILE", "data/processed/knowledge_graph.graphml")

# ── Known entities for extraction ──────────────────────────────────
KNOWN_DEPARTMENTS = [
    "Computer Science and Engineering", "Computer Science", "CSE",
    "Electrical Engineering", "EE",
    "Mechanical Engineering", "ME",
    "Civil Engineering", "CE",
    "Chemical Engineering", "CHE",
    "Mathematics", "Mathematics and Computing", "M&C",
    "Physics", "Engineering Physics",
    "Chemistry",
    "Humanities and Social Sciences", "HSS",
    "Materials Engineering",
    "Biosciences and Bioengineering",
    "Interdisciplinary Studies", "IDP",
    "Center for Data Science", "CDS",
]

KNOWN_PROGRAMS = [
    "B.Tech", "BTech", "M.Tech", "MTech", "Ph.D", "PhD",
    "M.Sc", "MSc", "Dual Degree", "Minor", "Honours",
]

ENTITY_TYPES = {
    "Institution", "Department", "Program", "Person",
    "Facility", "Fee", "Placement", "Scholarship",
    "Contact", "ResearchLab", "Event", "General",
}

RELATIONSHIP_TYPES = {
    "HAS_DEPARTMENT", "OFFERS_PROGRAM", "LED_BY", "HAS_FEE",
    "LOCATED_AT", "HAS_FACILITY", "HAS_SCHOLARSHIP", "HAS_PLACEMENT",
    "BELONGS_TO", "PART_OF", "RELATED_TO", "HAS_CONTACT",
    "CONDUCTS_RESEARCH", "HAS_LAB",
}


def _resolve_kg_path() -> str:
    """Resolve KG file path relative to project root."""
    path = KG_FILE
    if path.startswith("../"):
        path = path[3:]

    if os.path.isabs(path):
        return path

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    from_root = os.path.join(project_root, path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(from_root), exist_ok=True)
    return from_root


class KnowledgeGraph:
    """
    NetworkX-based Knowledge Graph for IIT Jammu.

    - Nodes = entities (departments, programs, people, fees, etc.)
    - Edges = relationships (HAS_DEPARTMENT, OFFERS_PROGRAM, etc.)
    - Persisted to GraphML file
    - Append-only: never deletes nodes/edges
    """

    def __init__(self):
        import networkx as nx
        self._nx = nx
        self._path = _resolve_kg_path()
        self._graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from disk if it exists."""
        if os.path.exists(self._path):
            try:
                self._graph = self._nx.read_graphml(self._path)
                logger.info(
                    f"Knowledge Graph loaded: {self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges from {self._path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load KG from {self._path}: {e}. Starting fresh.")
                self._graph = self._nx.DiGraph()
        else:
            logger.info("No existing KG found — starting fresh")
            # Add root node
            self._graph.add_node(
                "IIT Jammu",
                entity_type="Institution",
                description="Indian Institute of Technology Jammu",
                established="2016",
                location="Jagti, Nagrota, Jammu",
            )

    def save(self):
        """Persist graph to disk."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._nx.write_graphml(self._graph, self._path)
            logger.debug(f"KG saved: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to save KG: {e}")

    def add_entity(
        self,
        name: str,
        entity_type: str = "General",
        attributes: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Add or update an entity node. Returns the node ID (name).
        If the node already exists, attributes are merged (never deleted).
        """
        name = name.strip()
        if not name:
            return ""

        if entity_type not in ENTITY_TYPES:
            entity_type = "General"

        node_attrs = {
            "entity_type": entity_type,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if attributes:
            # Convert all values to strings for GraphML compatibility
            for k, v in attributes.items():
                node_attrs[k] = str(v) if v is not None else ""

        if self._graph.has_node(name):
            # Merge attributes (don't overwrite with empty)
            existing = dict(self._graph.nodes[name])
            for k, v in node_attrs.items():
                if v:  # only update non-empty
                    existing[k] = v
            self._graph.nodes[name].update(existing)
        else:
            self._graph.add_node(name, **node_attrs)

        return name

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "RELATED_TO",
        attributes: Optional[Dict[str, str]] = None,
    ):
        """Add a directed edge between two entities. Skips if already exists."""
        if not source or not target:
            return

        if rel_type not in RELATIONSHIP_TYPES:
            rel_type = "RELATED_TO"

        # Ensure both nodes exist
        if not self._graph.has_node(source):
            self.add_entity(source)
        if not self._graph.has_node(target):
            self.add_entity(target)

        edge_attrs = {"relationship": rel_type}
        if attributes:
            for k, v in attributes.items():
                edge_attrs[k] = str(v) if v is not None else ""

        if not self._graph.has_edge(source, target):
            self._graph.add_edge(source, target, **edge_attrs)
        else:
            # Update attributes on existing edge
            self._graph.edges[source, target].update(edge_attrs)

    def query(self, entity_name: str) -> Dict[str, Any]:
        """
        Query an entity and its 1-hop neighborhood.
        Returns entity attributes + all connected entities and relationships.
        """
        if not self._graph.has_node(entity_name):
            # Try fuzzy match
            entity_name = self._fuzzy_find(entity_name)
            if not entity_name:
                return {}

        result = {
            "entity": entity_name,
            "attributes": dict(self._graph.nodes[entity_name]),
            "outgoing": [],
            "incoming": [],
        }

        # Outgoing edges (this entity → others)
        for _, target, data in self._graph.out_edges(entity_name, data=True):
            result["outgoing"].append({
                "target": target,
                "relationship": data.get("relationship", "RELATED_TO"),
                "target_attrs": dict(self._graph.nodes.get(target, {})),
            })

        # Incoming edges (others → this entity)
        for source, _, data in self._graph.in_edges(entity_name, data=True):
            result["incoming"].append({
                "source": source,
                "relationship": data.get("relationship", "RELATED_TO"),
                "source_attrs": dict(self._graph.nodes.get(source, {})),
            })

        return result

    def _fuzzy_find(self, name: str) -> Optional[str]:
        """Find closest matching node name."""
        name_lower = name.lower().strip()
        for node in self._graph.nodes:
            if name_lower in node.lower() or node.lower() in name_lower:
                return node
        return None

    def search_relevant(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword-based search over KG nodes.
        Returns top_k entities with their context as formatted text.
        """
        query_words = set(re.findall(r"\b\w{2,}\b", query.lower()))
        if not query_words:
            return []

        scored = []
        for node, attrs in self._graph.nodes(data=True):
            node_text = f"{node} {' '.join(str(v) for v in attrs.values())}".lower()
            score = sum(1 for w in query_words if w in node_text)

            # Boost for entity type matching
            etype = attrs.get("entity_type", "")
            if any(w in etype.lower() for w in query_words):
                score += 2

            if score > 0:
                scored.append((score, node, attrs))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []

        for score, node, attrs in scored[:top_k]:
            # Build context string including relationships
            context_parts = [f"**{node}** ({attrs.get('entity_type', 'Entity')})"]

            for k, v in attrs.items():
                if k not in ("entity_type", "updated_at") and v:
                    context_parts.append(f"  - {k}: {v}")

            # Add relationship context
            for _, target, data in self._graph.out_edges(node, data=True):
                rel = data.get("relationship", "RELATED_TO")
                context_parts.append(f"  → {rel} → {target}")

            for source, _, data in self._graph.in_edges(node, data=True):
                rel = data.get("relationship", "RELATED_TO")
                context_parts.append(f"  ← {rel} ← {source}")

            results.append({
                "entity": node,
                "score": score,
                "context": "\n".join(context_parts),
                "attributes": dict(attrs),
            })

        return results

    def extract_and_add_from_text(self, text: str, title: str = "", source_url: str = ""):
        """
        Auto-extract entities and relationships from raw text and add to graph.
        Uses regex patterns to identify key information.
        """
        if not text or len(text) < 50:
            return

        # ── Extract Departments ────────────────────────────────────
        for dept in KNOWN_DEPARTMENTS:
            if dept.lower() in text.lower():
                dept_node = self.add_entity(dept, "Department")
                self.add_relationship("IIT Jammu", dept_node, "HAS_DEPARTMENT")

        # ── Extract Programs ───────────────────────────────────────
        for prog in KNOWN_PROGRAMS:
            if prog.lower() in text.lower():
                prog_node = self.add_entity(prog, "Program")
                self.add_relationship("IIT Jammu", prog_node, "OFFERS_PROGRAM")

        # ── Extract People (Prof./Dr.) ─────────────────────────────
        people = re.findall(
            r"(?:Prof\.|Dr\.|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            text
        )
        for person in people:
            person_name = person.strip()
            if len(person_name) > 5:
                self.add_entity(person_name, "Person", {"mentioned_in": title})

        # ── Extract Fees ───────────────────────────────────────────
        fee_patterns = re.findall(
            r"(?:Rs\.?\s*|₹\s*)([\d,]+(?:\.\d+)?)\s*(?:per\s+(?:year|annum|semester|month))?",
            text
        )
        for fee in fee_patterns:
            fee_val = fee.strip()
            if len(fee_val) >= 3:  # at least 3 digits
                fee_label = f"Fee: Rs {fee_val}"
                self.add_entity(fee_label, "Fee", {
                    "amount": fee_val,
                    "context": title,
                    "source_url": source_url,
                })
                self.add_relationship("IIT Jammu", fee_label, "HAS_FEE")

        # ── Extract Email addresses ────────────────────────────────
        emails = re.findall(r"[\w.+-]+@iitjammu\.ac\.in", text)
        for email in emails:
            contact_node = self.add_entity(email, "Contact", {"type": "email"})
            self.add_relationship("IIT Jammu", contact_node, "HAS_CONTACT")

        # ── Extract Phone numbers ──────────────────────────────────
        phones = re.findall(r"\+91[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}", text)
        for phone in phones:
            contact_node = self.add_entity(phone.strip(), "Contact", {"type": "phone"})
            self.add_relationship("IIT Jammu", contact_node, "HAS_CONTACT")

        # ── Extract Placement data ─────────────────────────────────
        if any(w in text.lower() for w in ["placement", "ctc", "lpa", "package", "recruit"]):
            lpa_matches = re.findall(r"([\d.]+)\s*(?:LPA|lpa|Lakhs?\s*Per\s*Annum)", text)
            for lpa in lpa_matches:
                placement_node = self.add_entity(
                    f"Placement CTC: {lpa} LPA", "Placement",
                    {"ctc_lpa": lpa, "context": title}
                )
                self.add_relationship("IIT Jammu", placement_node, "HAS_PLACEMENT")

        # ── Extract Scholarships ───────────────────────────────────
        scholarship_keywords = ["MCM", "Merit-cum-Means", "PMRF", "fellowship", "freeship", "scholarship"]
        for kw in scholarship_keywords:
            if kw.lower() in text.lower():
                sch_node = self.add_entity(kw, "Scholarship", {"context": title})
                self.add_relationship("IIT Jammu", sch_node, "HAS_SCHOLARSHIP")

        # ── Extract Facilities ─────────────────────────────────────
        facility_keywords = [
            "library", "hostel", "mess", "sports", "medical centre",
            "gym", "cafeteria", "canteen", "wi-fi", "wifi",
            "central workshop", "HPC", "high performance computing",
        ]
        for fac in facility_keywords:
            if fac.lower() in text.lower():
                fac_name = fac.title()
                fac_node = self.add_entity(fac_name, "Facility", {"context": title})
                self.add_relationship("IIT Jammu", fac_node, "HAS_FACILITY")

        # ── Link title as a general entity ─────────────────────────
        if title and len(title) > 5:
            self.add_entity(title, "General", {"source_url": source_url})
            self.add_relationship("IIT Jammu", title, "RELATED_TO")

    def seed_hods(self):
        """Seed canonical departments and their HOD relationships."""
        hod_data = [
            {
                "dept": "Computer Science and Engineering",
                "hod": "Dr. Yamuna Prasad",
                "email": "hod.cse@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/computer_science_engineering/message-from-deparment-hod"
            },
            {
                "dept": "Civil Engineering",
                "hod": "Dr. Surendra Beniwal",
                "email": "",
                "url": "https://www.iitjammu.ac.in/civil_engineering/hod-message"
            },
            {
                "dept": "Electrical Engineering",
                "hod": "Dr. Ravikant Saini",
                "email": "hod.ee@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/ee/hod.html"
            },
            {
                "dept": "Mechanical Engineering",
                "hod": "Dr. B. Satya Sekhar",
                "email": "hod.me@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/mechanical_engineering/hod.html"
            },
            {
                "dept": "Chemical Engineering",
                "hod": "Dr. Ravi Kumar Arun",
                "email": "hod.chemical@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/chemical-engineering/hod.html"
            },
            {
                "dept": "Materials Engineering",
                "hod": "Dr. Rani Rohini",
                "email": "",
                "url": "https://iitjammu.ac.in/materials-engineering"
            },
            {
                "dept": "Chemistry",
                "hod": "Dr. Guru B. Ramani",
                "email": "",
                "url": "https://www.iitjammu.ac.in/chemistry/message-from-head-of-the-department"
            },
            {
                "dept": "Physics",
                "hod": "Dr. Venkata Sathish Akella",
                "email": "hod.physics@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/physics/hod.html"
            },
            {
                "dept": "Mathematics",
                "hod": "Dr. Rahul Dattatraya Kitture",
                "email": "",
                "url": "https://www.iitjammu.ac.in/mathematics/hod-message"
            },
            {
                "dept": "Biosciences and Bioengineering",
                "hod": "Dr. Mithu Baidya",
                "email": "",
                "url": "https://iitjammu.ac.in/bsbe"
            },
            {
                "dept": "Humanities and Social Sciences",
                "hod": "Dr. Amitash Ojha",
                "email": "hod.hss@iitjammu.ac.in",
                "url": "https://www.iitjammu.ac.in/hss/hod.html"
            }
        ]

        for item in hod_data:
            dept_node = self.add_entity(item["dept"], "Department")
            self.add_relationship("IIT Jammu", dept_node, "HAS_DEPARTMENT")
            
            # Prepare HOD attributes
            hod_attrs = {"designation": "Head of Department", "department": item["dept"]}
            if item["email"]:
                hod_attrs["email"] = item["email"]
            if item["url"]:
                hod_attrs["source_url"] = item["url"]
                
            hod_node = self.add_entity(item["hod"], "Person", hod_attrs)
            self.add_relationship(dept_node, hod_node, "LED_BY")
            
            if item["email"]:
                contact_node = self.add_entity(item["email"], "Contact", {
                    "type": "email", 
                    "department": item["dept"],
                    "owner": item["hod"]
                })
                self.add_relationship(hod_node, contact_node, "HAS_CONTACT")

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        entity_types = {}
        for _, attrs in self._graph.nodes(data=True):
            etype = attrs.get("entity_type", "Unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "entity_types": entity_types,
            "kg_file": self._path,
        }

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()


# ── Singleton ─────────────────────────────────────────────────────
_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the singleton KnowledgeGraph instance."""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg
