"""
inject_phd_supervisor_faqs.py
Reads all department PhD-list markdown files, parses every student's
supervisor and research area, creates one FAQ per student, and injects
them into ChromaDB. Skips students who have no supervisor listed.
"""

import sys, os, re, json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

PHD_FILES = {
    "www.iitjammu.ac.in__ee__phd-list.html.md":                      ("https://www.iitjammu.ac.in/ee/phd-list.html",                      "Electrical Engineering"),
    "www.iitjammu.ac.in__computer_science_engineering__phd-list.md":  ("https://www.iitjammu.ac.in/computer_science_engineering/phd-list",  "Computer Science and Engineering"),
    "www.iitjammu.ac.in__civil_engineering__phd-list.md":             ("https://www.iitjammu.ac.in/civil_engineering/phd-list",             "Civil Engineering"),
    "www.iitjammu.ac.in__chemical-engineering__phd-list.html.md":     ("https://www.iitjammu.ac.in/chemical-engineering/phd-list.html",     "Chemical Engineering"),
    "www.iitjammu.ac.in__chemistry__phd-list.md":                     ("https://www.iitjammu.ac.in/chemistry/phd-list",                     "Chemistry"),
    "www.iitjammu.ac.in__mathematics__phd-list.md":                   ("https://www.iitjammu.ac.in/mathematics/phd-list",                   "Mathematics"),
    "www.iitjammu.ac.in__physics__phd-list.html.md":                  ("https://www.iitjammu.ac.in/physics/phd-list.html",                  "Physics"),
    "www.iitjammu.ac.in__hss__phd-list.html.md":                      ("https://www.iitjammu.ac.in/hss/phd-list.html",                     "Humanities and Social Sciences"),
    "www.iitjammu.ac.in__bsbe__phd-list.html.md":                     ("https://www.iitjammu.ac.in/bsbe/phd-list.html",                    "Biosciences and Bioengineering"),
}

def parse_phd_students(content):
    """Parse student name, supervisor, and research area from PhD list markdown."""
    students = []
    # Split into student blocks by #### heading
    blocks = re.split(r'\n#{2,4} ', content)
    for block in blocks[1:]:  # skip the first part (page header)
        lines = block.strip().split('\n')
        if not lines:
            continue
        name = lines[0].strip().lstrip('#').strip()
        if not name or len(name) < 2:
            continue

        full_block = '\n'.join(lines[1:])
        
        # Extract supervisor - look for "Supervisor" keyword followed by name
        sup_match = re.search(
            r'\*\*Supervisor\*\*\s*[\n\r]+\s*([^\n\r]+)|'
            r'Supervisor[:\s]*[\n\r]+\s*([^\n\r]+)|'
            r'\*\*Supervisor\*\*\s*([^\n\r]+)',
            full_block, re.I
        )
        supervisor = None
        if sup_match:
            supervisor = (sup_match.group(1) or sup_match.group(2) or sup_match.group(3) or '').strip()
            supervisor = re.sub(r'\*+', '', supervisor).strip()
            supervisor = re.sub(r'\s+', ' ', supervisor).strip()
            if len(supervisor) < 3:
                supervisor = None
        
        # Extract research area
        area_match = re.search(r'Research Area\s*[\n\r]+\s*([^\n\r]+)', full_block, re.I)
        research_area = None
        if area_match:
            research_area = area_match.group(1).strip()

        students.append({
            'name': name,
            'supervisor': supervisor,
            'research_area': research_area,
        })
    return students


def build_docs(students, source_url, department):
    docs = []
    for s in students:
        name = s['name']
        supervisor = s['supervisor']
        research_area = s['research_area']
        
        # Always create a doc with whatever info we have
        if supervisor:
            sup_text = (
                f"IIT Jammu PhD Scholar Information\n"
                f"Scholar Name: {name}\n"
                f"Department: {department}\n"
                f"Supervisor: {supervisor}\n"
            )
            if research_area:
                sup_text += f"Research Area: {research_area}\n"
            sup_text += f"Source: {source_url}"
            
            docs.append({
                'text': sup_text,
                'title': f"PhD Scholar: {name} - Supervisor: {supervisor}",
                'topic': 'PhD Scholars',
                'source_url': source_url,
                'department': department,
                'doc_type': 'phd_scholar',
                'document_type': 'PhD_Scholar',
                'last_updated': '2026-05-31',
                'target_audience': 'PhD',
                'year': '2026',
            })
            
            # Also build FAQ-style doc
            faq_text = (
                f"IIT Jammu FAQ\n"
                f"Question: Who is the supervisor of {name}?\n"
                f"Answer: The supervisor of {name} is {supervisor}.\n"
                f"Department: {department}\n"
            )
            if research_area:
                faq_text += f"Research Area: {research_area}\n"
            docs.append({
                'text': faq_text,
                'title': f"PhD FAQ: Supervisor of {name}",
                'topic': 'PhD Scholars',
                'source_url': source_url,
                'department': department,
                'doc_type': 'phd_supervisor_faq',
                'document_type': 'PhD_Scholar',
                'last_updated': '2026-05-31',
                'target_audience': 'PhD',
                'year': '2026',
            })
        elif research_area:
            # No supervisor, but has research area
            ra_text = (
                f"IIT Jammu PhD Scholar Information\n"
                f"Scholar Name: {name}\n"
                f"Department: {department}\n"
                f"Research Area: {research_area}\n"
                f"Source: {source_url}"
            )
            docs.append({
                'text': ra_text,
                'title': f"PhD Scholar: {name} - {department}",
                'topic': 'PhD Scholars',
                'source_url': source_url,
                'department': department,
                'doc_type': 'phd_scholar',
                'document_type': 'PhD_Scholar',
                'last_updated': '2026-05-31',
                'target_audience': 'PhD',
                'year': '2026',
            })
    return docs


def main():
    from vectorstore.chroma_store import get_chroma_store
    chroma = get_chroma_store()

    print("=" * 60)
    print("  INJECTING PhD SUPERVISOR DATA FOR ALL DEPARTMENTS")
    print("=" * 60)
    print(f"  Initial ChromaDB count: {chroma.count()}")
    
    total_students = 0
    total_with_supervisor = 0
    total_docs_added = 0

    for filename, (source_url, department) in PHD_FILES.items():
        fpath = os.path.join(raw_dir, filename)
        if not os.path.exists(fpath):
            print(f"  [SKIP] File not found: {filename}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()

        students = parse_phd_students(content)
        total_students += len(students)
        with_sup = sum(1 for s in students if s['supervisor'])
        total_with_supervisor += with_sup

        docs = build_docs(students, source_url, department)
        added = chroma.add_documents(docs)
        total_docs_added += added

        print(f"  [{department}] {len(students)} students, {with_sup} with supervisors -> {added} docs added")

        # Show sample
        for s in students:
            if s['supervisor']:
                print(f"      - {s['name']} -> {s['supervisor']}")
        print()

    print("=" * 60)
    print(f"  DONE!")
    print(f"  Total students parsed: {total_students}")
    print(f"  Students with supervisors: {total_with_supervisor}")
    print(f"  Docs added to ChromaDB: {total_docs_added}")
    print(f"  Final ChromaDB count: {chroma.count()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
