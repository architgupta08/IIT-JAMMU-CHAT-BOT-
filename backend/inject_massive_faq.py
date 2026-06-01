import json
import logging
from vectorstore.chroma_store import get_chroma_store
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dense fact sheets covering the requested questions
FACT_SHEETS = [
    {
        "title": "IIT Jammu General Information & Core Facts",
        "topic": "General Institute Info",
        "doc_type": "faq_factsheet",
        "text": """
IIT Jammu (Indian Institute of Technology Jammu) is a premier Institute of National Importance established in 2016 by the Government of India. It is a fully operational government institute recognized by AICTE. 
Location & Address: The main campus is located in Jagti, Nagrota, Jammu, Jammu and Kashmir 181221. The transit campus is in Paloura.
Campus Size: The main campus spans approximately 400 acres.
Director: Prof. Manoj Singh Gaur is the Director of IIT Jammu.
Contact: Official website is https://www.iitjammu.ac.in. Email is usually directed to specific departments, e.g., admissions@iitjammu.ac.in.
Is it good/famous?: It is rapidly growing, famous for its strategic location, state-of-the-art labs, strong CSE/AI focus, and being one of the newer generation IITs with excellent modern infrastructure.
Students & Departments: It has over 1200+ students and multiple departments including CSE, Electrical, Mechanical, Civil, Chemical, Materials, and HSS.
International Collaborations: Yes, it has MOUs with various international universities and research labs.
"""
    },
    {
        "title": "IIT Jammu Admissions, Fees & Cutoffs",
        "topic": "Admissions",
        "doc_type": "faq_factsheet",
        "text": """
BTech Admission: Strictly through JEE Advanced followed by JoSAA counselling. Admission is highly competitive.
BTech Branches: Computer Science and Engineering (CSE), Electrical Engineering (EE), Mechanical Engineering (ME), Civil Engineering (CE), Chemical Engineering (ChE), Materials Engineering.
MTech Admission: Based on GATE score and COAP counselling. Non-CS students can apply for some interdisciplinary/CS programs if they meet the specific GATE paper and interview criteria. 
PhD Admission: Applications open typically twice a year. Requires valid GATE/UGC-NET/CSIR-NET or CPI > 8.0 from CFTIs. Apply via the official portal.
Cutoffs (Approximate): CSE is the most sought-after, usually closing within the top 4000-5000 ranks in JEE Advanced. With 10,000 rank, getting core CSE is difficult, but other branches like Civil or Materials might be possible depending on the year and category.
Tuition Fee: For general BTech students, it is approx 1 Lakh per semester. SC/ST/PH students get full tuition fee waiver. MCM (Merit-cum-Means) scholarships are available for low-income students.
Foreign Students: Accepted through specific international admission modes or ICCR.
Branch Change: Subject to academic performance (CPI) at the end of the first year, strictly based on seat availability and merit.
"""
    },
    {
        "title": "Academics, Courses & Grading",
        "topic": "Academics",
        "doc_type": "faq_factsheet",
        "text": """
Curriculum: IIT Jammu offers flexible, modern curriculum aligned with NEP 2020. It includes interdisciplinary courses and minor programs.
Computer Science (CSE): The best and most demanding department. Teaches Machine Learning, AI, Cybersecurity, Data Structures, Python, C++, Java, etc.
Mechanical Engineering: Excellent labs, focuses on robotics, design, and thermal engineering.
AI & Data Science: Taught extensively as core/elective courses within CSE. There are dedicated AI research groups.
Grading System: Relative grading system. Performance is measured by CPI (Cumulative Performance Index) and SPI (Semester Performance Index) on a 10-point scale.
Attendance: Strictly monitored. Typically, 75% attendance is compulsory to sit for exams.
First Year: Common courses for all branches including Physics, Math, Chemistry, basic programming, and workshop.
Passing/Failing: If you fail a subject, you must repeat it or clear the backlog exam. Continuous poor performance can lead to academic probation.
"""
    },
    {
        "title": "Hostel, Mess & Campus Life",
        "topic": "Campus Life",
        "doc_type": "faq_factsheet",
        "text": """
Hostels: Fully operational, modern hostels. Typically twin-sharing for juniors, single for seniors. Girls' hostels are completely separate and secure.
AC Rooms: Generally, standard hostel rooms are NOT air-conditioned due to the pleasant weather, but centralized AC exists in academic blocks and libraries.
Facilities: 24x7 High-speed WiFi, laundry services, gym, and medical facilities are available.
Mess Food & Fees: Nutritious mess food is provided. Mess fee is separate from tuition and paid per semester.
Canteens: Campus has day canteens and night canteens operating till late hours.
Safety & Ragging: ZERO TOLERANCE for ragging. Strict anti-ragging policies. The campus is highly secure with 24x7 guards.
Rules: Students must follow hostel timings. Staying outside overnight usually requires prior permission (out-pass). Alcohol and illegal substances are strictly prohibited on campus.
Sports: Cricket ground, football field, basketball, badminton courts, and indoor sports facilities exist.
"""
    },
    {
        "title": "Placements, Internships & Career",
        "topic": "Placements",
        "doc_type": "faq_factsheet",
        "text": """
Placement Cell: The Training and Placement Cell (TnP) actively invites companies.
Companies: Top recruiters include Google, Amazon, Microsoft, Samsung, MathWorks, Arista Networks, and various PSUs.
Packages: The highest package often crosses 40-50 LPA. The average package is around 15-18 LPA overall, with CSE averaging higher (20+ LPA).
Internships: TnP assists with summer internships. Many students secure off-campus internships at top tech firms.
Minimum CGPA for Placements: While TnP allows students with no active backlogs, companies often set their own cutoffs (usually 7.0 or 7.5+ CPI).
Non-CS Software Jobs: Yes, Mechanical, Civil, and EE students frequently crack software/IT jobs by building strong coding skills.
Startup Support: Institute Innovation Council (IIC) and incubation cells support startup funding and entrepreneurship.
"""
    },
    {
        "title": "Student Clubs, Fests & Extracurriculars",
        "topic": "Student Life",
        "doc_type": "faq_factsheet",
        "text": """
Clubs: Active coding clubs (e.g., Coding Club, Google Developer Student Club), Robotics Club, Music, Dance, Dramatics, and Literature clubs.
Fests: 'Renao' is the annual cultural fest. 'Techno-Cultural' events and hackathons happen regularly.
Joining Clubs: Freshers are highly encouraged to join clubs during orientation and club recruitment drives.
Stress & Life: IIT life can be academically rigorous, but clubs and sports provide excellent stress relief. Counseling services are available for mental health support.
"""
    },
    {
        "title": "Campus Navigation & IT Services",
        "topic": "Navigation & Utilities",
        "doc_type": "faq_factsheet",
        "text": """
Library: Central library with extensive timings (often late night during exams).
IT Services & WiFi: To access WiFi or reset passwords, contact the Computer Center (CC) or use the intranet portal.
Locations: Admin Block, Lecture Hall Complex (LHC), and Medical Center are centrally located in the Jagti campus.
Transport: Campus transport (e-rickshaws/buses) runs between hostels and academic blocks. Buses also ply between the campus and Jammu city/railway station.
Bank/ATM: Banking facilities and ATMs are available on campus.
Complaints: Hostel or maintenance complaints are logged through an online portal (e.g., e-Governance portal).
"""
    },
    {
        "title": "Conversational & Human-like Advice",
        "topic": "Advice",
        "doc_type": "faq_factsheet",
        "text": """
Is it worth joining?: Yes, IIT Jammu carries the prestigious IIT tag, excellent faculty, and great placements.
Nervous about joining/Survival: It's normal! The first year has common courses. Focus on time management, and you will survive and thrive.
Coding Necessity: Even for non-CS branches, basic coding (Python/C++) is highly recommended as it helps in placements and modern research.
Depression/Homesickness: Common initially. Reach out to the student wellness center, counselors, or faculty advisors immediately.
High CPI Shortcut: There are no shortcuts. Attend classes regularly, submit assignments on time, and study past papers.
IIT Delhi vs IIT Jammu: IIT Delhi is an older, established generation 1 IIT with vast alumni networks. IIT Jammu is newer but offers excellent modern facilities and rapid growth. Choose based on rank and branch preference.
"""
    },
    {
        "title": "Research, Labs & Faculty Capabilities",
        "topic": "Research",
        "doc_type": "faq_factsheet",
        "text": """
Faculty Quality: IIT Jammu has highly qualified faculty, mostly PhDs from older IITs or foreign universities.
Research Labs: Advanced labs for AI, Robotics, Computer Vision, Cybersecurity, IoT, Drone Technology, and Advanced Manufacturing.
UG Research: BTech students are strongly encouraged to participate in research, publish papers, and work on funded projects.
Funding & Grants: Faculty execute projects funded by DRDO, ISRO, DST, and international agencies.
Publications: Faculty regularly publish in top-tier journals (IEEE, Springer, Nature) and conferences (CVPR, NeurIPS).
Approaching Faculty: Check their office hours or email them professionally outlining your interest in their specific research domain to ask for projects or internships.
"""
    }
]

def inject_facts():
    chroma = get_chroma_store()
    
    docs_to_add = []
    for sheet in FACT_SHEETS:
        docs_to_add.append({
            "text": sheet["text"],
            "title": sheet["title"],
            "topic": sheet["topic"],
            "doc_type": sheet["doc_type"],
            "source_url": "https://www.iitjammu.ac.in/faq",
            "year": "2025"
        })
        
    logger.info(f"Injecting {len(docs_to_add)} dense fact sheets into ChromaDB...")
    inserted = chroma.add_documents(docs_to_add, batch_size=10)
    logger.info(f"Successfully inserted {inserted} new fact sheets.")

if __name__ == "__main__":
    inject_facts()
