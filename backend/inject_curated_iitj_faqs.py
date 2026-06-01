"""
Generate and inject a current, source-tagged IIT Jammu FAQ set.

This script creates 300-400 natural-language question/answer records with
informal variants, then injects them into ChromaDB as high-priority
`curated_faq` documents.

Run from backend:
    python inject_curated_iitj_faqs.py
"""

import json
import os
import re
import sys
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.append(".")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
OUT_FILE = ROOT / "data" / "processed" / "iitj_curated_faqs.json"
FACULTY_FILE = BACKEND / "data" / "processed" / "faculty_data.json"

FACULTY_ROOMS_MAP = {}
BUILDING_FLOOR_MAPPING = []
BUILDING_GROUND_LEVEL = {}

def normalize_name(name: str) -> str:
    name = (name or "").lower().strip()
    name = re.sub(r"^(dr|prof|professor|shri|mr|ms|mrs)\b\.?", "", name).strip()
    name = re.sub(r"[.\s_-]+", " ", name).strip()
    return name

def get_room_floor_info(room_number: str) -> str:
    room = (room_number or "").strip().upper()
    if not room:
        return "Unknown Floor"
    if "G" in room:
        return "Ground Floor"
    match = re.search(r"11AC(\d)", room)
    if match:
        floor_num = match.group(1)
        suffix = "th"
        if floor_num == "1": suffix = "st"
        elif floor_num == "2": suffix = "nd"
        elif floor_num == "3": suffix = "rd"
        return f"{floor_num}{suffix} Floor (Level L{floor_num})"
    return "Unknown Floor"

def load_faculty_rooms():
    global FACULTY_ROOMS_MAP, BUILDING_FLOOR_MAPPING, BUILDING_GROUND_LEVEL
    FACULTY_ROOMS_FILE = BACKEND / "data" / "processed" / "faculty_rooms.json"
    if FACULTY_ROOMS_FILE.exists():
        try:
            data = json.loads(FACULTY_ROOMS_FILE.read_text(encoding="utf-8"))
            for entry in data.get("faculty_rooms", []):
                norm = normalize_name(entry.get("faculty_name", ""))
                if norm:
                    FACULTY_ROOMS_MAP[norm] = entry
            BUILDING_FLOOR_MAPPING = data.get("floor_mapping", [])
            BUILDING_GROUND_LEVEL = data.get("ground_level", {})
        except Exception as e:
            print(f"Error loading faculty rooms JSON: {e}")

load_faculty_rooms()


def clean_html(value) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def source_note(source: str) -> str:
    return f"Source: {source}"


CORE_FACTS = [
    {
        "topic": "11AC Academic Block Floor Mappings",
        "source": "IIT Jammu Pushkar Block Directory",
        "answer": (
            "The 11AC Academic Block (commonly known as the Pushkar Block) at IIT Jammu is an 8-story building. "
            "Its floor level mappings are as follows:\n"
            "- **Ground Floor (L0)**: Ground level rooms are split into left side (Rooms 11ACG001 - 11ACG007) and right side (Rooms 11ACG008 - 11ACG014).\n"
            "- **Floor 1 (Level L1)**: Room range 11AC1001 - 11AC1026.\n"
            "- **Floor 2 (Level L2)**: Room range 11AC2001 - 11AC2046.\n"
            "- **Floor 3 (Level L3)**: Room range 11AC3001 - 11AC3056.\n"
            "- **Floor 4 (Level L4)**: Room range 11AC4001 - 11AC4042.\n"
            "- **Floor 5 (Level L5)**: Room range 11AC5001 - 11AC5072.\n"
            "- **Floor 6 (Level L6)**: Room range 11AC6001 - 11AC6032.\n"
            "- **Floor 7 (Level L7)**: Room range 11AC7001 - 11AC7032.\n"
            "Rooms starting with 11AC followed by a digit indicate the floor level (e.g., room 11AC7021 is on Floor 7). "
            + source_note("IIT Jammu Pushkar Block Directory")
        ),
        "questions": [
            "What is the floor mapping of 11AC Block?",
            "What is the floor mapping of Pushkar Block?",
            "Show room range for floors in Pushkar Block.",
            "which floor has room 11AC6012?",
            "where is room 11AC5002?",
            "where are the rooms in 11AC Academic Block?",
            "which floor is level L5?",
            "rooms range for level L3",
            "where are ground level rooms 11ACG001 to 11ACG014?",
            "where is room 11ACG005?",
            "left side and right side rooms of ground floor pushkar block",
            "pushkar block room ranges"
        ]
    },
    {
        "topic": "Underwater Research Lab",
        "source": "https://www.iitjammu.ac.in/ee/index.html",
        "answer": (
            "The Underwater Research Lab (also known as the Underwater Artificial Intelligence Lab) at IIT Jammu is "
            "located at the Jagti Campus (near the SBI ATM). It houses a unique 25ft x 20ft underwater acoustic test facility simulating "
            "oceanic conditions including waves and noise. The research focuses on underwater robotics, autonomous "
            "systems, surveillance, and signal processing. The lab is headed by Dr. Karan Nathwani (Associate Professor, Electrical Engineering) "
            "along with Dr. Badri Narayan Subudhi (Associate Professor, Electrical Engineering) and Dr. Ankur Bansal (Assistant Professor, Electrical Engineering). "
            + source_note("https://www.iitjammu.ac.in/ee/index.html")
        ),
        "questions": [
            "Where is the underwater lab located in IIT Jammu?",
            "Where is the underwater lab located?",
            "underwater lab location campus Jagti Paloura?",
            "which faculty are in underwater research lab?",
            "Who is in the underwater research lab?",
            "who heads the underwater lab at IIT Jammu?",
            "What is the research focus of the underwater lab?",
            "underwater lab kaunse campus me hai?",
            "IIT Jammu underwater lab location?",
            "is there any underwater lab in iit jammu?",
            "is there any underwater AI lab?"
        ]
    },
    {
        "topic": "About IIT Jammu",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "IIT Jammu is an Indian Institute of Technology and an Institute of National Importance. "
            "Its permanent campus is at Jagti, NH-44, P.O. Nagrota, Jammu - 181221, Jammu and Kashmir. "
            "For the latest notices, admissions, tenders, and contacts, use the official website. "
            + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "What is IIT Jammu?",
            "Tell me about IIT Jammu.",
            "IIT Jammu kya hai?",
            "iit jammu basic info batao",
            "Is IIT Jammu a government IIT?",
            "Where is IIT Jammu located?",
            "iit jammu address?",
            "IIT Jammu campus kaha hai?",
        ],
    },
    {
        "topic": "Director and Leadership",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "The Director of IIT Jammu is Prof. Manoj Singh Gaur. For dean, registrar, committee, "
            "and office-holder changes, prefer the official IIT Jammu administration pages because "
            "these roles can change. " + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "Who is the Director of IIT Jammu?",
            "iit jammu director name?",
            "Director kaun hai IIT Jammu ka?",
            "Tell me the current director of IIT Jammu.",
            "Who leads IIT Jammu?",
            "IIT Jammu administration head?",
        ],
    },
    {
        "topic": "BTech Admission",
        "source": "https://www.iitjammu.ac.in/Programme/ugadmissions",
        "answer": (
            "B.Tech admission at IIT Jammu is through JEE Advanced followed by JoSAA counselling. "
            "Seat allotment depends on JEE Advanced rank, category, seat matrix, and JoSAA choices. "
            "Always check JoSAA and the IIT Jammu UG admissions page for the current year's rules. "
            + source_note("https://www.iitjammu.ac.in/Programme/ugadmissions")
        ),
        "questions": [
            "How can I get BTech admission in IIT Jammu?",
            "btech admission kaise milega IIT Jammu me?",
            "Can I join IIT Jammu through JEE Main only?",
            "I got JEE Advanced rank, how to apply IIT Jammu?",
            "IIT Jammu BTech admission process?",
            "How to take admission in IIT Jammu after 12th?",
            "iit jammu me btech entry ka process kya hai?",
            "JOSAA se IIT Jammu milega kya?",
        ],
    },
    {
        "topic": "MTech Admission",
        "source": "https://www.iitjammu.ac.in/Programme/pgadmissions",
        "answer": (
            "M.Tech admission at IIT Jammu is generally based on a valid GATE score and the official "
            "PG admission process, which may include shortlisting, department criteria, and/or interview. "
            "Shortlists and criteria are published on the official PG admissions page for the relevant year. "
            + source_note("https://www.iitjammu.ac.in/Programme/pgadmissions")
        ),
        "questions": [
            "How can I apply for MTech at IIT Jammu?",
            "mtech admission ka process kya hai?",
            "Is GATE required for MTech IIT Jammu?",
            "IIT Jammu MTech eligibility?",
            "MTech CSE me admission kaise?",
            "Can non-CS student apply for MTech CSE at IIT Jammu?",
            "PG admission IIT Jammu details?",
            "mtech shortlist kaha milegi?",
        ],
    },
    {
        "topic": "PhD Admission",
        "source": "https://www.iitjammu.ac.in/Programme/phdadmissions",
        "answer": (
            "Ph.D admission at IIT Jammu is handled through the official Ph.D admissions portal and "
            "department-wise criteria. Eligibility can depend on degree, CPI/marks, GATE/NET or other "
            "valid qualifications, and interview performance. Use the current call for applications for "
            "deadlines and exact criteria. " + source_note("https://www.iitjammu.ac.in/Programme/phdadmissions")
        ),
        "questions": [
            "How to apply for PhD in IIT Jammu?",
            "phd admission kaise hota hai IIT Jammu me?",
            "Can I do direct PhD after BTech at IIT Jammu?",
            "IIT Jammu PhD eligibility?",
            "PhD interview process IIT Jammu?",
            "When PhD forms open IIT Jammu?",
            "I want research admission in IIT Jammu.",
            "PhD ke liye GATE compulsory hai kya?",
        ],
    },
    {
        "topic": "BTech Fee 2025-26",
        "source": "https://www.iitjammu.ac.in/academics/fee/2025-26/B%20Tech%202025_Short%202025-26.pdf",
        "answer": (
            "For the B.Tech 2025 batch in academic year 2025-26, the official fee PDF lists tuition "
            "per semester as Rs. 1,00,000 for General/OBC/EWS students with parental income above "
            "Rs. 5 lakh, Rs. 33,333 for income between Rs. 1 lakh and Rs. 5 lakh, and zero tuition "
            "for income below Rs. 1 lakh. SC/ST/PH students have zero tuition fee. The same PDF lists "
            "semester fee of Rs. 35,480 for single occupancy and Rs. 30,370 for shared occupancy, plus "
            "one-time and security fees at admission. Mess charges are notified separately. "
            + source_note("IIT Jammu B.Tech 2025-26 fee PDF")
        ),
        "questions": [
            "What is the BTech fee at IIT Jammu?",
            "btech fees kitni hai IIT Jammu me?",
            "IIT Jammu BTech 2025 fee structure?",
            "How much tuition fee for BTech general category?",
            "SC ST BTech fee waiver IIT Jammu?",
            "income below 1 lakh fee IIT Jammu?",
            "BTech hostel plus tuition total kitna?",
            "shared occupancy fee IIT Jammu BTech?",
            "single room fee BTech IIT Jammu?",
            "mess fee included in BTech fee?",
        ],
    },
    {
        "topic": "Hostel Facilities",
        "source": "https://iitjammu.ac.in/Programme/ugadmissions/Hostels%20Admission%20FAQ.pdf",
        "answer": (
            "IIT Jammu has separate hostels for boys and girls, with 9 Boys' hostels (including Canary Hostel, "
            "Braeg Hostel, and Fulgar Hostel) and 2 Girls' hostels (including Dedhar Hostel and Egret Hostel) "
            "at the Jagti campus. Hostel facilities include WiFi, laundry centers, reading rooms, common halls, "
            "recreation areas, indoor games facilities, water coolers, and washing machines. Rooms are generally "
            "twin-sharing, equipped with basic furniture (bed, table, chair, wardrobe, bookshelf, book rack). "
            "Additionally, there is a New Hostel Block, which is an eight-storey centrally air-conditioned building "
            "providing 662 single-seater rooms. A four-storey dining hall is also available that can accommodate "
            "approximately 480 students at a time. Mattresses, pillows, buckets, etc. must be brought by the student or purchased "
            "locally. Prohibited items include water heaters, irons, electric stoves, and table fans. "
            + source_note("IIT Jammu Hostels & Facilities FAQ")
        ),
        "questions": [
            "What hostel facilities are available at IIT Jammu?",
            "hostel room me kya kya milta hai?",
            "IIT Jammu hostel twin sharing hai kya?",
            "Are boys and girls hostels separate?",
            "Can I bring electric kettle or iron to hostel?",
            "hostel me mattress milta hai kya?",
            "IIT Jammu hostel gym available?",
            "Is internet available in hostel rooms?",
            "hostel allocation kaise hota hai?",
            "Can I bring table fan in IIT Jammu hostel?",
            "What are the names of the hostels at IIT Jammu?",
            "hostel names IIT Jammu",
            "how many hostels are in iit jammu?",
            "total hostels in iit jammu?",
            "boys hostel and girls hostel details?",
            "what is the new hostel block?",
            "details about new hostel block at iit jammu",
            "how many seats in new hostel block",
            "dining hall seating capacity in hostel"
        ],
    },
    {
        "topic": "Mess & Dining Facilities",
        "source": "https://iitjammu.ac.in/academics/fee/2025-26",
        "answer": (
            "IIT Jammu features multiple dining options including the Canary Mess, Dedhar Mess, Egret Mess, and "
            "Annapurna Mess, serving vegetarian and non-vegetarian meals. The mess menu is managed by a student committee "
            "with mess charges around Rs. 3,200 to 3,500 per month (notified as a mess advance). Food & utility facilities "
            "on campus include Café Coffee Day (CCD) and Nescafe outlets, multiple academic canteens, and a night canteen "
            "operating until midnight. For conveniences, a daily utility store and stationery facilities are available inside the campus. "
            "Banking facilities (including an SBI ATM) are also accessible on campus. "
            + source_note("IIT Jammu Mess & Dining Circular")
        ),
        "questions": [
            "What is the mess fee at IIT Jammu?",
            "mess charges kitne hain IIT Jammu?",
            "Is mess fee included in BTech fee?",
            "Where can I pay mess fee?",
            "IIT Jammu mess advance kya hai?",
            "hostel fee and mess fee same hai kya?",
            "what are the canteens on campus?",
            "canteen timing and locations?",
            "is there a night canteen in iit jammu?",
            "what dining options are available?",
            "mess names at iit jammu",
            "convenience store inside campus?",
            "is there an ATM or bank inside iit jammu?",
            "banking facilities at iit jammu"
        ],
    },
    {
        "topic": "Campus Buildings & Blocks Overview",
        "source": "https://www.iitjammu.ac.in/permanent-campus",
        "answer": (
            "The permanent IIT Jammu main campus is located at Jagti on NH-44, Nagrota, spread across 400 acres "
            "with Phase 1A operational. The primary buildings/blocks include Pushkar Block (Academic Block), Mansar Block (Lecture Hall Block), "
            "and the Paloura Campus (the original/old campus). An SBI ATM is located near the campus entrance. "
            + source_note("IIT Jammu Infrastructure & Permanent Campus Page")
        ),
        "questions": [
            "how many blocks are in iit?",
            "how many blocks are in iit jammu?",
            "what blocks are on campus?",
            "iit jammu locations and buildings"
        ],
    },
    {
        "topic": "Pushkar Block (Academic Block)",
        "source": "https://www.iitjammu.ac.in/permanent-campus",
        "answer": (
            "Pushkar Block (Academic Block) is a major eight-story engineering academic block at the IIT Jammu Jagti campus "
            "serving as a hub for modern academic and research infrastructure. The building contains 10 classrooms (studio-like, fully equipped "
            "spaces for undergraduate and postgraduate lectures), 52 laboratories (including 25 engineering laboratories), "
            "104 faculty cabins/offices, seminar halls (such as Seminar Hall 11AC3027 and Seminar Hall-I), student discussion areas, "
            "WiFi-enabled infrastructure, and Innovation Centers for STEM outreach programs, hackathons, and entrepreneurship workshops. "
            "It supports departmental teaching, labs, and research activities. The departments associated with Pushkar Block include:\n"
            "- **Mechanical Engineering**: Founding department, located in Pushkar Block, containing mechanical labs, workshops, classrooms, and offices. Research includes thermal engineering, robotics, manufacturing, materials, and design.\n"
            "- **Computer Science and Engineering (CSE)**: Classes, computer labs, AI/ML research, coding infrastructure, and project spaces are hosted here.\n"
            "- **Electrical Engineering (EE)**: Houses electrical labs, electronics systems labs, signal processing, embedded systems, and power systems facilities.\n"
            "- **Civil Engineering (CE)**: Includes structural engineering labs, surveying labs, geotechnical facilities, and transportation engineering research spaces.\n"
            "- **Chemical Engineering**: Main department is in North Block, but shares academic/laboratory spaces and interdisciplinary activities in Pushkar. "
            + source_note("IIT Jammu Infrastructure & Permanent Campus Page")
        ),
        "questions": [
            "What is Pushkar building?",
            "what is pushkar block",
            "pushkar block details",
            "how many floors in pushkar block",
            "how many laboratories in pushkar block",
            "departments in pushkar block",
            "pushkar block classrooms and labs",
            "which departments are associated with pushkar block",
            "where is mechanical engineering department located?"
        ],
    },
    {
        "topic": "Mansar Block (Lecture Hall Block)",
        "source": "https://www.iitjammu.ac.in/permanent-campus",
        "answer": (
            "Mansar Block / Building (Lecture Hall Block) at IIT Jammu is a major three-story academic facility located within the permanent Jagti campus. "
            "It primarily serves as a central hub for student learning and institutional events. It houses modern multi-story lecture rooms/classrooms, "
            "the Central Library, and the Mansar Auditorium (a 300-seat auditorium used for conclaves, talks, and events). "
            "The total lecture capacity of Mansar Block exceeds 2,500 students. "
            + source_note("IIT Jammu Infrastructure & Permanent Campus Page")
        ),
        "questions": [
            "What is Mansar building?",
            "what is mansar block",
            "mansar block details",
            "mansar building capacity",
            "mansar auditorium seating capacity",
            "mansar lecture halls",
            "where is central library located?"
        ],
    },
    {
        "topic": "Paloura Campus (Old Campus)",
        "source": "https://www.iitjammu.ac.in/permanent-campus",
        "answer": (
            "The Paloura Campus is the original/old campus of IIT Jammu located in Paloura, Jammu. It currently houses some PhD scholars "
            "and is being developed into a high-end research facility. "
            + source_note("IIT Jammu Infrastructure & Permanent Campus Page")
        ),
        "questions": [
            "what is paloura campus?",
            "paloura campus details",
            "what ploura camopus is used for?",
            "where is iit jammu old campus located?",
            "iit jammu old campus"
        ],
    },
    {
        "topic": "Central Library",
        "source": "https://www.iitjammu.ac.in/library",
        "answer": (
            "The Central Library at IIT Jammu (located in the Mansar Building) supports students, faculty, and "
            "researchers with printed books, journals, e-books, e-journals, and digital research resources. "
            "Library facilities feature spacious reading rooms/halls, research support services, digital access systems, "
            "online academic resources, and a WiFi-enabled study environment. "
            + source_note("IIT Jammu Central Library Facilities")
        ),
        "questions": [
            "where is the library located?",
            "library facilities at iit jammu",
            "what does central library provide?",
            "is wifi available in library?",
            "central library reading rooms"
        ],
    },
    {
        "topic": "Sports Facilities",
        "source": "https://www.iitjammu.ac.in/sports",
        "answer": (
            "IIT Jammu offers extensive sports facilities to foster active student life:\n"
            "- **Indoor Sports**: Badminton, Table Tennis, Chess, Carrom, Pool/Snooker, and Squash.\n"
            "- **Outdoor Sports**: Cricket, Football, Volleyball, Basketball, cricket nets, and outdoor badminton courts.\n"
            "- **Gymnasium**: A fully equipped gymnasium with modern fitness equipment for cardio and weight training, "
            "alongside an open gym facility.\n"
            "- **Sports Culture**: Students actively participate in the Inter-IIT Sports Meet, annual sports fests, "
            "and various sports clubs and internal competitions. "
            + source_note("IIT Jammu Sports and Fitness Facilities")
        ),
        "questions": [
            "What sports facilities are available at IIT Jammu?",
            "sports facilities at iit jammu?",
            "is there a gym in iit jammu?",
            "indoor sports available in iit jammu?",
            "outdoor sports available in iit jammu?",
            "Inter-IIT sports participation iit jammu",
            "gym details at iit jammu"
        ],
    },
    {
        "topic": "Campus Facilities & Infrastructure",
        "source": "https://www.iitjammu.ac.in/facilities",
        "answer": (
            "IIT Jammu Jagti campus features premium utilities and facilities:\n"
            "- **Internet & WiFi**: The entire campus is WiFi-enabled with 1 Gbps internet connectivity, managed "
            "by the Central Computing and Communication Infrastructure (C3I).\n"
            "- **Smart Classrooms**: Modern digital classrooms with advanced teaching systems and facilities.\n"
            "- **High Performance Computing (HPC)**: The 'Agastya' HPC facility supports advanced computing and research.\n"
            "- **Central Instrumentation Facility (CIF)**: The 'Saptarshi' facility houses advanced research instruments.\n"
            "- **Medical Services**: A medical center inside campus provides 24x7 paramedic support, regular visits "
            "by senior doctors, and medical insurance for all students.\n"
            "- **Security**: Centrally secured campus with smart CCTV surveillance, security staff, and deployment "
            "control systems.\n"
            "- **Banking**: Banking facilities and an SBI ATM are available on campus. "
            + source_note("IIT Jammu General Campus Facilities & C3I")
        ),
        "questions": [
            "what general facilities are available on campus?",
            "internet speed at iit jammu",
            "what is agastya hpc?",
            "what is saptarshi central instrumentation facility?",
            "medical facilities at iit jammu?",
            "is there 24x7 doctor inside iit jammu?",
            "security on campus cctv?",
            "atm inside iit jammu campus",
            "c3i iit jammu"
        ],
    },
    {
        "topic": "PhD Student Supervisors",
        "source": "https://www.iitjammu.ac.in/ee/phd-list.html",
        "answer": (
            "IIT Jammu maintains structured lists of PhD scholars and their supervisors. Examples of research group "
            "supervision include:\n"
            "- **Dr. Badri Narayan Subudhi (Associate Professor, EE)**: Supervises Alisha Gupta (co-supervised, heart "
            "rate detection), Gokul Singh Chauhan (medical image analysis), Himanshu Singh (co-supervised, action "
            "recognition), Mahima Gandotra (co-supervised, grid-connected renewable energy), Meghna (co-supervised, "
            "underwater surveillance), Mehvish Nissar (underwater surveillance), Sarif Saleem (deep-sea video surveillance), "
            "and Zareena Amin (co-supervised, DeepFake detection).\n"
            "- **Dr. Karan Nathwani (Associate Professor, EE)**: Supervises Murtiza Ali (underwater and aerial acoustics), "
            "Pawan Kumar, Rantu Buragohain (co-supervised, EEG signal classification), Ritujoy Biswas, Tarun Bali, "
            "and Zareena Amin (co-supervised, DeepFake detection).\n"
            "- **Other Faculty**: Dr. Harpreet Kour and Dr. Harleen Kour are not registered faculty members at IIT Jammu; "
            "there are no faculty members by those names. (Please check Harkeerat Kaur for CSE-related biometrics/blockchains)."
            + source_note("IIT Jammu PhD Scholars & Research Directory")
        ),
        "questions": [
            "who are the phd students under badri narayan subudhi?",
            "phd students under badri?",
            "who are the phd students under karan nathwani?",
            "phd students under nathwani?",
            "phd students under harpreet kour?",
            "phd students under harleen kour?",
            "Zareena Amin supervisor?",
            "Murtiza Ali supervisor?"
        ],
    },
    {
        "topic": "Research Areas & Faculty Mapping",
        "source": "https://www.iitjammu.ac.in/faculty",
        "answer": (
            "IIT Jammu faculty members work on cutting-edge research topics across engineering and sciences:\n"
            "- **Machine Learning & Deep Learning**: Dr. Yamuna Prasad, Dr. Karan Nathwani, Dr. Mrinmoy Bhattacharjee, "
            "Dr. Harkeerat Kaur, Dr. Vinit Jakhetiya, Dr. Badri Narayan Subudhi, and Dr. Divyesh Varade.\n"
            "- **Drones & Drone Detection**: Dr. Karan Nathwani (developed the cutting-edge drone detection prototype "
            "selected for the IInvenTiv-2024 showcase).\n"
            "- **Quantum Computing & Cryptography**: Dr. Sumit Kumar Pandey (finite fields, quantum cryptography/computation) "
            "and Dr. Sarada Prasad Gochhayat (post-quantum cryptography).\n"
            "- **IoT (Internet of Things)**: Dr. Ravikant Saini and Dr. Sudhakar Modem.\n"
            "- **Wireless Communication**: Dr. Ankur Bansal, Dr. Ankit Dubey, Dr. Ravikant Saini, Dr. Archana Rajput, "
            "Dr. Ajay Singh, Dr. Chinmoy Kundu, and Dr. Rakesh Sharma.\n"
            "- **Audio & Speech Processing**: Dr. Karan Nathwani and Dr. Mrinmoy Bhattacharjee. "
            + source_note("IIT Jammu Faculty Directory & Research Interests")
        ),
        "questions": [
            "which professor working on ml?",
            "who is working on deep learning?",
            "who is working on drones?",
            "professors working on quantum computing",
            "research on iot",
            "who is working on wireless communication",
            "faculty working on audio processing",
            "speech signal processing professors"
        ],
    },
    {
        "topic": "Departments",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "IIT Jammu has engineering, science, and humanities departments including Computer Science "
            "and Engineering, Electrical Engineering, Mechanical Engineering, Civil Engineering, Chemical "
            "Engineering, Materials Engineering, Biosciences and Bioengineering, Mathematics, Physics, "
            "Chemistry, and Humanities and Social Sciences. Department pages should be checked for the "
            "latest faculty, labs, programmes, and HoD information. " + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "Which departments are there in IIT Jammu?",
            "iit jammu branches batao",
            "IIT Jammu me kaun kaun se departments hain?",
            "Does IIT Jammu have CSE?",
            "Does IIT Jammu have Electrical Engineering?",
            "Does IIT Jammu have Chemical Engineering?",
            "Is Biosciences available at IIT Jammu?",
            "Does IIT Jammu have HSS department?",
        ],
    },
    {
        "topic": "Campus Facilities",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "Campus facilities commonly referenced by IIT Jammu official pages include hostels, academic "
            "buildings, departmental laboratories, library facilities, medical support, sports/gym facilities, "
            "internet connectivity, and student activity spaces. For exact timing or availability, use the "
            "current official page or circular for that facility. " + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "What facilities are available on IIT Jammu campus?",
            "campus facilities batao IIT Jammu",
            "Does IIT Jammu have medical facility?",
            "Is gym available at IIT Jammu?",
            "Does campus have WiFi?",
            "Sports facilities kaise hain?",
            "Is library available at IIT Jammu?",
            "IIT Jammu me transport facility hai kya?",
        ],
    },
    {
        "topic": "Placements",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "Placement statistics, recruiter names, average CTC, median CTC and highest CTC change every "
            "placement season. The safest answer is to use IIT Jammu's official placement/training pages "
            "or latest placement brochure/circular instead of relying on old 2021-2024 numbers. "
            + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "What is the placement record of IIT Jammu?",
            "iit jammu placement kaisa hai?",
            "highest package IIT Jammu?",
            "average package CSE IIT Jammu?",
            "Which companies visit IIT Jammu?",
            "placement stats latest kaha milega?",
            "Does IIT Jammu provide internships?",
            "Can non-CS students get software jobs?",
        ],
    },
    {
        "topic": "Research",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "IIT Jammu research is spread across departments and labs. Official department pages list "
            "research areas, funded projects, publications, research labs, awards and faculty profiles. "
            "For a research match, search by department or by faculty research interests. "
            + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "What research areas are available at IIT Jammu?",
            "AI research IIT Jammu me kis ke paas hai?",
            "Which faculty works on machine learning?",
            "Which labs are available for robotics?",
            "How can I find publications of IIT Jammu faculty?",
            "I want research internship under professor.",
            "How to approach faculty for research?",
            "funded projects kaha milenge?",
        ],
    },
    {
        "topic": "Student Life",
        "source": "https://www.iitjammu.ac.in",
        "answer": (
            "Student life at IIT Jammu includes hostels, sports/gym facilities, clubs, events, student "
            "activities, academic mentoring and institute support services. Specific club names, fest "
            "dates and event schedules should be taken from the latest official student body pages or "
            "notices. " + source_note("https://www.iitjammu.ac.in")
        ),
        "questions": [
            "How is student life at IIT Jammu?",
            "campus life kaisi hai IIT Jammu?",
            "Are clubs available at IIT Jammu?",
            "coding club hai kya?",
            "tech fest kab hota hai?",
            "IIT Jammu cultural fest?",
            "Are hackathons organized?",
            "freshers ke liye clubs kaise join kare?",
        ],
    },
    {
        "topic": "HOD Directory - Civil Engineering",
        "source": "https://www.iitjammu.ac.in/civil_engineering/hod-message",
        "answer": (
            "The Head of Department (HoD) for Civil Engineering at IIT Jammu is "
            "Dr. Surendra Beniwal. "
            + source_note("https://www.iitjammu.ac.in/civil_engineering/hod-message")
        ),
        "questions": [
            "Who is the Head of Department for Civil Engineering?",
            "Who is the HOD of Civil Engineering at IIT Jammu?",
            "civil engineering hod name?",
            "Who is the head of Civil Engineering department?",
            "IIT Jammu Civil Engineering HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Computer Science and Engineering",
        "source": "https://www.iitjammu.ac.in/computer_science_engineering/message-from-deparment-hod%C2%A0",
        "answer": (
            "The Head of Department (HoD) for Computer Science and Engineering (CSE) at IIT Jammu is "
            "Dr. Yamuna Prasad. You can reach him at hod.cse@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/computer_science_engineering/message-from-deparment-hod%C2%A0")
        ),
        "questions": [
            "Who is the Head of Department for CSE?",
            "Who is the HOD of CSE at IIT Jammu?",
            "cse hod name?",
            "Who is the head of Computer Science department?",
            "IIT Jammu CSE HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Electrical Engineering",
        "source": "https://www.iitjammu.ac.in/ee/hod.html",
        "answer": (
            "The Head of Department (HoD) for Electrical Engineering at IIT Jammu is "
            "Dr. Ravikant Saini. You can reach him at hod.ee@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/ee/hod.html")
        ),
        "questions": [
            "Who is the Head of Department for Electrical Engineering?",
            "Who is the HOD of Electrical Engineering at IIT Jammu?",
            "electrical engineering hod name?",
            "Who is the head of Electrical Engineering department?",
            "IIT Jammu EE HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Mechanical Engineering",
        "source": "https://www.iitjammu.ac.in/mechanical_engineering/hod.html",
        "answer": (
            "The Head of Department (HoD) for Mechanical Engineering at IIT Jammu is "
            "Dr. B. Satya Sekhar. You can reach him at hod.me@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/mechanical_engineering/hod.html")
        ),
        "questions": [
            "Who is the Head of Department for Mechanical Engineering?",
            "Who is the HOD of Mechanical Engineering at IIT Jammu?",
            "mechanical engineering hod name?",
            "Who is the head of Mechanical Engineering department?",
            "IIT Jammu ME HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Chemical Engineering",
        "source": "https://www.iitjammu.ac.in/chemical-engineering/hod.html",
        "answer": (
            "The Head of Department (HoD) for Chemical Engineering at IIT Jammu is "
            "Dr. Ravi Kumar Arun. You can reach him at hod.chemical@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/chemical-engineering/hod.html")
        ),
        "questions": [
            "Who is the Head of Department for Chemical Engineering?",
            "Who is the HOD of Chemical Engineering at IIT Jammu?",
            "chemical engineering hod name?",
            "Who is the head of Chemical Engineering department?",
            "IIT Jammu Chemical Engineering HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Materials Engineering",
        "source": "https://iitjammu.ac.in/materials-engineering",
        "answer": (
            "The Head of Department (HoD) for Materials Engineering at IIT Jammu is "
            "Dr. Rani Rohini. "
            + source_note("https://iitjammu.ac.in/materials-engineering")
        ),
        "questions": [
            "Who is the Head of Department for Materials Engineering?",
            "Who is the HOD of Materials Engineering at IIT Jammu?",
            "materials engineering hod name?",
            "Who is the head of Materials Engineering department?",
            "IIT Jammu Materials Engineering HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Chemistry",
        "source": "https://www.iitjammu.ac.in/chemistry/message-from-head-of-the-department%C2%A0",
        "answer": (
            "The Head of Department (HoD) for Chemistry at IIT Jammu is "
            "Dr. Guru B. Ramani. "
            + source_note("https://www.iitjammu.ac.in/chemistry/message-from-head-of-the-department%C2%A0")
        ),
        "questions": [
            "Who is the Head of Department for Chemistry?",
            "Who is the HOD of Chemistry at IIT Jammu?",
            "chemistry hod name?",
            "Who is the head of Chemistry department?",
            "IIT Jammu Chemistry HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Physics",
        "source": "https://www.iitjammu.ac.in/physics/hod.html",
        "answer": (
            "The Head of Department (HoD) for Physics at IIT Jammu is "
            "Dr. Venkata Sathish Akella. You can reach him at hod.physics@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/physics/hod.html")
        ),
        "questions": [
            "Who is the Head of Department for Physics?",
            "Who is the HOD of Physics at IIT Jammu?",
            "physics hod name?",
            "Who is the head of Physics department?",
            "IIT Jammu Physics HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Mathematics",
        "source": "https://www.iitjammu.ac.in/mathematics/hod-message",
        "answer": (
            "The Head of Department (HoD) for Mathematics at IIT Jammu is "
            "Dr. Rahul Dattatraya Kitture (since July 2022). Dr. Prasant Singh has also served as HoD. "
            + source_note("https://www.iitjammu.ac.in/mathematics/hod-message")
        ),
        "questions": [
            "Who is the Head of Department for Mathematics?",
            "Who is the HOD of Mathematics at IIT Jammu?",
            "mathematics hod name?",
            "Who is the head of Mathematics department?",
            "IIT Jammu Mathematics HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Biosciences and Bioengineering",
        "source": "https://iitjammu.ac.in/bsbe",
        "answer": (
            "The Head of Department (HoD) for Biosciences and Bioengineering (BSBE) at IIT Jammu is "
            "Dr. Mithu Baidya. "
            + source_note("https://iitjammu.ac.in/bsbe")
        ),
        "questions": [
            "Who is the Head of Department for Biosciences and Bioengineering?",
            "Who is the HOD of BSBE at IIT Jammu?",
            "bsbe hod name?",
            "Who is the head of Biosciences and Bioengineering department?",
            "IIT Jammu BSBE HoD?",
        ],
    },
    {
        "topic": "HOD Directory - Humanities and Social Sciences",
        "source": "https://www.iitjammu.ac.in/hss/hod.html",
        "answer": (
            "The Head of Department (HoD) for Humanities and Social Sciences (HSS) at IIT Jammu is "
            "Dr. Amitash Ojha. You can reach him at hod.hss@iitjammu.ac.in. "
            + source_note("https://www.iitjammu.ac.in/hss/hod.html")
        ),
        "questions": [
            "Who is the Head of Department for Humanities and Social Sciences?",
            "Who is the HOD of HSS at IIT Jammu?",
            "hss hod name?",
            "Who is the head of Humanities and Social Sciences department?",
            "IIT Jammu HSS HoD?",
        ],
    },
    {
        "topic": "Hostel Leave and Outpass",
        "source": "IIT Jammu Student Affairs Office",
        "answer": (
            "To apply for hostel leave or an outpass at IIT Jammu, students must submit an application "
            "online through the student portal or contact the Student Affairs Office (Email: swoffice@iitjammu.ac.in) "
            "for necessary warden or administrative approval. "
            + source_note("IIT Jammu Student Affairs Office")
        ),
        "questions": [
            "How can I apply for hostel leave?",
            "How to apply for hostel leave?",
            "Hostel leave process at IIT Jammu?",
            "How can I request a hostel outpass?",
            "hostel outpass process?",
            "How to get permission to go out from hostel?",
        ],
    },
    {
        "topic": "MTech GATE Cutoffs",
        "source": "https://www.iitjammu.ac.in/Programme/pgadmissions",
        "answer": (
            "GATE cutoff scores and shortlisting requirements for M.Tech admissions at IIT Jammu vary each "
            "year depending on the number of applicants, vacancies, and department-specific criteria. Numerical "
            "GATE cutoffs are not fixed. For details on shortlists and criteria, check the PG Admissions portal "
            "at https://www.iitjammu.ac.in/Programme/pgadmissions or contact the Post Graduate Office at "
            "pgoffice.acad@iitjammu.ac.in. Note: JEE Advanced cutoff ranks (e.g. top 4000-5000) are only for "
            "undergraduate B.Tech admissions and do not apply to M.Tech admissions. "
            + source_note("https://www.iitjammu.ac.in/Programme/pgadmissions")
        ),
        "questions": [
            "What is the GATE cutoff for CSE at IIT Jammu?",
            "what are the cutoff marks for MTech admission?",
            "What are the GATE cutoff requirements for MTech?",
            "IIT Jammu MTech GATE cutoff?",
            "GATE cutoff marks for MTech admission?",
        ],
    },
    {
        "topic": "BTech JEE Cutoffs",
        "source": "https://www.iitjammu.ac.in/Programme/ugadmissions",
        "answer": (
            "Cutoff ranks for B.Tech admission at IIT Jammu are based on JEE Advanced ranks and are determined "
            "during JoSAA counseling. Computer Science and Engineering (CSE) is the most sought-after branch, "
            "typically closing within the top 4000-5000 ranks in JEE Advanced. Cutoffs for other branches like "
            "Civil, Chemical, or Materials Engineering can go higher (e.g., around 10,000 to 15,000+ depending "
            "on category). For the most up-to-date and batch-specific cutoff ranks, please visit JoSAA or the "
            "official IIT Jammu UG admissions page. "
            + source_note("https://www.iitjammu.ac.in/Programme/ugadmissions")
        ),
        "questions": [
            "what are the cutoff marks for BTech admission?",
            "What is the JEE Advanced cutoff for IIT Jammu?",
            "JEE cutoff for BTech CSE?",
            "IIT Jammu BTech JEE cutoff marks?",
        ],
    },
]


DEPARTMENT_PAGES = {
    "Mechanical Engineering": {
        "source": "https://iitjammu.ac.in/mechanical_engineering/faculty-list",
        "answer": (
            "The Mechanical Engineering department was established in 2016 and is one of IIT Jammu's "
            "founding departments. Official pages say it offers B.Tech., M.Tech. and Ph.D. programmes, "
            "with labs such as Fluid Mechanics, Control Engineering, Kinematics & Dynamics of Machine, "
            "Heat & Mass Transfer, Energy Systems, Solid Mechanics and Manufacturing. "
            + source_note("IIT Jammu Mechanical Engineering department page")
        ),
    },
    "Chemical Engineering": {
        "source": "https://iitjammu.ac.in/chemical-engineering/faculty-list/faculty-list.html",
        "answer": (
            "The Chemical Engineering department page says the department was established in 2018 with "
            "the first undergraduate batch and has grown with faculty, UG/PG students, research scholars, "
            "laboratories, workshops and activities. " + source_note("IIT Jammu Chemical Engineering department page")
        ),
    },
    "Physics": {
        "source": "https://iitjammu.ac.in/physics/faculty-list/index.html",
        "answer": (
            "The Physics department page highlights research-led teaching and labs such as Material "
            "Research Laboratory, Solar Research Lab, Shivalik Plasma Laboratory and Optoelectronics "
            "and Device Physics Laboratory. " + source_note("IIT Jammu Physics department page")
        ),
    },
    "Biosciences and Bioengineering": {
        "source": "https://iitjammu.ac.in/bsbe/faculty-list.html",
        "answer": (
            "The BSBE department focuses on bioengineering-based teaching, interdisciplinary research, "
            "health/environment related challenges and community outreach. Official pages list labs such "
            "as UG Bio Lab, Genetic Engineering and Tissue Culture Lab, and Nanodiagnostics and "
            "Therapeutics Lab. " + source_note("IIT Jammu BSBE department page")
        ),
    },
    "Humanities and Social Sciences": {
        "source": "https://iitjammu.ac.in/hss/faculty-list.html",
        "answer": (
            "The HSS department covers areas such as English, Economics, Sociology, Philosophy, Psychology, "
            "Cognitive Science, Culture and Polity. The department page also mentions the B.Sc. in "
            "Behavioural Science and Predictive Analytics and research groups/projects. "
            + source_note("IIT Jammu HSS department page")
        ),
    },
}


def add_variants(question: str) -> List[str]:
    q = question.strip().rstrip("?.!")
    variants = [question]
    variants.append(q.lower() + "?")
    variants.append(q.replace("IIT Jammu", "iit jammu") + "?")
    if "What is" in q:
        variants.append(q.replace("What is", "Tell me") + "?")
    if "How" in q:
        variants.append(q.replace("How", "How do I") + "?")
    if "faculty" in q.lower():
        variants.append(q + " pls?")
    return list(dict.fromkeys(variants))


def build_core_faqs() -> List[Dict[str, str]]:
    rows = []
    seen = set()
    for fact in CORE_FACTS:
        for q in fact["questions"]:
            for variant in add_variants(q):
                key = variant.lower()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "q": variant,
                    "a": fact["answer"],
                    "topic": fact["topic"],
                    "source_url": fact["source"],
                })

    for dept, data in DEPARTMENT_PAGES.items():
        questions = [
            f"Tell me about {dept} at IIT Jammu.",
            f"What programmes are offered by {dept}?",
            f"What labs are in {dept} at IIT Jammu?",
            f"{dept} department info?",
            f"{dept} IIT Jammu me kya hai?",
            f"Who should check {dept} page?",
        ]
        for q in questions:
            rows.append({
                "q": q,
                "a": data["answer"],
                "topic": "Departments",
                "source_url": data["source"],
            })
    return rows


def load_faculty() -> List[Dict]:
    if not FACULTY_FILE.exists():
        return []
    records = json.loads(FACULTY_FILE.read_text(encoding="utf-8"))
    faculty = []
    for rec in records:
        if str(rec.get("status", "1")) != "1":
            continue
        name = (rec.get("faculty_name") or "").strip()
        designation = (rec.get("designation") or "").strip()
        if not name or "professor" not in designation.lower():
            continue
        faculty.append(rec)
    return faculty


def profile_url(rec: Dict) -> str:
    slug = (rec.get("faculty_url") or "").lstrip("~")
    return f"https://iitjammu.ac.in/faculty/{slug}" if slug else "https://iitjammu.ac.in/faculty"


def faculty_answer(rec: Dict) -> str:
    sal = (rec.get("salutation") or "").strip()
    name = (rec.get("faculty_name") or "").strip()
    full_name = f"{sal} {name}".strip()
    designation = (rec.get("designation") or "").strip()
    depts = rec.get("department") or []
    dept = ", ".join(depts) if isinstance(depts, list) else str(depts)
    email = (rec.get("email") or "").strip()
    research = clean_html(rec.get("academicinterests") or rec.get("researchinterest") or "")
    publications = clean_html(rec.get("publications") or "")

    parts = [f"{full_name} is listed as {designation} in {dept} at IIT Jammu."]
    
    # Check if we have room mapping for this faculty
    norm_name = normalize_name(name)
    room_entry = FACULTY_ROOMS_MAP.get(norm_name)
    if room_entry:
        room_num = room_entry.get("room_number", "")
        if room_num:
            floor_desc = get_room_floor_info(room_num)
            parts.append(f"Their office is located in Room {room_num} on the {floor_desc} of the 11AC Academic Block (Pushkar Block).")

    if email and "@iitjammu" in email:
        parts.append(f"Email: {email}.")
    if research:
        parts.append(f"Research/academic interests: {research[:600]}.")
    if publications:
        parts.append(f"Selected publications are listed on the official profile; sample: {publications[:350]}.")
    parts.append(source_note(profile_url(rec)))
    return " ".join(parts)


def build_faculty_faqs(faculty: List[Dict]) -> List[Dict[str, str]]:
    rows = []
    extra_rows = []
    by_dept: Dict[str, List[str]] = {}
    processed_normalized_names = set()

    for rec in faculty:
        sal = (rec.get("salutation") or "").strip()
        name = (rec.get("faculty_name") or "").strip()
        full_name = f"{sal} {name}".strip()
        depts = rec.get("department") or ["IIT Jammu"]
        dept = depts[0] if isinstance(depts, list) else str(depts)
        by_dept.setdefault(dept, []).append(full_name)

        norm_name = normalize_name(name)
        processed_normalized_names.add(norm_name)

        profile_row = {
            "q": f"Who is {full_name}?",
            "a": faculty_answer(rec),
            "topic": "Faculty",
            "source_url": profile_url(rec),
        }
        rows.append(profile_row)

        for q in [
            f"What are the research interests of {full_name}?",
            f"Email of {full_name}?",
        ]:
            extra_rows.append({
                "q": q,
                "a": profile_row["a"],
                "topic": "Faculty",
                "source_url": profile_row["source_url"],
            })

        # Add room-specific queries if mapping exists
        room_entry = FACULTY_ROOMS_MAP.get(norm_name)
        if room_entry:
            names_to_try = [full_name]
            if name != full_name:
                names_to_try.append(name)
            for n in names_to_try:
                for q in [
                    f"Where is the office of {n}?",
                    f"Which room number does {n} sit in?",
                    f"What floor is {n}'s room on?",
                    f"Where is {n}'s office located?",
                    f"What is the office location of {n}?",
                ]:
                    extra_rows.append({
                        "q": q,
                        "a": profile_row["a"],
                        "topic": "Faculty",
                        "source_url": profile_row["source_url"],
                    })

    # For any faculty member in faculty_rooms.json not present in faculty_data.json:
    for norm_name, room_entry in FACULTY_ROOMS_MAP.items():
        if norm_name not in processed_normalized_names:
            fac_name = room_entry["faculty_name"]
            dept = room_entry.get("department", "IIT Jammu")
            room_num = room_entry.get("room_number", "")
            floor_desc = get_room_floor_info(room_num)
            ans = f"{fac_name} is listed in the {dept} department at IIT Jammu. Their office is located in Room {room_num} on the {floor_desc} of the 11AC Academic Block (Pushkar Block). " + source_note("IIT Jammu Academic Block Directory")
            
            profile_row = {
                "q": f"Who is {fac_name}?",
                "a": ans,
                "topic": "Faculty",
                "source_url": "https://iitjammu.ac.in/faculty"
            }
            rows.append(profile_row)
            
            names_to_try = [fac_name, f"Dr. {fac_name}", f"Prof. {fac_name}"]
            for n in names_to_try:
                for q in [
                    f"Where is the office of {n}?",
                    f"Which room number does {n} sit in?",
                    f"What floor is {n}'s room on?",
                    f"Where is {n}'s office located?",
                    f"What is the office location of {n}?",
                ]:
                    extra_rows.append({
                        "q": q,
                        "a": ans,
                        "topic": "Faculty",
                        "source_url": "https://iitjammu.ac.in/faculty"
                    })

    for dept, names in sorted(by_dept.items()):
        names = sorted(set(names))
        answer = (
            f"Faculty listed in {dept} include: {', '.join(names)}. "
            "For current additions, departures, visiting faculty or profile details, check the official "
            "IIT Jammu faculty/department page. " + source_note("https://iitjammu.ac.in/faculty")
        )
        for q in [
            f"Who are the faculty members in {dept}?",
            f"List faculty of {dept} IIT Jammu.",
            f"{dept} faculty list?",
            f"{dept} ke professors kaun hain?",
        ]:
            rows.append({
                "q": q,
                "a": answer,
                "topic": "Faculty",
                "source_url": "https://iitjammu.ac.in/faculty",
            })
    rows.extend(extra_rows)
    return rows


def write_faqs(rows: List[Dict[str, str]]) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def inject(rows: Iterable[Dict[str, str]]) -> int:
    from vectorstore.chroma_store import get_chroma_store

    chroma = get_chroma_store()
    old = chroma._collection.get(where={"doc_type": "curated_faq"})
    old_ids = old.get("ids", [])
    if old_ids:
        chroma._collection.delete(ids=old_ids)

    docs = []
    for idx, row in enumerate(rows, 1):
        text = (
            f"Curated IIT Jammu FAQ\n"
            f"Question: {row['q']}\n"
            f"Answer: {row['a']}\n"
            f"Informal search hints: pls, bro, batao, kya hai, kaise, fee kitna, faculty kaun, hostel kaisa\n"
        )
        docs.append({
            "text": text,
            "title": f"Curated FAQ {idx:03d}: {row['q'][:90]}",
            "topic": row.get("topic", "FAQ"),
            "source_url": row.get("source_url", "https://www.iitjammu.ac.in"),
            "doc_type": "curated_faq",
            "year": "2026",
        })
    return chroma.add_documents(docs, batch_size=50)


def main() -> None:
    rows = build_core_faqs()
    faculty = load_faculty()
    rows.extend(build_faculty_faqs(faculty))

    # Cap at 2000 to ensure all core FAQs and faculty room locations are included.
    rows = rows[:2000]
    write_faqs(rows)
    inserted = inject(rows)
    print(f"Wrote {len(rows)} FAQs to {OUT_FILE}")
    print(f"Inserted {inserted} curated FAQ documents into ChromaDB")


if __name__ == "__main__":
    main()
