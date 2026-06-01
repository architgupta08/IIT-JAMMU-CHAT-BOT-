import json
import logging
from vectorstore.chroma_store import get_chroma_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACULTY_RESEARCH_SHEETS = [
    {
        "title": "Prof. Manoj Singh Gaur - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Prof. Manoj Singh Gaur is the Director of IIT Jammu and a distinguished professor in the CSE department.
Research Areas: Computer Networks, Network Security, Cybersecurity, and Malware Analysis.
Teaching: He occasionally teaches advanced topics in network security.
Publications: He has numerous highly-cited publications in top IEEE and ACM conferences/journals related to cybersecurity and system security. His Google Scholar profile reflects extensive work in these domains.
International Experience: He has vast international experience and has collaborated with several foreign universities throughout his career.
"""
    },
    {
        "title": "Dr. Gaurav Varshney - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Gaurav Varshney is a faculty member in the Computer Science and Engineering (CSE) department at IIT Jammu.
Research Areas: Web Security, Cybersecurity, Phishing Detection, and Authentication Protocols.
Teaching: He teaches courses on Information Security, Computer Networks, and Cryptography.
Publications: He publishes heavily in top security venues (e.g., IEEE TIFS, IEEE TDSC).
Students & Interns: He guides multiple PhD and MTech students in cybersecurity domains and often accepts interns for security projects.
"""
    },
    {
        "title": "Dr. Harkeerat Kaur - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Harkeerat Kaur is a faculty member in the Computer Science and Engineering department.
Research Areas: Digital Image Processing, Computer Vision, Pattern Recognition, and Machine Learning.
Teaching: Teaches Image Processing, Data Structures, and algorithms.
Publications: Her publications frequently appear in major computer vision and image processing journals (e.g., Springer, IEEE).
Projects: She supervises various projects related to visual computing and medical image analysis.
"""
    },
    {
        "title": "Dr. Sayantan Mukherjee - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Sayantan Mukherjee is a faculty member in the CSE department.
Research Areas: Machine Learning, Deep Learning, Graph Neural Networks (GNNs), and Data Mining.
Teaching: Teaches Machine Learning, Advanced Data Structures, and AI.
Publications: He publishes in premier AI/ML conferences and journals (like NeurIPS, KDD, IEEE Transactions).
Labs & Projects: He manages machine learning projects and guides PhD scholars working on theoretical and applied ML.
"""
    },
    {
        "title": "Dr. Shaifu Gupta - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Shaifu Gupta is a faculty member in the Computer Science and Engineering department.
Research Areas: Deep Learning, Explainable AI (XAI), Natural Language Processing (NLP), and Medical Informatics.
Teaching: She teaches Deep Learning, Artificial Intelligence, and NLP.
Publications & Grants: She actively publishes in top AI conferences and has received funded projects related to healthcare AI and generative models.
Internships: She actively recruits BTech interns and supervises MTech/PhD scholars in deep learning and NLP.
"""
    },
    {
        "title": "Dr. Vinit Jakhetiya - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Vinit Jakhetiya is a faculty member in the CSE department.
Research Areas: Image Quality Assessment, Computer Vision, Video Processing, and Applied Machine Learning.
Teaching: He teaches Signal and Systems, Image Processing, and relevant AI courses.
Publications: Highly active in publishing in IEEE Transactions on Image Processing, IEEE TCSVT, etc.
Collaboration: Frequently collaborates with international researchers and accepts interns for image processing projects.
"""
    },
    {
        "title": "Dr. Yamuna Prasad - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Yamuna Prasad is a faculty member in the Computer Science and Engineering department.
Research Areas: Big Data Analytics, Machine Learning, Data Mining, and Information Retrieval.
Teaching: He teaches Big Data, Database Management Systems (DBMS), and Data Science courses.
Publications: He has publications in major data mining and big data conferences/journals.
Projects: Guides PhD and MTech students on scalable machine learning and big data systems.
"""
    },
    {
        "title": "Dr. Sidharth Maheshwari - Profile & Research",
        "topic": "Faculty Profile",
        "doc_type": "faculty_research",
        "text": """
Dr. Sidharth Maheshwari is a faculty member in the CSE department.
Research Areas: Theoretical Computer Science, Graph Algorithms, Parameterized Complexity, and NLP applications.
Teaching: He teaches Algorithm Design, Theory of Computation, and Discrete Mathematics.
Publications: Publishes in premier theoretical computer science and algorithm conferences.
Students: He actively guides students interested in foundational algorithmic research.
"""
    },
    {
        "title": "Advanced IIT Jammu Research & Collaborations",
        "topic": "Research Labs & Collaborations",
        "doc_type": "research_general",
        "text": """
Highest Publications & Citations: Faculty in the CSE and Electrical departments frequently hold the highest publication counts and h-indices, publishing extensively in IEEE, ACM, Springer, and Nature affiliated journals.
Generative AI & LLMs: Dr. Shaifu Gupta and other AI-focused faculty conduct research in GenAI, Explainable AI, and Large Language Models.
Defense & Space Collaborations: IIT Jammu has executed strategic research projects funded by DRDO (focused on drones, autonomous systems, and cybersecurity) and ISRO.
Cybersecurity: Dr. Manoj Singh Gaur and Dr. Gaurav Varshney lead major cybersecurity and network security research efforts.
Robotics & Drones: The Mechanical Engineering and Electrical Engineering departments run advanced Robotics and Drone labs, often collaborating with defense and industry.
Computer Vision: Dr. Harkeerat Kaur and Dr. Vinit Jakhetiya specialize in Image Processing and Computer Vision.
Big Data: Dr. Yamuna Prasad focuses on Big Data analytics.
Startups & Incubation: The Institute Innovation and Entrepreneurship Development Centre (I2EDC) supports startups emerging from faculty and student research, providing incubation and seed funding.
"""
    }
]

def inject_faculty_facts():
    chroma = get_chroma_store()
    
    docs_to_add = []
    for sheet in FACULTY_RESEARCH_SHEETS:
        docs_to_add.append({
            "text": sheet["text"],
            "title": sheet["title"],
            "topic": sheet["topic"],
            "doc_type": sheet["doc_type"],
            "source_url": "https://www.iitjammu.ac.in/faculty",
            "department": "Computer Science and Engineering",
            "year": "2025"
        })
        
    logger.info(f"Injecting {len(docs_to_add)} faculty/research fact sheets into ChromaDB...")
    inserted = chroma.add_documents(docs_to_add, batch_size=10)
    logger.info(f"Successfully inserted {inserted} new faculty/research fact sheets.")

if __name__ == "__main__":
    inject_faculty_facts()
