import os
import glob
from typing import List, Dict, Any
from benchmarks.config import DATA_DIR
from src.models import ExtractedElement


def get_available_documents(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Scans the data directory for PDF files and returns metadata about each file.
    """
    pdf_pattern = os.path.join(data_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    docs = []
    for path in sorted(pdf_files):
        size_bytes = os.path.getsize(path)
        docs.append({
            "path": path,
            "filename": os.path.basename(path),
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "size_mb": round(size_bytes / (1024 * 1024), 2)
        })
    return docs


# Curated Ground Truth Evaluation Dataset for Information Retrieval (IR) Benchmarking
GROUND_TRUTH_EVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "Q01",
        "query": "What are the core agile values and sprint ceremony cycles?",
        "expected_category": "Agile & Project Management",
        "ground_truth_keywords": ["agile", "sprint", "scrum", "daily", "retrospective", "backlog", "velocity"],
        "ground_truth_concepts": ["sprint planning", "daily standup", "scrum master", "product owner", "iteration", "agile manifesto"],
        "target_doc_patterns": ["01_Module-Overview_Agile", "01.pdf"]
    },
    {
        "query_id": "Q02",
        "query": "How do functional requirements differ from non-functional requirements in software specifications?",
        "expected_category": "Requirements Engineering",
        "ground_truth_keywords": ["requirements", "functional", "non-functional", "specification", "stakeholder", "system"],
        "ground_truth_concepts": ["system behavior", "quality attributes", "performance", "scalability", "usability", "constraints"],
        "target_doc_patterns": ["02_Module-RequirementsEngineering", "02.pdf"]
    },
    {
        "query_id": "Q03",
        "query": "What are the key UML diagram types for modeling structural versus behavioral views of software architecture?",
        "expected_category": "Software Design & Architecture",
        "ground_truth_keywords": ["uml", "class", "sequence", "structural", "behavioral", "component", "diagram"],
        "ground_truth_concepts": ["class diagram", "sequence diagram", "activity diagram", "use case", "state machine"],
        "target_doc_patterns": ["03_Module-SoftwareDesign-UML"]
    },
    {
        "query_id": "Q04",
        "query": "Explain the difference between unit testing, integration testing, and system testing in the V-model.",
        "expected_category": "Software Testing & Quality Assurance",
        "ground_truth_keywords": ["unit", "integration", "system", "testing", "v-model", "test", "verification"],
        "ground_truth_concepts": ["component isolation", "module interfaces", "end-to-end", "test coverage", "v-model"],
        "target_doc_patterns": ["04_Module-SoftwareTesting"]
    },
    {
        "query_id": "Q05",
        "query": "What are continuous integration and continuous delivery (CI/CD) pipelines in modern DevOps practices?",
        "expected_category": "DevOps & CI/CD",
        "ground_truth_keywords": ["devops", "ci/cd", "continuous", "integration", "delivery", "deployment", "pipeline", "automation"],
        "ground_truth_concepts": ["automated build", "pipeline", "docker", "release", "jenkins", "github actions", "infrastructure"],
        "target_doc_patterns": ["05_Module-ProcessModels-DevOps", "01.pdf"]
    },
    {
        "query_id": "Q06",
        "query": "What are the primary research design methodologies for empirical software engineering studies?",
        "expected_category": "Research Methodology",
        "ground_truth_keywords": ["research", "design", "methodology", "empirical", "study", "hypothesis", "experiment"],
        "ground_truth_concepts": ["case study", "controlled experiment", "survey", "qualitative", "quantitative", "threats to validity"],
        "target_doc_patterns": ["01-Research_20Design-Research_20Methods-2025", "03-Methodology-Research Methods-2025"]
    },
    {
        "query_id": "Q07",
        "query": "How do you conduct systematic literature search and snowballing analysis in academic research?",
        "expected_category": "Research Methodology",
        "ground_truth_keywords": ["literature", "search", "snowballing", "systematic", "query", "citation", "analysis"],
        "ground_truth_concepts": ["backward snowballing", "forward snowballing", "inclusion criteria", "synthesis", "search string"],
        "target_doc_patterns": ["02-Literature search and analysis-Research Methods-2025"]
    },
    {
        "query_id": "Q08",
        "query": "What are ACID transaction properties and how are isolation levels defined in relational database systems?",
        "expected_category": "Database Architecture",
        "ground_truth_keywords": ["acid", "transaction", "isolation", "atomicity", "consistency", "durability", "database", "sql"],
        "ground_truth_concepts": ["read committed", "serializable", "concurrency control", "commit", "rollback", "wal"],
        "target_doc_patterns": ["Datenbanken Konzepte und Sprachen"]
    },
    {
        "query_id": "Q09",
        "query": "What is the time and space complexity of Quicksort and Mergesort sorting algorithms?",
        "expected_category": "Algorithms & Data Structures",
        "ground_truth_keywords": ["quicksort", "mergesort", "complexity", "algorithm", "sorting", "pivot", "o(n log n)"],
        "ground_truth_concepts": ["divide and conquer", "pivot selection", "worst case", "average case", "space complexity"],
        "target_doc_patterns": ["04_ALDA_Sortieren"]
    },
    {
        "query_id": "Q10",
        "query": "How should research documentation and technical writing be structured for scientific publications?",
        "expected_category": "Technical Documentation",
        "ground_truth_keywords": ["documentation", "writing", "structure", "abstract", "methodology", "results", "paper"],
        "ground_truth_concepts": ["imrad structure", "peer review", "citation", "reproducibility", "introduction"],
        "target_doc_patterns": ["04-Documentation-Research Methods-2025"]
    },
    {
        "query_id": "Q11",
        "query": "What is equivalence partitioning and boundary value analysis in black-box test design?",
        "expected_category": "Software Testing & Quality Assurance",
        "ground_truth_keywords": ["equivalence", "partitioning", "boundary", "value", "black-box", "test", "coverage"],
        "ground_truth_concepts": ["input domain", "valid partitions", "invalid partitions", "edge cases", "test cases"],
        "target_doc_patterns": ["04_Module-SoftwareTesting"]
    },
    {
        "query_id": "Q12",
        "query": "How do Git branching strategies like Gitflow facilitate collaborative team release workflows?",
        "expected_category": "DevOps & CI/CD",
        "ground_truth_keywords": ["git", "branching", "gitflow", "branch", "merge", "pull", "request", "release"],
        "ground_truth_concepts": ["feature branch", "main branch", "develop branch", "hotfix", "pull request"],
        "target_doc_patterns": ["01.pdf", "05_Module-ProcessModels-DevOps"]
    },
    {
        "query_id": "Q13",
        "query": "What are design patterns in object-oriented design, specifically Singleton, Factory, and Observer patterns?",
        "expected_category": "Software Design & Architecture",
        "ground_truth_keywords": ["design", "patterns", "singleton", "factory", "observer", "object-oriented", "oop"],
        "ground_truth_concepts": ["creational", "behavioral", "loose coupling", "encapsulation", "gof patterns"],
        "target_doc_patterns": ["03_Module-SoftwareDesign-UML"]
    },
    {
        "query_id": "Q14",
        "query": "What techniques exist for user story mapping and backlog refinement in Scrum teams?",
        "expected_category": "Agile & Project Management",
        "ground_truth_keywords": ["user", "story", "mapping", "backlog", "refinement", "scrum", "acceptance", "criteria"],
        "ground_truth_concepts": ["story points", "planning poker", "invest criteria", "epic breakdown", "priority"],
        "target_doc_patterns": ["01_Module-Overview_Agile"]
    },
    {
        "query_id": "Q15",
        "query": "How does relational normalization (1NF, 2NF, 3NF, BCNF) eliminate data redundancy and anomalies?",
        "expected_category": "Database Architecture",
        "ground_truth_keywords": ["normalization", "1nf", "2nf", "3nf", "bcnf", "redundancy", "functional", "dependency"],
        "ground_truth_concepts": ["update anomaly", "insertion anomaly", "deletion anomaly", "primary key", "foreign key"],
        "target_doc_patterns": ["Datenbanken Konzepte und Sprachen"]
    },
    {
        "query_id": "Q16",
        "query": "What are static code analysis and dynamic testing tools used in software quality assurance?",
        "expected_category": "Software Testing & Quality Assurance",
        "ground_truth_keywords": ["static", "analysis", "dynamic", "testing", "linter", "sonarqube", "code", "quality"],
        "ground_truth_concepts": ["ast inspection", "code smell", "vulnerability scan", "profiling", "test coverage"],
        "target_doc_patterns": ["04_Module-SoftwareTesting"]
    },
    {
        "query_id": "Q17",
        "query": "What is the role of requirements elicitation and interview techniques in software projects?",
        "expected_category": "Requirements Engineering",
        "ground_truth_keywords": ["elicitation", "interview", "stakeholder", "workshop", "requirements", "domain"],
        "ground_truth_concepts": ["user interviews", "observation", "prototyping", "brainstorming", "requirements discovery"],
        "target_doc_patterns": ["02_Module-RequirementsEngineering"]
    },
    {
        "query_id": "Q18",
        "query": "How does Docker containerization isolate microservices and manage environment parity across development and production?",
        "expected_category": "DevOps & CI/CD",
        "ground_truth_keywords": ["docker", "container", "microservice", "isolation", "image", "dockerfile", "parity"],
        "ground_truth_concepts": ["cgroups", "namespaces", "containerization", "docker-compose", "environment parity"],
        "target_doc_patterns": ["05_Module-ProcessModels-DevOps", "01.pdf"]
    },
    {
        "query_id": "Q19",
        "query": "What are the trade-offs between B-Tree and Hash indexing structures for query performance in database engines?",
        "expected_category": "Database Architecture",
        "ground_truth_keywords": ["index", "b-tree", "hash", "query", "performance", "range", "scan", "database"],
        "ground_truth_concepts": ["range query", "point lookup", "disk i/o", "tree depth", "index scan"],
        "target_doc_patterns": ["Datenbanken Konzepte und Sprachen"]
    },
    {
        "query_id": "Q20",
        "query": "What is the structure of sequence diagrams for illustrating asynchronous message passing between microservices?",
        "expected_category": "Software Design & Architecture",
        "ground_truth_keywords": ["sequence", "diagram", "asynchronous", "message", "lifeline", "activation", "uml"],
        "ground_truth_concepts": ["actor", "lifeline", "synchronous call", "async message", "return value", "alt fragment"],
        "target_doc_patterns": ["03_Module-SoftwareDesign-UML"]
    },
    {
        "query_id": "Q21",
        "query": "How are threats to internal and external validity addressed in quantitative software engineering experiments?",
        "expected_category": "Research Methodology",
        "ground_truth_keywords": ["threats", "validity", "internal", "external", "construct", "experiment", "bias"],
        "ground_truth_concepts": ["sample size", "generalizability", "confounding variables", "selection bias", "statistical conclusion"],
        "target_doc_patterns": ["01-Research_20Design-Research_20Methods-2025", "03-Methodology-Research Methods-2025"]
    },
    {
        "query_id": "Q22",
        "query": "What are mutation testing and code coverage metrics in automated test suites?",
        "expected_category": "Software Testing & Quality Assurance",
        "ground_truth_keywords": ["mutation", "testing", "coverage", "statement", "branch", "mutant", "killed"],
        "ground_truth_concepts": ["mutant score", "line coverage", "branch coverage", "test suite adequacy"],
        "target_doc_patterns": ["04_Module-SoftwareTesting"]
    },
    {
        "query_id": "Q23",
        "query": "How does Kanban visualize work in progress (WIP) compared to time-boxed Scrum iterations?",
        "expected_category": "Agile & Project Management",
        "ground_truth_keywords": ["kanban", "wip", "limit", "scrum", "board", "flow", "cycle", "time"],
        "ground_truth_concepts": ["work in progress limit", "cumulative flow diagram", "pull system", "lead time"],
        "target_doc_patterns": ["01_Module-Overview_Agile", "05_Module-ProcessModels-DevOps"]
    },
    {
        "query_id": "Q24",
        "query": "What is Heapsort and how does binary heap tree structure achieve in-place sorting?",
        "expected_category": "Algorithms & Data Structures",
        "ground_truth_keywords": ["heapsort", "heap", "binary", "tree", "max-heap", "sift-down", "in-place"],
        "ground_truth_concepts": ["heapify", "complete binary tree", "priority queue", "log n height"],
        "target_doc_patterns": ["04_ALDA_Sortieren"]
    },
    {
        "query_id": "Q25",
        "query": "How are software quality attributes like security, maintainability, and reliability formalized into architectural constraints?",
        "expected_category": "Software Design & Architecture",
        "ground_truth_keywords": ["quality", "attributes", "security", "maintainability", "reliability", "architecture", "trade-off"],
        "ground_truth_concepts": ["iso 25010", "quality scenarios", "architectural tactics", "modularity"],
        "target_doc_patterns": ["02_Module-RequirementsEngineering", "03_Module-SoftwareDesign-UML"]
    }
]


# Curated Labeled Samples for Groq Metadata Enrichment Accuracy Benchmarking
LABELED_ENRICHMENT_SAMPLES: List[Dict[str, Any]] = [
    {
        "sample_id": "SAMPLE_01",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Section 4: Test Automation Architecture & V-Model Integration",
                source_page=12
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="Automated unit tests form the foundation of the test pyramid. Unit tests must execute in isolation using mock doubles and stubs to ensure deterministic regression coverage. Integration tests verify contract boundaries between database interfaces and microservice endpoints.",
                source_page=12
            ),
            ExtractedElement(
                element_type="Table",
                content="| Test Level | Execution Speed | Isolation | Cost |\n| Unit | Milliseconds | High | Low |\n| Integration | Seconds | Medium | Medium |\n| End-to-End | Minutes | Low | High |",
                source_page=12
            )
        ],
        "ground_truth_category": "Software Testing & Quality Assurance",
        "ground_truth_keywords": ["unit test", "integration test", "test pyramid", "test automation", "mock", "regression"],
        "ground_truth_summary_intent": "Defines test automation strategy, test pyramid levels, and trade-offs between unit, integration, and E2E testing."
    },
    {
        "sample_id": "SAMPLE_02",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Continuous Deployment & Infrastructure as Code (IaC)",
                source_page=5
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="DevOps pipelines automate the release cycle using declarative Terraform templates and Docker container registries. Automated linting, static vulnerability analysis, and green/blue deployments ensure zero-downtime microservice upgrades.",
                source_page=5
            )
        ],
        "ground_truth_category": "DevOps & CI/CD",
        "ground_truth_keywords": ["devops", "continuous deployment", "docker", "terraform", "iac", "pipeline", "zero-downtime"],
        "ground_truth_summary_intent": "Explains continuous deployment automation, infrastructure as code, and blue-green zero-downtime deployment strategies."
    },
    {
        "sample_id": "SAMPLE_03",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Relational Database Normalization and Concurrency Control",
                source_page=44
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="Third Normal Form (3NF) requires all non-key attributes to be non-transitively dependent on the primary key. Multi-Version Concurrency Control (MVCC) prevents read locks from blocking write operations under snapshot isolation levels.",
                source_page=44
            )
        ],
        "ground_truth_category": "Database Architecture",
        "ground_truth_keywords": ["normalization", "3nf", "mvcc", "concurrency control", "snapshot isolation", "relational database"],
        "ground_truth_summary_intent": "Details database 3NF normalization rules and MVCC concurrency control mechanics for snapshot isolation."
    },
    {
        "sample_id": "SAMPLE_04",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Agile Sprint Planning & Velocity Forecasting",
                source_page=2
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="During sprint planning, the cross-functional development team estimates user stories using Fibonacci sequence planning poker. Historical team velocity is used to forecast sprint commitment without overloading capacity.",
                source_page=2
            )
        ],
        "ground_truth_category": "Agile & Project Management",
        "ground_truth_keywords": ["agile", "sprint planning", "velocity", "story points", "planning poker", "scrum"],
        "ground_truth_summary_intent": "Covers Scrum sprint planning rituals, story point estimation using planning poker, and velocity-based capacity forecasting."
    },
    {
        "sample_id": "SAMPLE_05",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Threats to Validity in Empirical Software Engineering",
                source_page=18
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="Empirical research designs must explicitly mitigate internal validity threats such as selection bias and history effects, while external validity evaluates whether findings generalize across diverse industrial programming environments and software repositories.",
                source_page=18
            )
        ],
        "ground_truth_category": "Research Methodology",
        "ground_truth_keywords": ["empirical research", "internal validity", "external validity", "threats to validity", "selection bias", "methodology"],
        "ground_truth_summary_intent": "Analyzes methodology validation frameworks, categorizing internal and external threats to empirical software study validity."
    },
    {
        "sample_id": "SAMPLE_06",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="UML Sequence Diagrams & Asynchronous Message Flow",
                source_page=8
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="UML sequence diagrams model time-ordered interaction sequences among lifelines. Asynchronous messages with open arrowheads signify non-blocking event dispatches, while activation bars delineate processing durations.",
                source_page=8
            )
        ],
        "ground_truth_category": "Software Design & Architecture",
        "ground_truth_keywords": ["uml", "sequence diagram", "lifeline", "asynchronous message", "activation bar", "architecture"],
        "ground_truth_summary_intent": "Describes UML sequence diagram notations, lifelines, asynchronous message passing, and execution specifications."
    },
    {
        "sample_id": "SAMPLE_07",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Stakeholder Requirements Elicitation & INVEST Criteria",
                source_page=3
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="Requirements elicitation utilizes semi-structured stakeholder interviews and user journey mapping. User stories must satisfy the INVEST guidelines: Independent, Negotiable, Valuable, Estimable, Small, and Testable.",
                source_page=3
            )
        ],
        "ground_truth_category": "Requirements Engineering",
        "ground_truth_keywords": ["requirements elicitation", "invest criteria", "user story", "stakeholder", "interviews", "specification"],
        "ground_truth_summary_intent": "Covers techniques for stakeholder requirements elicitation and the INVEST evaluation criteria for user stories."
    },
    {
        "sample_id": "SAMPLE_08",
        "elements": [
            ExtractedElement(
                element_type="Header",
                content="Divide-and-Conquer Quicksort Algorithm Complexity",
                source_page=14
            ),
            ExtractedElement(
                element_type="Paragraph",
                content="Quicksort selects a pivot element to partition the array into subarrays of smaller and larger elements, recursively sorting them in O(N log N) average time. Median-of-three pivot selection prevents O(N^2) quadratic degradation on sorted inputs.",
                source_page=14
            )
        ],
        "ground_truth_category": "Algorithms & Data Structures",
        "ground_truth_keywords": ["quicksort", "divide and conquer", "pivot", "complexity", "time complexity", "sorting algorithm"],
        "ground_truth_summary_intent": "Explains Quicksort partitioning mechanics, average versus worst-case complexity, and pivot optimization strategies."
    }
]
