# Healthcare Doctor Recommendation AI Platform

## 1. Project Overview

### Purpose
Build an AI-powered healthcare contact center platform capable of receiving patient inquiries via voice or SMS, understanding patient symptoms and preferences, identifying the most suitable doctor, recommending appointment slots, reducing operator workload, and increasing recommendation accuracy and patient satisfaction.

## 2. Business Problems
Current process:
Patient -> Operator Interview -> Manual Database Search -> Doctor Selection -> Recommendation

Problems:
- High handling time
- Operator skill dependency
- Inconsistent recommendations
- Difficult scalability
- High operational cost

## 3. Success Criteria
- Recommendation Accuracy > 95%
- AI Resolution Rate > 80%
- Average Search Time < 3 sec
- End-to-End Response < 10 sec
- CSAT > 4.5/5
- Availability 99.9%

## 4. Data Understanding
Dataset contains:
- Hospitals
- Doctors
- Alias mappings (ENT -> Otolaryngology, Pedia -> Pediatrics)
- Availability Slots
- Ratings, Scores, Languages, Personality attributes

Doctor matching dimensions:
- Specialty
- Region
- Availability
- Personality
- Language
- Doctor Rating
- Doctor Score
- Hospital Rating

## 5. Functional Requirements
### Inputs
- Voice Calls
- SMS
- Web Chat

### Outputs
- Recommended doctors
- Available appointment slots
- Confidence score

## 6. High Level Architecture
Voice/SMS Gateway -> STT -> AI Agent Gateway -> Search Service -> PostgreSQL + pgvector -> Audit System

## 7. AI Agent Architecture

- Language: Python 3.13
- Framework: LangChain
- LLM: Claude Haiku 4.5

### Agent 1: Intake Agent
Collects patient information.

### Agent 2: Medical Classification Agent
Maps symptoms to specialties.

### Agent 3: Search Agent
Builds structured search query.

### Agent 4: Ranking Agent
Ranking Formula:
- 35% Specialty Match
- 20% Availability
- 15% Region Match
- 10% Language Match
- 10% Rating
- 5% Hospital Rating
- 5% Personality Match

### Agent 5: Response Agent
Generates recommendation response.

## 8. Search Architecture
Hybrid Search:
- Structured SQL Search
- Vector Search
- Business Re-ranking

Top-K Strategy:
- Retrieve 20
- Return Top 3

## 9. Database Design
### Core Tables
- hospital
- doctor
- doctor_schedule
- keyword_alias
- reservation
- contact_log
- audit_log
- operator

### Hospital Table
- hospital_id
- name
- region
- rating
- address
- internal_number

### Doctor Table
- doctor_id
- name
- gender
- age
- expertise
- hospital_id
- region
- language
- rating
- score
- personality
- register_date
- leave_date

### Doctor Schedule Table
- schedule_id
- doctor_id
- available_time
- status

### Reservation Table
- reservation_id
- patient_id_hash
- doctor_id
- slot_time
- status

## 10. Network Design
Internet
-> WAF / Firewall
-> Load Balancer
-> AI Gateway + API Gateway
-> Internal Network
-> PostgreSQL / pgvector / Redis
-> Backup Cluster

Security Zones:
- DMZ
- Application Zone
- Data Zone

## 11. Security Design
- Azure AD
- OIDC
- MFA
- TLS 1.3
- AES-256 Encryption
- Vault-based Secret Management
- Prompt Injection Protection

## 12. Reliability Design
- 2 API Nodes
- 2 AI Nodes
- 3 PostgreSQL Nodes
- Hourly WAL Backup
- Daily Full Backup
- 90 Day Retention

## 13. Data Ingestion Pipeline
JSON -> Schema Validation -> Normalization -> Deduplication -> PostgreSQL -> Embeddings -> pgvector

## 14. Infrastructure Recommendation
### POC
- Docker Compose
- PostgreSQL
- Backend API
- LLM Service

### Production
- Kubernetes
- Ingress
- API Service
- AI Service
- PostgreSQL HA
- Redis
- Monitoring

## 15. Observability
- Prometheus
- Grafana
- Loki

Metrics:
- Search Latency
- LLM Latency
- Token Cost
- Recommendation Accuracy
- Reservation Rate
- CSAT

## 16. Future Enhancements
- Appointment Booking Integration
- Calendar Synchronization
- Multilingual Expansion
- RAG Medical Knowledge Base
- Patient Personalization
- Autonomous Scheduling Agent

## Principal Engineer Review Conclusion
This architecture separates AI reasoning from data access, implements scalable multi-agent design, uses PostgreSQL + pgvector hybrid retrieval, supports on-premise deployment, provides enterprise security controls, and is designed for healthcare-grade reliability and maintainability.
