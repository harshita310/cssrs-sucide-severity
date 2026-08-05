"""Curated seed data for the Neo4j clinical decision support graph."""

from __future__ import annotations

SEED_CONCEPTS = [
    {
        "label": "Symptom",
        "name": "Hopelessness",
        "aliases": ["hopeless", "no hope", "hopelessness"],
        "description": "Language indicating loss of hope about the future.",
    },
    {
        "label": "Symptom",
        "name": "Worthlessness",
        "aliases": ["worthless", "burden", "useless"],
        "description": "Language indicating low self-worth or perceived burden.",
    },
    {
        "label": "Symptom",
        "name": "Isolation",
        "aliases": ["alone", "lonely", "isolated"],
        "description": "Language indicating social disconnection or loneliness.",
    },
    {
        "label": "Symptom",
        "name": "Insomnia",
        "aliases": ["cant sleep", "cannot sleep", "sleepless", "insomnia"],
        "description": "Language indicating difficulty sleeping.",
    },
    {
        "label": "Symptom",
        "name": "Self Harm",
        "aliases": ["self harm", "cut myself", "hurt myself"],
        "description": "Language indicating self-injury or self-harm behavior.",
    },
    {
        "label": "Emotion",
        "name": "Sadness",
        "aliases": ["sad", "sadness", "depressed"],
        "description": "Language indicating sadness or depressed mood.",
    },
    {
        "label": "Emotion",
        "name": "Anxiety",
        "aliases": ["anxious", "panic", "afraid"],
        "description": "Language indicating anxiety, fear, or panic.",
    },
]

SEED_INTERVENTIONS = [
    {
        "name": "Safety Planning",
        "description": "Collaborative steps to identify warning signs, coping actions, social contacts, and emergency supports.",
        "action_steps": [
            "Write down personal warning signs that show distress is increasing.",
            "List two coping actions that can be tried before contacting others.",
            "Add trusted contacts and emergency numbers to the plan.",
            "Reduce access to immediately dangerous means where possible.",
        ],
        "support_options": [
            "Text a trusted person using a prepared check-in message.",
            "Audio call a crisis helpline or trusted contact if distress rises.",
            "Video call a mental health professional for planned follow-up.",
        ],
    },
    {
        "name": "Crisis Planning",
        "description": "Structured support planning for acute risk escalation.",
        "action_steps": [
            "Identify the nearest urgent support option before distress escalates.",
            "Keep crisis contacts visible and easy to access.",
            "Do not stay alone if there is immediate danger.",
        ],
        "support_options": [
            "Audio call emergency services during immediate danger.",
            "Text or call a crisis helpline for real-time support.",
            "Ask a trusted person to stay nearby until professional help is reached.",
        ],
    },
    {
        "name": "CBT",
        "description": "Cognitive behavioral therapy strategies for thoughts, emotions, and behavior patterns.",
        "action_steps": [
            "Write the distressing thought in one sentence.",
            "Write one alternative explanation that is less self-blaming.",
            "Choose one small behavior that supports the alternative thought.",
        ],
        "support_options": [
            "Use a text worksheet for thought reframing.",
            "Discuss recurring thought patterns with a mental health professional.",
        ],
    },
    {
        "name": "Behavioral Activation",
        "description": "Planned activity scheduling to reduce withdrawal and improve mood.",
        "action_steps": [
            "Choose one low-effort activity that can be done in ten minutes.",
            "Schedule it at a specific time today.",
            "Record mood before and after the activity.",
        ],
        "support_options": [
            "Text a friend before starting the activity for accountability.",
            "Use a short audio reminder to begin the planned activity.",
        ],
    },
    {
        "name": "Peer Support",
        "description": "Connection with supportive peers, groups, or trusted contacts.",
        "action_steps": [
            "Pick one trusted person or moderated group.",
            "Send a short message asking for company or a check-in.",
            "If possible, move from text to audio or video support when isolation feels strong.",
        ],
        "support_options": [
            "Text message to a trusted person.",
            "Audio call with a friend, peer supporter, or helpline.",
            "Video call or moderated online support group.",
        ],
    },
    {
        "name": "Sleep Hygiene",
        "description": "Structured behavioral habits that support improved sleep.",
        "action_steps": [
            "Set a fixed wake-up time for tomorrow.",
            "Reduce bright-screen use before bed.",
            "Write worries down before trying to sleep.",
        ],
        "support_options": [
            "Use an audio relaxation exercise.",
            "Discuss persistent sleep difficulty with a mental health professional.",
        ],
    },
    {
        "name": "Grounding Technique",
        "description": "Present-focused exercises for distress or anxiety reduction.",
        "action_steps": [
            "Name five things you can see.",
            "Name four things you can feel physically.",
            "Take three slow breaths and repeat the exercise if needed.",
        ],
        "support_options": [
            "Use guided audio grounding.",
            "Ask a trusted person to stay on an audio call while grounding.",
        ],
    },
    {
        "name": "Mindfulness",
        "description": "Attention and awareness practices used for emotion regulation.",
        "action_steps": [
            "Notice the current emotion without judging it.",
            "Focus attention on breathing for one minute.",
            "Return attention gently when the mind wanders.",
        ],
        "support_options": [
            "Use a short guided audio practice.",
            "Join a structured mindfulness group if available.",
        ],
    },
]

SEED_EVIDENCE = [
    {
        "name": "WHO Suicide Prevention Guidance",
        "source_type": "WHO Guideline",
        "citation": "World Health Organization suicide prevention implementation guidance.",
        "url": "https://www.who.int/health-topics/suicide",
        "passage": "WHO guidance emphasizes prevention, support, and timely access to care for suicide risk.",
    },
    {
        "name": "NICE Self-harm Guideline",
        "source_type": "NICE Guideline",
        "citation": "NICE guideline for assessment and management after self-harm.",
        "url": "https://www.nice.org.uk/guidance/ng225",
        "passage": "NICE guidance supports psychosocial assessment, safety planning, and evidence-informed care after self-harm.",
    },
    {
        "name": "APA Suicide Risk Practice Guidance",
        "source_type": "APA Guideline",
        "citation": "American Psychiatric Association practice guidance for suicidal behaviors.",
        "url": "https://psychiatry.org",
        "passage": "APA guidance describes structured assessment, therapeutic intervention, and follow-up for suicide risk.",
    },
    {
        "name": "Behavioral Activation Clinical Review",
        "source_type": "Clinical Review",
        "citation": "Clinical review literature on behavioral activation for depressive symptoms.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "passage": "Behavioral activation targets withdrawal by increasing engagement with meaningful activities.",
    },
    {
        "name": "CDC Suicide Prevention Strategies",
        "source_type": "CDC",
        "citation": "Centers for Disease Control and Prevention suicide prevention strategies.",
        "url": "https://www.cdc.gov/suicide/prevention/index.html",
        "passage": "CDC prevention resources describe connectedness, coping and problem-solving skills, and safer environments as suicide prevention strategies.",
    },
    {
        "name": "SAMHSA Crisis Support Resources",
        "source_type": "SAMHSA",
        "citation": "Substance Abuse and Mental Health Services Administration crisis support resources.",
        "url": "https://www.samhsa.gov/find-help",
        "passage": "SAMHSA resources describe crisis support pathways and connection to behavioral health support services.",
    },
]

SEED_RESOURCES = [
    {
        "name": "Emergency Services",
        "resource_type": "Emergency",
        "description": "Local emergency response services for immediate danger.",
    },
    {
        "name": "Crisis Helpline",
        "resource_type": "Helpline",
        "description": "A crisis support helpline appropriate to the user's location.",
    },
    {
        "name": "Mental Health Professional",
        "resource_type": "Professional",
        "description": "A licensed mental health professional for assessment and ongoing care.",
    },
    {
        "name": "Support Group",
        "resource_type": "Support Group",
        "description": "Structured peer or community support group.",
    },
]

SEED_EVIDENCE_DOCUMENTS = [
    {
        "name": "WHO Suicide Q&A",
        "source_type": "WHO",
        "citation": "World Health Organization. Suicide questions and answers.",
        "url": "https://www.who.int/news-room/questions-and-answers/item/suicide",
        "sections": [
            {
                "title": "Support and effective interventions",
                "chunks": [
                    {
                        "chunk_id": "who-suicide-qa-001",
                        "text": "WHO describes suicide as preventable and highlights timely emotional support, health-worker contact, support groups, emergency services, and crisis lines as important support routes.",
                        "supports": ["Peer Support", "Crisis Planning", "Safety Planning"],
                    },
                    {
                        "chunk_id": "who-suicide-qa-002",
                        "text": "WHO notes that social isolation and acute emotional distress are relevant risk contexts, and encourages regular check-ins and connection to appropriate support.",
                        "supports": ["Peer Support", "Behavioral Activation"],
                    },
                ],
            }
        ],
    },
    {
        "name": "NICE Self-harm Guideline NG225",
        "source_type": "NICE Guideline",
        "citation": "NICE NG225: Self-harm assessment, management and preventing recurrence.",
        "url": "https://www.nice.org.uk/guidance/ng225/chapter/Recommendations",
        "sections": [
            {
                "title": "Safety planning and psychosocial assessment",
                "chunks": [
                    {
                        "chunk_id": "nice-ng225-001",
                        "text": "NICE recommends collaborative care and safety planning that identifies warning signs, coping strategies, social contacts, family or friends, professional contacts, emergency contacts, and ways to keep the environment safer.",
                        "supports": ["Safety Planning", "Crisis Planning"],
                    },
                    {
                        "chunk_id": "nice-ng225-002",
                        "text": "NICE recommends focusing assessment on needs, strengths, vulnerabilities, protective factors, current circumstances, and support for immediate and longer-term psychological and physical safety.",
                        "supports": ["CBT", "Safety Planning"],
                    },
                ],
            }
        ],
    },
    {
        "name": "APA Suicide Risk Practice Guidance",
        "source_type": "APA Guidance",
        "citation": "American Psychiatric Association practice guidance for suicidal behaviors.",
        "url": "https://psychiatry.org",
        "sections": [
            {
                "title": "Structured support and follow-up",
                "chunks": [
                    {
                        "chunk_id": "apa-guidance-001",
                        "text": "APA-oriented clinical guidance emphasizes structured assessment, therapeutic intervention, follow-up, and professional support for people experiencing suicide-related distress.",
                        "supports": ["CBT", "Sleep Hygiene", "Grounding Technique", "Mindfulness"],
                    }
                ],
            }
        ],
    },
    {
        "name": "Behavioral Activation Clinical Review",
        "source_type": "Clinical Review",
        "citation": "Clinical review literature on behavioral activation for depressive symptoms.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "sections": [
            {
                "title": "Activity scheduling and withdrawal",
                "chunks": [
                    {
                        "chunk_id": "ba-review-001",
                        "text": "Behavioral activation targets withdrawal by scheduling meaningful or manageable activities and monitoring the relationship between activity and mood.",
                        "supports": ["Behavioral Activation"],
                    }
                ],
            }
        ],
    },
    {
        "name": "CDC Suicide Prevention Strategies",
        "source_type": "CDC",
        "citation": "Centers for Disease Control and Prevention. Suicide prevention strategies.",
        "url": "https://www.cdc.gov/suicide/prevention/index.html",
        "sections": [
            {
                "title": "Connectedness and coping skills",
                "chunks": [
                    {
                        "chunk_id": "cdc-prevention-001",
                        "text": "CDC prevention guidance describes strengthening connectedness and improving coping and problem-solving skills as strategies that can reduce suicide risk factors.",
                        "supports": ["Peer Support", "Grounding Technique", "CBT"],
                    },
                    {
                        "chunk_id": "cdc-prevention-002",
                        "text": "CDC prevention resources include safer environments, support for people at risk, and connected community approaches as parts of a broader prevention strategy.",
                        "supports": ["Safety Planning", "Crisis Planning", "Peer Support"],
                    },
                ],
            }
        ],
    },
    {
        "name": "SAMHSA Crisis Support Resources",
        "source_type": "SAMHSA",
        "citation": "Substance Abuse and Mental Health Services Administration. Find help and treatment resources.",
        "url": "https://www.samhsa.gov/find-help",
        "sections": [
            {
                "title": "Crisis and behavioral health support",
                "chunks": [
                    {
                        "chunk_id": "samhsa-find-help-001",
                        "text": "SAMHSA describes crisis and behavioral health support resources for people who need immediate or ongoing mental health support.",
                        "supports": ["Crisis Planning", "Safety Planning", "Peer Support"],
                    },
                    {
                        "chunk_id": "samhsa-find-help-002",
                        "text": "SAMHSA find-help resources connect people to treatment locators and support services for mental health and substance use needs.",
                        "supports": ["CBT", "Behavioral Activation", "Mindfulness"],
                    },
                ],
            }
        ],
    },
]

CONCEPT_INTERVENTION_LINKS = [
    ("Hopelessness", "TREATED_BY", "CBT"),
    ("Hopelessness", "BENEFITS_FROM", "Behavioral Activation"),
    ("Hopelessness", "BENEFITS_FROM", "Safety Planning"),
    ("Worthlessness", "TREATED_BY", "CBT"),
    ("Worthlessness", "BENEFITS_FROM", "Behavioral Activation"),
    ("Isolation", "BENEFITS_FROM", "Peer Support"),
    ("Isolation", "BENEFITS_FROM", "Support Group"),
    ("Insomnia", "BENEFITS_FROM", "Sleep Hygiene"),
    ("Self Harm", "BENEFITS_FROM", "Safety Planning"),
    ("Self Harm", "BENEFITS_FROM", "Crisis Planning"),
    ("Sadness", "BENEFITS_FROM", "Behavioral Activation"),
    ("Anxiety", "BENEFITS_FROM", "Grounding Technique"),
    ("Anxiety", "BENEFITS_FROM", "Mindfulness"),
]

INTERVENTION_EVIDENCE_LINKS = [
    ("Safety Planning", "WHO Suicide Prevention Guidance"),
    ("Safety Planning", "NICE Self-harm Guideline"),
    ("Safety Planning", "CDC Suicide Prevention Strategies"),
    ("Safety Planning", "SAMHSA Crisis Support Resources"),
    ("Crisis Planning", "WHO Suicide Prevention Guidance"),
    ("Crisis Planning", "CDC Suicide Prevention Strategies"),
    ("CBT", "APA Suicide Risk Practice Guidance"),
    ("Behavioral Activation", "Behavioral Activation Clinical Review"),
    ("Peer Support", "WHO Suicide Prevention Guidance"),
    ("Peer Support", "CDC Suicide Prevention Strategies"),
    ("Peer Support", "SAMHSA Crisis Support Resources"),
    ("Sleep Hygiene", "APA Suicide Risk Practice Guidance"),
    ("Grounding Technique", "APA Suicide Risk Practice Guidance"),
    ("Grounding Technique", "CDC Suicide Prevention Strategies"),
    ("Mindfulness", "APA Suicide Risk Practice Guidance"),
    ("Crisis Planning", "SAMHSA Crisis Support Resources"),
]

INTERVENTION_RESOURCE_LINKS = [
    ("Safety Planning", "Crisis Helpline"),
    ("Safety Planning", "Mental Health Professional"),
    ("Crisis Planning", "Emergency Services"),
    ("Crisis Planning", "Crisis Helpline"),
    ("CBT", "Mental Health Professional"),
    ("Behavioral Activation", "Mental Health Professional"),
    ("Peer Support", "Support Group"),
    ("Sleep Hygiene", "Mental Health Professional"),
    ("Grounding Technique", "Crisis Helpline"),
]

SEVERITY_INTERVENTION_LINKS = [
    (0, "Peer Support"),
    (1, "Peer Support"),
    (2, "Grounding Technique"),
    (3, "CBT"),
    (4, "Safety Planning"),
    (5, "Safety Planning"),
    (5, "Crisis Planning"),
    (6, "Crisis Planning"),
    (6, "Emergency Services"),
]
