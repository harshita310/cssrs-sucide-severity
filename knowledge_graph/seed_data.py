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
    },
    {
        "name": "Crisis Planning",
        "description": "Structured support planning for acute risk escalation.",
    },
    {
        "name": "CBT",
        "description": "Cognitive behavioral therapy strategies for thoughts, emotions, and behavior patterns.",
    },
    {
        "name": "Behavioral Activation",
        "description": "Planned activity scheduling to reduce withdrawal and improve mood.",
    },
    {
        "name": "Peer Support",
        "description": "Connection with supportive peers, groups, or trusted contacts.",
    },
    {
        "name": "Sleep Hygiene",
        "description": "Structured behavioral habits that support improved sleep.",
    },
    {
        "name": "Grounding Technique",
        "description": "Present-focused exercises for distress or anxiety reduction.",
    },
    {
        "name": "Mindfulness",
        "description": "Attention and awareness practices used for emotion regulation.",
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
    ("Crisis Planning", "WHO Suicide Prevention Guidance"),
    ("CBT", "APA Suicide Risk Practice Guidance"),
    ("Behavioral Activation", "Behavioral Activation Clinical Review"),
    ("Peer Support", "WHO Suicide Prevention Guidance"),
    ("Sleep Hygiene", "APA Suicide Risk Practice Guidance"),
    ("Grounding Technique", "APA Suicide Risk Practice Guidance"),
    ("Mindfulness", "APA Suicide Risk Practice Guidance"),
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
