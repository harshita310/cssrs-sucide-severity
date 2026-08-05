export const sampleDashboard = {
  inputPreview: "I cannot sleep and I feel alone",
  prediction: {
    severityLabel: 1,
    riskLevel: "Low monitored",
    confidence: 0.9044,
    probabilities: [0.02, 0.9044, 0.03, 0.02, 0.01, 0.01, 0.0056],
    modelVersion: "MentalBERT focused CE",
    validatedModel: true,
  },
  explainability: {
    positiveTokens: [
      { token: "cannot sleep", value: 0.0716, rank: 1, direction: "positive" },
    ],
    negativeTokens: [
      { token: "alone", value: -0.0104, rank: 1, direction: "negative" },
    ],
    allTokens: [],
  },
  concepts: [
    {
      name: "Insomnia",
      label: "Symptom",
      matched_alias: "cannot sleep",
      shap_value: 0.0716,
    },
    {
      name: "Isolation",
      label: "Symptom",
      matched_alias: "alone",
      shap_value: -0.0104,
    },
  ],
  evidence: [
    {
      id: "who-suicide-qa-002",
      title: "WHO Suicide Q&A",
      organization: "WHO",
      publicationYear: "Curated",
      evidenceLevel: "Clinical Guideline",
      sourceType: "WHO",
      confidence: 1,
      similarityScore: 0.748,
      snippet:
        "WHO notes that social isolation and acute emotional distress are relevant risk contexts, and encourages regular check-ins and connection to appropriate support.",
      section: "Support and effective interventions",
      sourceUrl: "https://www.who.int/news-room/questions-and-answers/item/suicide",
      citation: "World Health Organization. Suicide questions and answers.",
      supports: "Peer Support",
      mappedConcepts: ["Isolation"],
    },
    {
      id: "apa-guidance-001",
      title: "APA Suicide Risk Practice Guidance",
      organization: "APA",
      publicationYear: "Curated",
      evidenceLevel: "Clinical Guideline",
      sourceType: "APA Guideline",
      confidence: 0.6716,
      similarityScore: 0.704,
      snippet:
        "APA-oriented clinical guidance emphasizes structured assessment, therapeutic intervention, follow-up, and professional support for people experiencing suicide-related distress.",
      section: "Structured support and follow-up",
      sourceUrl: "https://psychiatry.org",
      citation: "American Psychiatric Association practice guidance.",
      supports: "Sleep Hygiene",
      mappedConcepts: ["Insomnia"],
    },
  ],
  recommendations: [
    {
      name: "Peer Support",
      score: 1.3208,
      purpose: "Connection with supportive peers, groups, or trusted contacts.",
      mappedConcepts: ["Isolation"],
      supportingEvidence: ["WHO Suicide Q&A"],
      resources: [{ name: "Support Group", resource_type: "Support Group" }],
      actionSteps: [
        "Pick one trusted person or moderated group.",
        "Send a short message asking for company or a check-in.",
        "If possible, move from text to audio or video support when isolation feels strong.",
      ],
      supportOptions: [
        "Text message to a trusted person.",
        "Audio call with a friend, peer supporter, or helpline.",
        "Video call or moderated online support group.",
      ],
    },
  ],
  graph: {
    nodes: [
      { id: "token:cannot sleep", label: "cannot sleep", type: "SHAP Token" },
      { id: "concept:Insomnia", label: "Insomnia", type: "Symptom" },
      { id: "intervention:Sleep Hygiene", label: "Sleep Hygiene", type: "Intervention" },
      { id: "evidence:apa-guidance-001", label: "APA Guidance", type: "Evidence" },
      { id: "token:alone", label: "alone", type: "SHAP Token" },
      { id: "concept:Isolation", label: "Isolation", type: "Symptom" },
      { id: "intervention:Peer Support", label: "Peer Support", type: "Intervention" },
      { id: "evidence:who-suicide-qa-002", label: "WHO Suicide Q&A", type: "Evidence" },
    ],
    edges: [
      {
        id: "token:cannot sleep->concept:Insomnia",
        source: "token:cannot sleep",
        target: "concept:Insomnia",
        label: "MAPS_TO",
      },
      {
        id: "concept:Insomnia->intervention:Sleep Hygiene",
        source: "concept:Insomnia",
        target: "intervention:Sleep Hygiene",
        label: "BENEFITS_FROM",
      },
      {
        id: "token:alone->concept:Isolation",
        source: "token:alone",
        target: "concept:Isolation",
        label: "MAPS_TO",
      },
      {
        id: "concept:Isolation->intervention:Peer Support",
        source: "concept:Isolation",
        target: "intervention:Peer Support",
        label: "BENEFITS_FROM",
      },
    ],
    trace: [
      {
        token: "alone",
        concept: "Isolation",
        intervention: "Peer Support",
        evidence: "WHO Suicide Q&A",
      },
      {
        token: "cannot sleep",
        concept: "Insomnia",
        intervention: "Sleep Hygiene",
        evidence: "APA Suicide Risk Practice Guidance",
      },
    ],
  },
  exports: { htmlReport: null, pdfReport: null },
  system: {
    postPredictionLLM: false,
    generationPolicy:
      "No LLM text generation after prediction; dashboard text is model metadata, graph data, and retrieved evidence.",
  },
};

