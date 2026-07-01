# Graph Report - tribev2-test  (2026-05-30)

## Corpus Check
- 23 files · ~52,111 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 265 nodes · 318 edges · 24 communities (12 shown, 12 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `16d6adbd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]

## God Nodes (most connected - your core abstractions)
1. `ndarray` - 14 edges
2. `Neural Content Intelligence` - 13 edges
3. `Neural Content Intelligence: Using Brain Encoding Models to Predict Social Media Engagement Before Publication` - 13 edges
4. `load_npy()` - 11 edges
5. `Meta Built an AI That Simulates Your Brain. I Used It to Decode What Makes Content Go Viral.` - 11 edges
6. `str` - 10 edges
7. `run_inference()` - 10 edges
8. `save_results()` - 9 edges
9. `build_ui()` - 9 edges
10. `PROMPT 1: Research for the Paper` - 9 edges

## Surprising Connections (you probably didn't know these)
- `build_references_section()` --calls--> `render()`  [INFERRED]
  paper/build_latex.py → render_landing_brains.py

## Import Cycles
- None detected.

## Communities (24 total, 12 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (58): float, ndarray, str, Figure, int, _build_ai_prompt(), _build_mesh_cache(), build_ui() (+50 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (39): 10. Conclusion, 1.1 The Content Optimization Problem, 1.2 The Neuroscience Opportunity, 1.3 The Brain Encoding Model Breakthrough, 1. Introduction, 2.1 Neural Prediction of Market Outcomes, 2.2 Neuromarketing Industry Landscape, 2.3 Brain Encoding Models (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (30): 1. Executive Summary, 2.1 The Content Optimization Problem, 2.2 The Neuroscience Opportunity, 2.3 The Brain Encoding Model Breakthrough, 2. Introduction, 3.1 A Brief History of Neuromarketing, 3.2 Brain Encoding Models: From Regression to Deep Prediction, 3.3 TRIBE v2: The State of the Art (+22 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (17): 1. Business Education (Talking Head, 49s), 2. Tech/AI News Commentary (60s), 3. UGC Product Review (Street Interview, 35s), 4. Product Demonstration (25s), 5. Viral "Satisfying" Content (Japanese Ice Cutter, 48s), Five Metrics Built on Top, Image Reference Guide (for Medium upload), Limitations (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (17): 1. Neural Prediction of Market Outcomes & Ad Effectiveness, 1. Where to Publish — Venue Options, 2. Brain Networks and Content Engagement, 2. The Publication Process — Step by Step, 3. Making It Credible Without Academic Affiliation, 3. TRIBE v2 and Brain Encoding Models, 4. Current Social Media Content Optimization Landscape, 4. Maximizing Impact (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (16): 5 Engagement Metrics, Deep-Dive Analysis Examples, How It Works, Installation, Key Numbers, License, Limitations, Neural Content Intelligence (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (15): build(), build_references_section(), _escape_ref_text(), md_to_latex(), str, Convert the paper markdown to arXiv-ready LaTeX source., Convert markdown to LaTeX body text. Returns (latex_body, references)., Escape LaTeX specials in reference text, leaving URLs alone. (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (11): 1. Pull latest code, 2. Sanity check on one video, 3. Run on all videos, 4. Regenerate figures, 5. Update the paper, 6. Delete the test script, Peak Detection Fix — Desktop Next Steps, Problem (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (7): extract_key_frames(), extract_thumbnail(), find_robust_peaks(), Generate all paper figures using the real content videos (not bears)., Find peaks with detrending and asymmetric boundary trimming.      Start trim is, Extract a single frame from a video., Extract frames at specific timestamps.

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (6): 5.1 Attention Retention Score (ARS), 5.2 Emotional Impact Index (EII), 5.3 Hook Strength Score (HSS), 5.4 CTA Activation Score (CAS), 5.5 Neural Engagement Score (NES), 5. Proposed Engagement Metrics

### Community 10 - "Community 10"
Cohesion: 0.50
Nodes (3): find_robust_peaks(), Generate all paper figures from cached TRIBE v2 results., Find peaks with detrending and asymmetric boundary trimming.      Start trim is

### Community 17 - "Community 17"
Cohesion: 0.67
Nodes (3): bool, on_play_tick(), Auto-advance the slider by 1 step when playing.

## Knowledge Gaps
- **117 isolated node(s):** `setup.sh script`, `str`, `ndarray`, `float`, `bool` (+112 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Neural Content Intelligence: Using Brain Encoding Models to Predict Social Media Engagement Before Publication` connect `Community 1` to `Community 9`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `Visualize TRIBE v2 predictions on the fsaverage5 cortical surface.`, `setup.sh script`, `str` to the rest of the system?**
  _165 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07481005260081823 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._