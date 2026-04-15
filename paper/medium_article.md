# We Built a System That Predicts How Your Brain Responds to Content Before You Hit Publish

## The $500 billion content industry has been flying blind. We used neuroscience to turn the lights on.

Every day, over 500 hours of video get uploaded to YouTube every single minute. TikTok processes millions of new videos daily. Instagram, LinkedIn, and a growing list of platforms add even more volume on top.

And yet, the way we optimize content in 2026 looks almost identical to how we did it a decade ago: create something, publish it, wait for the analytics, and iterate. The entire $500 billion digital advertising industry runs on a "spray and pray" loop. Even the best growth teams are guessing which creative will win before they commit ad spend.

We wanted to change that. So we built Neural Content Intelligence (NCI), a framework that uses Meta's latest brain encoding AI to predict how the human brain would respond to video content, entirely computationally. No lab. No fMRI scanner. No $150,000 neuromarketing study. Just your video file and a GPU.

Here's what we found.

---

## The Problem Every Growth Marketer Knows Too Well

Let's be honest about the current toolkit.

**A/B testing** is the gold standard, but it requires publishing content first. You need an audience to react before you learn anything. That means burning ad spend on the losing variants.

**Platform analytics** tell you what happened after the fact. Watch time, drop-off curves, click-through rates. All valuable, all retrospective.

**AI scoring tools** like vidIQ, TubeBuddy, and Spotter analyze your titles, thumbnails, tags, and posting times. They can tell you whether your metadata is optimized. What they cannot do is evaluate the actual content: the visuals, pacing, emotional arc, and narrative that determine whether someone watches for two seconds or two minutes.

**Traditional neuromarketing** can evaluate the content itself. Companies like Neuro-Insight and Nielsen Consumer Neuroscience have built businesses around measuring brain responses to ads. The problem? A single fMRI study costs $15,000 to $150,000, requires 20 to 40 physical subjects, and takes weeks. That simply does not scale to a team producing dozens of video assets per week.

<!-- IMAGE: figures/fig5_method_comparison.png -->
<!-- Caption: How NCI compares to existing content optimization approaches across key dimensions. Green = strength, yellow = moderate, red = weakness. -->

So there's a gap. A massive one. The industry needs a way to evaluate actual video content, before publication, at scale, with an explanation of *why* something works. Until now, that combination did not exist.

---

## What Neural Content Intelligence Actually Does

NCI is built on top of **TRIBE v2**, Meta's tri-modal brain encoding foundation model. TRIBE v2 was trained on over 1,000 hours of functional MRI brain imaging data from approximately 720 participants watching natural video content. The model learned to predict, voxel by voxel, how the human brain responds to any video stimulus.

The key insight: once trained, the model runs entirely on standard computing hardware. You feed it a video, and it outputs predicted brain activation patterns across 20,484 cortical surface points. No scanner. No subjects. Roughly two minutes of compute time per 30 to 60 second video, at a cost of about $0.10 to $1.00.

But raw voxel-level brain predictions are not useful to a marketer. That's where our framework comes in.

NCI takes those 20,484 predicted brain signals and maps them onto the **Yeo 7-network brain parcellation**, a standard neuroscience atlas that divides the brain into seven functionally distinct networks. Each network corresponds to a specific cognitive process relevant to content engagement.

The pipeline looks like this:

**Video in → TRIBE v2 brain prediction → 20,484 voxel activations → Yeo 7-network mapping → 7 interpretable engagement signals → 5 composite metrics**

The result is a complete neural profile of your content: which brain systems it activates, when those systems peak, and what that means for viewer behavior.

<!-- IMAGE: figures/comparison_dashboard.png -->
<!-- Caption: The NCI comparison dashboard showing neural profiles across five different video content types. Each video produces a distinct brain activation signature. -->

---

## The 7 Brain Signals That Drive Engagement

The Yeo 7-network parcellation has been a standard reference in neuroscience since 2011, derived from functional connectivity data across 1,000 subjects. Here is what each network tells us about content engagement.

<!-- IMAGE: figures/yeo_parcellation.png -->
<!-- Caption: The Yeo 7-network brain parcellation mapped onto the cortical surface. Each color represents a distinct functional brain network. -->

**1. Visual Salience** (Visual Network)
How visually stimulating and processing-intensive your content is. High values indicate rich, complex scenes with strong motion, color contrast, and visual detail. Product demos and cinematic content score highest here.

**2. Embodied Response** (Somatomotor Network)
The degree to which your content activates physical and motor resonance in the viewer. This network lights up for speech processing, facial expression tracking, and "feeling" what someone on screen is doing. Talking-head content and UGC reviews are dominated by this signal.

**3. Sustained Attention** (Dorsal Attention Network)
Top-down, voluntary, focused attention. When this network is active, the viewer is locked in, deliberately tracking visual elements or following a complex argument. Product demonstrations and visual content drive this signal.

**4. Surprise and Novelty Detection** (Ventral Attention Network)
The brain's salience detector. This network fires when something unexpected or particularly striking happens. It is the neural mechanism behind "hook moments" and the addictive quality of satisfying content. This signal is the single best predictor of scroll-stopping power.

**5. Emotional Resonance** (Limbic Network)
Emotional processing, affective engagement, and reward-related activation. Content that makes viewers feel something, whether empathy, excitement, or desire, drives this signal.

**6. Decision Activation** (Frontoparietal Network)
Executive function, evaluative thinking, and decision readiness. When this network peaks, the viewer is actively weighing information and considering action. This is the signal that tells you when to place your call to action.

**7. Narrative Engagement** (Default Mode Network)
Self-referential processing, story comprehension, and mental simulation. When this network is active during content viewing, the viewer is "transported" into the content, relating it to their own experience. This is the neural signature of content that feels personally relevant.

---

## 5 Metrics That Replace Guesswork

From those seven signals, NCI computes five composite engagement metrics. Each one answers a specific strategic question.

**Attention Retention Score (ARS):** Does this content hold attention across its full duration? Combines sustained attention, surprise detection, and visual processing, with a penalty for inconsistency.

**Emotional Impact Index (EII):** How strongly does this content engage emotional processing? Combines limbic activation with narrative engagement and rewards peak emotional moments.

**Hook Strength Score (HSS):** How effectively do the first one to three seconds capture attention? Prioritizes surprise/salience, visual intensity, and emotional punch in the opening moments. Scores above 0.7 suggest a strong hook. Below 0.3, your audience is scrolling past.

**CTA Activation Score (CAS):** When are viewers in a decision-making neural state? Measures frontoparietal activation at or near your intended call-to-action moment. This metric tells you whether the viewer's brain is primed to act.

**Neural Engagement Score (NES):** A single composite score across all dimensions. Most useful for ranking a batch of content assets or comparing variants head to head.

<!-- IMAGE: figures/fig3_radar_neural_profile.png -->
<!-- Caption: A radar chart showing one video's neural engagement profile across all seven brain networks. Each axis represents a different cognitive dimension. -->

---

## We Tested It on 5 Real Marketing Videos. Here's What Happened.

Theory is one thing. Results are another. We ran the NCI pipeline on five real short-form videos representing the major content archetypes used across TikTok, Instagram Reels, YouTube Shorts, and paid social.

The most striking finding: **different content formats activate fundamentally different brain systems.** We are not talking about minor variations in the same signal. We are talking about entirely different neural architectures of engagement.

This has massive implications for content strategy. Optimizing a talking-head video and optimizing a product demo require completely different approaches, because they engage different brain systems. Metadata-based tools cannot detect this distinction. NCI can.

<!-- IMAGE: figures/fig4a_comparative_radar_overlay.png -->
<!-- Caption: Neural engagement profiles for all five content types overlaid on a single radar chart. Each line represents a different video's neural "fingerprint." The separation between talking-head content and visual content is immediately visible. -->

<!-- IMAGE: figures/fig4b_comparative_radar_grid.png -->
<!-- Caption: Individual neural profiles for each video, revealing the distinct neural fingerprint of each content archetype. -->

### 1. Business Education (Talking Head, 49s)

**Dominant networks:** Somatomotor (25%) + Default Mode (18%)

The biggest insight here challenges conventional wisdom. For talking-head content, the Visual network activation was just 15%. Viewers were not visually stimulated. The Somatomotor network, which processes speech, vocal delivery, and facial expressions, dominated at 25%.

**What this means for creators:** Voice quality, delivery cadence, and facial expressiveness are the primary engagement drivers for this format. Creators investing in expensive sets and lighting may be optimizing the wrong variable entirely. The high Default Mode activation (18%) confirms that viewers are deeply processing the narrative, relating the advice to their own lives. Optimize for speaker delivery and personal relevance, not production value.

### 2. Tech/AI News Commentary (60s)

**Dominant networks:** Somatomotor (22%) + Default Mode (19%) + Frontoparietal (16%)

Similar profile to business education, but with one crucial difference: elevated Frontoparietal activation at 16%. Viewers are not just absorbing information. They are critically evaluating claims and weighing the implications.

**What this means for creators:** That elevated decision-activation signal creates natural windows for call-to-action placement. When your audience is already in an evaluative cognitive state, a CTA feels like a logical next step rather than an interruption. The strongest salience peak at 29 seconds suggests a structural break point, an ideal moment to transition toward a CTA.

### 3. UGC Product Review (Street Interview, 35s)

**Dominant networks:** Somatomotor (27%) + Default Mode (16%)

The highest Somatomotor activation of any video in the study. This video generated the strongest embodied, sensory response. Viewers were physically resonating with the content at a level the other formats could not match.

**What this means for creators:** The authenticity of UGC content has a measurable neural basis. The strong embodied response, combined with elevated Limbic activation (9%) compared to other talking-head content, suggests that the combination of genuine reaction and sensory product experience drives a uniquely powerful engagement profile. The 4-second opening hook was the strongest opening of any video tested, confirming that this format grabs attention immediately.

### 4. Product Demonstration (25s)

**Dominant networks:** Dorsal Attention (32%) + Visual (31%)

A radically different profile from the talking-head videos. Visual and Dorsal Attention networks accounted for 63% of total activation combined. The viewer's brain was locked into focused visual tracking of the product demonstration.

**What this means for creators:** This is the neural signature of "show, don't tell" content. Pure visual proof. But there is a critical caveat: Frontoparietal activation was just 5%, and Limbic was 3%. The viewer is watching intently, but not in a decision-making or emotional state. For e-commerce marketers, this means you may need a deliberate cognitive shift after the visual demonstration, perhaps a brief verbal prompt or text overlay, to move the viewer from passive observation into active evaluation before presenting a purchase CTA.

<!-- IMAGE: figures/analysis_sanitaryPadProductDemo.png -->
<!-- Caption: Full NCI analysis of the product demonstration video. Visual + Dorsal Attention account for 63% of total activation. The viewer's brain is in focused visual tracking mode, with minimal narrative, emotional, or decision-making processing. -->

### 5. Viral "Satisfying" Content (Japanese Ice Cutter, 48s)

**Dominant networks:** Visual (28%) + Dorsal Attention (27%) + Ventral Attention (17%)

The Ventral Attention activation, 17%, was the highest of any video in the analysis by a significant margin. This is the neural fingerprint of "satisfying" content: repeated surprise and salience peaks that create an addictive watch-through quality.

**What this means for creators:** The near-complete absence of Limbic (2%) and Default Mode (5%) activation tells us something important. Satisfying content does not work through emotional connection or storytelling. It is purely perceptual engagement: visual rhythm, timing of reveals, and the frequency of surprise moments. The late salience peak at 40 seconds, just 8 seconds before the end, means the viewer's brain is still in a highly activated state when the video loops on TikTok or Reels. This creates a neurally seamless transition into a replay, inflating watch-time metrics that platform algorithms reward.

<!-- IMAGE: figures/analysis_viralJapaneseIceCutter.png -->
<!-- Caption: Full NCI analysis of the viral Japanese ice cutter video. The highest Ventral Attention (surprise/salience) activation of any video reveals the neural basis of "satisfying" content's addictive quality. -->

---

## What This Means for Your Content Strategy

The proof-of-concept analysis surfaced several insights that apply across content types.

### 1. Optimize for the right brain system, not a generic "engagement" score.

Talking-head content lives in the Somatomotor and Default Mode networks (speech processing, narrative engagement). Visual content lives in the Visual and Dorsal Attention networks (focused tracking). "Satisfying" viral content adds the Ventral Attention network (surprise peaks). Each format has its own neural playbook. A single "engagement score" obscures these critical differences.

### 2. Use neural data to find your hook moments.

The Ventral Attention time course identifies the exact frames where the brain's salience detector fires most intensely. These are your strongest hook candidates. A content editor can move the highest-peak frame to the opening of a short-form video, backed by neural data rather than intuition.

<!-- IMAGE: figures/fig2_ventral_attention_peaks.png -->
<!-- Caption: Ventral Attention peaks over the video timeline. Each peak represents a moment where the brain's surprise/salience detection system fires, identifying the strongest "hook" candidates in the content. -->

### 3. Time your CTA to the decision-making window.

Frontoparietal activation tells you when the viewer is in an evaluative, decision-ready cognitive state. Placing a CTA during a Frontoparietal peak means the viewer's brain is already primed for action. Placing it during a trough means you are interrupting passive observation and asking the viewer to shift cognitive modes, a much harder conversion.

<!-- IMAGE: figures/fig1_network_timecourse.png -->
<!-- Caption: Network activation time courses over the video duration, showing how each brain network rises and falls throughout the content. The temporal resolution enables precise identification of optimal hook and CTA moments. -->

### 4. Score content before allocating budget.

A marketing team producing 20 video assets per week can run all of them through the NCI pipeline and rank them by Neural Engagement Score. Paid promotion budget, posting priority, and featured placement can be allocated to the highest-scoring content, before a single dollar of ad spend is committed.

### 5. Benchmark against competitors.

NCI can analyze any public video. Run your top-performing content and a competitor's through the same pipeline, and you get a side-by-side neural engagement comparison. You might discover that their viral content hits a surprise-peak frequency yours does not, or that their CTA timing aligns with decision-network activation while yours does not.

---

## What We Don't Know Yet (And Why That Matters)

Credibility demands honesty. Here are the limitations.

**No real-world validation yet.** This is the most critical gap. The neural profiles are theoretically grounded in two decades of neuroscience research, and the patterns we see are consistent with established science. But we have not yet correlated NCI scores with actual engagement metrics like view counts, watch time, shares, or conversions. That validation study is the immediate next priority.

**Population-level predictions only.** The model was trained on approximately 720 subjects. It predicts an "average" brain response. Individual differences in preferences, cultural background, age, and cognitive style are not captured.

**Visual only, for now.** Although TRIBE v2 supports audio input, our current implementation processes only visual information. Audio, which is a critical component of content effectiveness, is planned for integration.

The existing neuroscience literature gives us strong reason to expect that computationally predicted brain responses will carry real predictive value. Studies have repeatedly shown that neural measures from small samples (dozens of people) predict population-level outcomes (hundreds of thousands of viewers) with meaningful accuracy, sometimes outperforming self-report and expert judgment. But until we validate NCI against real-world metrics, the framework remains a proof of concept.

---

## The Shift from Reactive to Predictive

For the first time, we can provide neuroscience-grounded content evaluation at the speed and scale modern content creation demands. Two minutes of compute time. Pennies per video. Seven interpretable brain signals. Five actionable metrics. All before a single viewer sees the content.

The content industry has spent decades optimizing surfaces: titles, thumbnails, posting times, hashtags. NCI opens the door to optimizing the thing that actually matters: the cognitive experience your content creates in the viewer's brain.

The full research paper ([PDF](https://github.com/JoshW-dev/tribev2-test/blob/main/paper/neural_content_intelligence.pdf)) and open-source code are available on [GitHub](https://github.com/JoshW-dev/tribev2-test).

If you could neurally profile any piece of content before publishing it, what would you test first?

---

*Josh W. is an independent researcher working at the intersection of neuroscience and content optimization. This article is based on the paper "Neural Content Intelligence: Using Brain Encoding Models to Predict Social Media Engagement Before Publication."*

---

### Image Reference Guide (for Medium upload)

When pasting this article into Medium, upload the following images at the marked locations:

1. `figures/fig5_method_comparison.png` : Method comparison matrix
2. `figures/comparison_dashboard.png` : Hero dashboard (all 5 videos)
3. `figures/yeo_parcellation.png` : Brain network map
4. `figures/fig3_radar_neural_profile.png` : Single radar profile
5. `figures/fig4a_comparative_radar_overlay.png` : All videos overlaid
6. `figures/fig4b_comparative_radar_grid.png` : Individual radar grids
7. `figures/analysis_sanitaryPadProductDemo.png` : Product demo deep analysis
8. `figures/analysis_viralJapaneseIceCutter.png` : Viral content deep analysis
9. `figures/fig2_ventral_attention_peaks.png` : Hook moment detection
10. `figures/fig1_network_timecourse.png` : Network time courses
