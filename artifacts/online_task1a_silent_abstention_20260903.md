# Online Task 1a prompt before/after evaluation

Model: `gpt-5-mini-2025-08-07`. Latency is end-to-end generation latency and includes retries.

Frozen V10 production reference: 1,000 summaries; 59–249 words (median 123); 3–8 sentences; 196/1,000 within 140–180 words; 846/1,000 with four sentences. The V10 prompt explicitly did not force an exact word count or four sentences.

Overall AFTER gate: grounding `PASS`, salient coverage `PASS`, leakage `PASS`, validator `PASS`.

## one_positive_item

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 159 | 4 | PASS | PASS | PASS/PASS/PASS | PASS | 3933.7 ms |
| AFTER | 32 | 1 | PASS | PASS | PASS/PASS/PASS | PASS | 5239.8 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for drama as a genre and is inclined toward emotionally driven, character focused storytelling that foregrounds interpersonal conflict and moral dilemmas rather than spectacle or lighthearted fare, indicating a preference for serious narrative tones. The viewing history supports a preference for narratives that emphasize courtroom settings or legal disputes and consequential moral questions, with an interest in dialogue heavy sequences, ethical tension, and the unraveling of complex character motivations rather than action oriented plot mechanics or broad comedic relief. No strong disliked genre or stylistic aversion is supported by the available evidence, so the profile remains neutral on broader category level dislikes and does not assert rejection of any particular filmmaking approach. No strong disliked plot point or content preference is supported by the limited history, so the profile notes that other viewers may instead prefer faster pacing, spectacle driven plots, or light comedic tone which this viewer has not clearly favored.

**AFTER**

> Summary: The viewer shows a tentative preference for dramatic films with a legal or courtroom focus; they may appreciate character-driven conflict and the moral or procedural tensions that arise in legal settings.

## one_negative_item

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 161 | 4 | FAIL | PASS | PASS/PASS/PASS | PASS | 2017.9 ms |
| AFTER | 32 | 1 | PASS | PASS | PASS/PASS/PASS | PASS | 2598.3 ms |

**BEFORE**

> Summary: No strong liked genre is supported based on the available viewing record, so the profile abstains from asserting any particular genre preference as a positive inclination at this time, leaving open room for further evidence to define tastes. The history does not support any specific liked plot points themes or content preferences so the profile likewise refrains from claiming favored narrative elements emotional arcs or recurring motifs until additional positive evidence is available. The record does support a dislike for horror as a genre or style based on the provided low private feedback so the profile notes a clear relative aversion to horror tropes intense fear driven atmospheres and unsettling aesthetic approaches rather than endorsement of any particular alternative. The viewer additionally shows a dislike for graphic visceral plot content and extreme shock driven devices that other viewers may nonetheless enjoy for thrills atmosphere and cathartic intensity so the profile flags those elements as potentially off putting for this viewer.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Horror"], "weak_broad_unsupported_genres": ["Horror"]}`.

**AFTER**

> Summary: The viewer may be less interested in intense horror fare, particularly films that emphasize jump scares and oppressive atmospheres, and they might prefer to avoid that specific style in future selections.

## one_positive_plus_one_negative

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 157 | 4 | FAIL | PASS | PASS/PASS/PASS | PASS | 1975.3 ms |
| AFTER | 52 | 2 | PASS | PASS | PASS/PASS/PASS | PASS | 5000.5 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for Drama as a preferred genre, indicating an interest in narratives grounded in emotional realism, character conflict, and interpersonal stakes that center on moral dilemmas, tense exchanges, and consequential decisions over spectacle. The viewer supports liked themes involving courtroom and legal procedural elements, focused argumentation, ethical quandaries, and tense interpersonal confrontations that foreground dialogue driven revelations and the slow unraveling of truth rather than jump scares or graphic content. The viewer expresses a clear dislike for Horror as a genre, suggesting a preference against atmospheric dread, sustained suspense built around fear, and graphic or shocking sensory tactics that aim primarily to provoke visceral reactions instead of contemplative engagement. The viewer also dislikes plot elements centered on jump scares and gratuitous gore, and other viewers who enjoy visceral shock driven horror effects and relentless tension may prefer those films instead, whereas this viewer favors emotionally charged realistic conflict and deliberative pacing.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Horror"], "categorical_positive_on_insufficient_genres": ["Drama"], "weak_broad_unsupported_genres": ["Horror"]}`.

**AFTER**

> Summary: The viewer shows a tentative liking for character-driven drama: they respond to emotionally grounded stories centered on moral dilemmas, interpersonal conflict, and the procedural or institutional settings that highlight personal choices. They may be less interested in intense horror fare, particularly films that rely on sustained shock, gore, or jump scares.

## sparse_weak_negative_evidence

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 60 | 4 | FAIL | FAIL | PASS/PASS/PASS | FAIL: word_count:60 | 7093.8 ms |
| AFTER | 30 | 1 | PASS | PASS | PASS/PASS/PASS | PASS | 3804.0 ms |

**BEFORE**

> Summary: No strong liked genre is supported. No strong liked themes or content are supported. The viewer shows dislike for Adventure. The viewer also shows dislike for plot elements typically associated with low engagement and prefers that other viewers who enjoy quieter, character focused pacing or moderate dramatic beats may be better suited for more neutral or varied Drama content.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Adventure"], "weak_broad_unsupported_genres": ["Adventure"]}`.

**AFTER**

> Summary: The viewer may be somewhat less interested in films that emphasize broad, outward adventure-style expedition narratives; this tendency appears isolated and should be treated as tentative rather than categorical.

## rich_positive_evidence

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 151 | 4 | FAIL | PASS | PASS/PASS/PASS | PASS | 1723.7 ms |
| AFTER | 91 | 4 | PASS | PASS | PASS/PASS/PASS | PASS | 1948.3 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for Drama and Romance and also enjoys blends that pair comedic elements with dramatic storytelling, such as Comedy Drama combinations that keep emotional grounding while offering lighter moments and interpersonal focus. They prefer narratives centered on family relationships, reconnection and second chances, everyday domestic interactions, and gentle interpersonal comedy that foregrounds warmth, reconciliation, and character driven emotional arcs rather than spectacle or high concept twists. No strong disliked genre or style is supported by the viewing history, so there is no firm evidence of aversion to any particular genre or presentation style beyond the absence of data for other categories. No strong disliked plot points or content preferences are supported by the evidence, and viewers who prefer action forward pacing or high stakes spectacle may find this selection less aligned with their tastes while fans of character driven emotional nuance will likely appreciate it.

Grounding findings: `{"none_fabricated_dislike": true, "represented_positive_genres": ["Comedy", "Drama"]}`.

**AFTER**

> Summary: The viewer favors character-driven comedies and dramas that emphasize warm interpersonal relationships, second chances, and everyday domestic moments. They respond well to gentle humor woven into emotionally grounded stories—light, humane comedies with sympathetic neighbors or community interactions—as well as quieter dramatic narratives focused on family bonds, reunions, and personal growth. Storylines that center on reconciliation, heartfelt conversations, and the small but meaningful gestures that bring people back together are likely to appeal to them. Tone matters: they prefer films that balance emotional sincerity with warmth rather than heavy, confrontational intensity.

Grounding findings: `{"represented_positive_genres": ["Comedy", "Drama"]}`.

## rich_mixed_evidence

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 170 | 4 | FAIL | PASS | PASS/PASS/PASS | PASS | 4513.0 ms |
| AFTER | 81 | 4 | PASS | PASS | PASS/PASS/PASS | PASS | 1982.6 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for Drama and Romance as preferred genres and this preference reflects consistent engagement with emotionally driven narratives and relationship focused stories that explore personal growth and interpersonal reconciliation in grounded settings with human stakes and character centered arcs. The viewer prefers themes of family reconciliation, emotional reunions, and intimate character development that emphasize healing, communication, and long term bonds over shock or spectacle and values slow burn emotional revelations and thoughtful explorations of connection. The viewer expresses a clear dislike for Horror and for intensely suspense driven Thriller styles that rely on jump scares, sustained dread, or visceral shock tactics and tends to avoid movies built primarily around fear inducement, grotesque imagery, or relentless tension as a dominant stylistic choice. The viewer dislikes plot devices centered on sustained terror, extreme gore, or shocking violent surprises and might prefer that other viewers who enjoy visceral scares, relentless suspense, or high adrenaline fright experiences seek out more intense horror oriented offerings tailored to those tastes.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Thriller"], "missing_core_positive_genres": ["Drama"], "represented_negative_genres": ["Horror"]}`.

**AFTER**

> Summary: The viewer shows a clear preference for drama-focused films. They respond to emotionally driven, character-centered storytelling—stories that explore relationships, personal growth, and family dynamics resonate with them. By contrast, they have a strong aversion to horror: they prefer to avoid films that rely on jump scares, sustained suspense built around violence or terror, or atmospheres designed primarily to frighten. When recommending or selecting movies for them, prioritize sober, human-scale narratives with emotional stakes and character development, and exclude horror-oriented content.

Grounding findings: `{"represented_negative_genres": ["Horror"], "represented_positive_genres": ["Drama"]}`.

## conflicting_genres

| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---|---:|
| BEFORE | 116 | 4 | FAIL | FAIL | PASS/PASS/PASS | FAIL: word_count:116 | 5946.6 ms |
| AFTER | 88 | 3 | PASS | PASS | PASS/PASS/PASS | PASS | 1720.3 ms |

**BEFORE**

> Summary: This viewer shows a clear liking for Drama. They prefer character focused narratives that explore family relationships, reunions, and intimate emotional reconnections, favoring warm portrayals of interpersonal healing and depth of feeling over plot driven spectacle and preferring slow revelations about personal bonds. They dislike Drama works that feel relentlessly bleak or dominated by protracted conflict and cold emotional distance, indicating an aversion to stories centered on prolonged arguments, alienation, or icy farewells that leave little emotional resolution. Other viewers who enjoy more confrontational, stark, or pessimistic storytelling may still appreciate those colder or more argumentative dramas, so the viewer may prefer to avoid such tones while others might value their rawness and abrasive honesty.

Grounding findings: `{"categorical_negative_on_mixed_genres": ["Drama"], "weak_broad_unsupported_genres": ["Drama"]}`.

**AFTER**

> Summary: The viewer shows a tentative, narrow liking for intimate, character-focused stories that explore personal relationships and emotional reunions; they may respond well to dramas grounded in family dynamics and warm interpersonal reconnections. They may be less interested in dramas that emphasize bleak, prolonged confrontations or cold, emotionally distant farewells, particularly films built around monologues or seemingly endless argumentative exchanges. Overall their response to drama appears selective and context-dependent: they seem to prefer emotionally warm, relationally focused executions while reacting less favorably to bleak or relentlessly confrontational treatments.
