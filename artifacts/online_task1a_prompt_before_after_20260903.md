# Online Task 1a prompt before/after evaluation

Model: `gpt-5-mini-2025-08-07`. Latency is end-to-end generation latency and includes retries.

Frozen V10 production reference: 1,000 summaries; 59–249 words (median 123); 3–8 sentences; 196/1,000 within 140–180 words; 846/1,000 with four sentences. The V10 prompt explicitly did not force an exact word count or four sentences.

Overall AFTER gate: grounding `PASS`, leakage `PASS`, validator `PASS`.

## one_positive_item

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 168 | 4 | FAIL | PASS/PASS/PASS | PASS | 4110.5 ms |
| AFTER | 38 | 3 | PASS | PASS/PASS/PASS | PASS | 2291.6 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for Drama as a primary genre and related emotionally driven storytelling, indicating a preference for character focused narratives that explore interpersonal tension and moral dilemmas with weight and seriousness across extended scenes and conversations. The viewer evidences a strong liking for courtroom and legal procedural themes with intense dialogue, ethical conflict, and focused examinations of justice, portraying layered arguments and careful unraveling of motives that foreground moral complexity and dramatic testimony over spectacle. No strong disliked genre or stylistic preference is supported by the available evidence, so there is insufficient information to assert an aversion to particular genres or filmmaking styles, and the profile therefore abstains from claiming negative genre preferences. No strong disliked plot points or specific content preferences are supported, so it remains possible that other viewers might prefer faster pacing, lighter tones, or action oriented narratives while this viewer may favor contemplative and dialogue heavy works but that conclusion is not definitively supported by the single record provided.

Grounding findings: `{"none_fabricated_dislike": true}`.

**AFTER**

> Summary: The viewer shows a clear positive inclination toward dramas. They may prefer films that foreground emotional, character-driven narratives and tense interpersonal or institutional conflicts commonly explored in dramatic storytelling. The viewing history does not reveal clear dislikes.

## one_negative_item

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 157 | 4 | PASS | PASS/PASS/PASS | PASS | 2383.9 ms |
| AFTER | 65 | 3 | PASS | PASS/PASS/PASS | PASS | 1302.3 ms |

**BEFORE**

> Summary: No strong liked genre is supported by the viewing history, so the profile should not assert a clear favorite genre and should instead remain open to various categories until more positive evidence appears. No strong liked plot points themes or content are supported by the available record, so the profile should refrain from claiming affinity for particular narrative devices or themes and should note uncertainty pending further viewing. The record does support a dislike of horror as a genre or style based on the available evidence, so the profile should indicate avoidance of intense horror elements and similar stylistic approaches while acknowledging that this is grounded in limited data. The record supports a dislike for frightening or unsettling plot content and viscera oriented sequences that other viewers may enjoy for their shock value or catharsis, so the profile should note that these elements are likely unappealing to this viewer while suggesting that others might appreciate them.

**AFTER**

> Summary: The viewing history suggests a mild aversion to at least some horror films; the user may prefer to avoid intense or frightening horror experiences. There is not enough information yet to identify clear positive genre preferences or to characterize likes for specific themes, tones, or plot elements. Reactions to other genres (including romance, comedy, drama, action, and thriller) remain unclear from the available history.

## one_positive_plus_one_negative

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 160 | 4 | FAIL | PASS/PASS/PASS | PASS | 2176.6 ms |
| AFTER | 76 | 3 | PASS | PASS/PASS/PASS | PASS | 1903.3 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for drama genres and emotionally driven storytelling that centers on realistic interpersonal conflict, moral dilemmas, and character focused narratives which unfold primarily through dialogue and tense exchanges between central figures. The viewer demonstrates a supported preference for courtroom and legal procedural themes involving ethical confrontation, persuasive argumentation, and the unraveling of complex personal motives that emphasize resolution through dialogue and revelation rather than supernatural explanation. The viewer shows a clear dislike for horror genres and atmospheric shock driven storytelling that relies on gore, jump scares, or sustained dread to generate emotional response rather than character driven stakes and moral complexity, indicating avoidance of intense visceral fear as a primary mode. The viewer demonstrates a supported dislike for plots that primarily use supernatural, grotesque, or shock based devices to provoke reactions, and other viewers favoring visceral scares, occult elements, or relentless tension may still enjoy those approaches despite this viewer preferring grounded dramatic resolution.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Horror"], "weak_broad_unsupported_genres": ["Horror"]}`.

**AFTER**

> Summary: The viewer shows a clear liking for dramas that focus on structured, dialogue-driven situations such as legal or procedural conflicts and the interpersonal tensions those settings create. They tend to avoid horror films that emphasize sustained suspense, shock, or overt supernatural threat, though this dislike is tentative and may be limited in scope. There is not enough information yet to identify clear preferences for other genres or for more specific themes, tones, or stylistic choices.

## sparse_weak_negative_evidence

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 156 | 4 | PASS | PASS/PASS/FAIL | FAIL: rating_leakage | 6987.1 ms |
| AFTER | 61 | 3 | PASS | PASS/PASS/PASS | PASS | 2992.5 ms |

**BEFORE**

> Summary: No strong liked genre is supported by the available history because the viewer rated an Adventure entry low and a Drama entry middling, so the evidence does not support a clear positive preference for any specific genre at this time. The viewer shows no supported liked plot points themes or content preferences in the record provided, and the neutral and negative ratings leave only inconclusive signals about what narrative elements they favor without inventing particulars. The viewer disliked the Adventure style as supported by low feedback and there is no supported strong dislike for Drama given a middle rating, so apart from avoiding Adventure as indicated there is no broader confirmed aversion to other genres or cinematic styles. The viewer dislikes low-scoring adventurous expedition oriented content while other viewers may enjoy that same adventurous emphasis and high energy exploration focus, and the evidence does not support additional specific disliked plot points beyond this relative aversion.

**AFTER**

> Summary: The viewing history does not reveal clear likes or favorites yet. They have expressed a mild negative reaction to an adventure title, suggesting they may be less inclined toward some adventure-style films, but this tendency appears limited and tentative. There is not enough information to identify definitive positive preferences or broader dislikes across other genres or themes at this time.

## rich_positive_evidence

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 150 | 4 | PASS | PASS/PASS/PASS | PASS | 2484.9 ms |
| AFTER | 74 | 4 | PASS | PASS/PASS/PASS | PASS | 1281.2 ms |

**BEFORE**

> Summary: The viewer shows a clear liking for Drama and Romance and also enjoys blends where Comedy is paired with Drama, indicating a preference for emotionally grounded stories that often include interpersonal relationships and lighter moments without shifting fully into broad comedy. The viewer prefers narratives focused on family dynamics, reunions, second chances, everyday neighborly interactions, and warm interpersonal reconnections, showing an appetite for character driven plots that emphasize reconciliation, domestic settings, and gentle humor woven into serious themes. No strong disliked genre or cinematic style is supported by the provided evidence, so the profile abstains from asserting aversions to any particular genre, tone, or filmmaking approach in general. No strong disliked specific plot point or content preference is supported by the evidence, and other viewers who favor more high concept, action oriented, or experimental narratives may find different appeal than this viewer who leans toward intimate, relationship centered stories.

Grounding findings: `{"represented_positive_genres": ["Comedy", "Drama"]}`.

**AFTER**

> Summary: The viewer shows a clear preference for dramas and comedies, especially films that combine gentle humor with emotionally grounded storytelling. They respond well to character-driven narratives focusing on family, relationships, reunions, and second chances, and they seem to appreciate warm, human moments and modest, everyday conflicts resolved through connection. There is not enough information yet to identify clear preferences regarding romance as a distinct genre. The viewing history does not reveal clear dislikes.

Grounding findings: `{"represented_positive_genres": ["Comedy", "Drama"]}`.

## rich_mixed_evidence

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 151 | 4 | FAIL | PASS/PASS/PASS | PASS | 2462.9 ms |
| AFTER | 67 | 4 | PASS | PASS/PASS/PASS | PASS | 1162.1 ms |

**BEFORE**

> Summary: This viewer shows a clear preference for drama and romance genres supported by multiple positive entries in their history, indicating a strong tilt toward emotionally driven storytelling with interpersonal relationships and character focused narratives that center on reconciliation and family bonds. The viewer prefers plots and themes that emphasize family dynamics, reunion driven emotional arcs, second chance romance elements, and character centric explorations of relationships and reconciliation rather than action oriented or concept driven premises. The viewer clearly dislikes horror and thriller styles as reflected by consistently low evaluations, showing an aversion to intense suspense, graphic scares, high tension pacing, and unsettling atmosphere that prioritize shock and fear over emotional warmth. The viewer disfavors plot elements centered on violence driven scares, graphic horror content, relentless jump scares, and bleak suspenseful ambiguity, while other viewers who enjoy visceral fear, relentless tension, and gore heavy scare tactics may still appreciate those works.

Grounding findings: `{"categorical_negative_on_insufficient_genres": ["Thriller"], "categorical_positive_on_insufficient_genres": ["Romance"], "represented_negative_genres": ["Horror"], "represented_positive_genres": ["Drama"]}`.

**AFTER**

> Summary: The viewer shows a clear preference for drama: they tend to respond positively to character-driven stories focused on emotional relationships and family dynamics. They have a strong dislike for horror and generally avoid films built around shocks, gore, or sustained suspense and fear. Reactions to romance and thriller elements are not yet clear from the viewing history. The viewing history does not reveal other clear dislikes.

Grounding findings: `{"represented_negative_genres": ["Horror"], "represented_positive_genres": ["Drama"]}`.

## conflicting_genres

| Version | Words | Sentences | Grounding | Leakage (title/year/rating) | Validator | Latency |
|---|---:|---:|---|---|---|---:|
| BEFORE | 169 | 4 | PASS | PASS/PASS/PASS | PASS | 4945.2 ms |
| AFTER | 98 | 4 | PASS | PASS/PASS/PASS | PASS | 1643.5 ms |

**BEFORE**

> Summary: The viewer shows a supported liking for drama as a genre, indicating a clear preference for films that prioritize emotional realism, character driven narratives, interpersonal tensions, and grounded everyday conflicts that unfold through extended scenes and intimate portrayals rather than high concept spectacle. The viewer also shows a supported liking for themes centered on reconciliation, familial connection, and warm interpersonal resolution, preferring stories that emphasize healing conversations, restorative encounters, and tender reunions that foreground emotional closure and relational repair over ambiguous or unresolved outcomes. The viewer shows a supported dislike for bleak, pessimistic, or relentlessly somber dramatic styles, indicating they generally avoid works that sustain a heavy tone of despair, prolonged negativity, or unrelieved melancholy without moments of consolation or emotional uplift. The viewer shows a supported dislike for narratives that focus on prolonged arguments, cold estrangement, or bleak monologues and may prefer films that offer viewers softer emotional arcs, restorative scenes, affectionate reunions, or comforting resolutions even if other audiences enjoy more uncompromisingly dark or confrontational drama.

**AFTER**

> Summary: The viewer shows specific, nuanced responses to dramatic stories rather than a simple like-or-dislike for the genre as a whole. They respond positively to dramas that center on warm, intimate human connections and reconciliations—quiet, character-focused narratives about family, reunions, and personal warmth appeal to them. They also react negatively to dramas that emphasize bleakness, prolonged conflict, or emotionally cold partings; stories dominated by relentless argument, stark monologues, or detached farewells are less appealing. There is not enough information yet to identify clear preferences for other genres or broader stylistic tendencies beyond these contrasting reactions within dramatic material.
