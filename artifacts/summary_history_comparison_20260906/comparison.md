# Historical versus current TEARS summary comparison

Recovered source: August 25 commit `1754778339dbba31926abf5cfa8250b4d50d0f95`.

Historical generation requested 120–260 words and four sentences, using minimal reasoning, low verbosity, 450 output tokens, and only a nonempty-response check. Regenerated outputs are fresh samples using those settings, not recovered historical outputs.

All profiles below were scored by the same frozen TEARS checkpoint, with raw logits, the same current onboarding exclusions, and release year >=2015. The genre reranker was bypassed. The saved August 26 profile is an original artifact; its recommendations here are newly rescored, not recovered historical rankings.

Counts of Animation/Comedy tags are diagnostic metadata counts, not ground-truth relevance or NDCG measurements.

The additional without_negative_boilerplate variants remove only the exact two fixed negative-abstention sentences from the current backend; the preceding preference text is unchanged. These are offline diagnostic variants, not deployed summaries.

| Case | Version | Run | Words | Animation /12 | Comedy /12 |
|---|---|---:|---:|---:|---:|
| Soul | saved_current_backend | 0 | 147 | 0 | 0 |
| Soul | saved_current_display | 0 | 47 | 3 | 1 |
| Barbie | saved_current_backend | 0 | 155 | 1 | 1 |
| Barbie | saved_current_display | 0 | 42 | 2 | 1 |
| Soul_mixed_history | archived_20260826 | 0 | 122 | 4 | 1 |
| Barbie | current_backend | 1 | 160 | 2 | 2 |
| Barbie | current_display | 1 | 40 | 3 | 2 |
| Barbie | current_backend | 2 | 167 | 2 | 2 |
| Barbie | current_display | 2 | 37 | 1 | 1 |
| Barbie | current_backend | 3 | 164 | 1 | 1 |
| Barbie | current_display | 3 | 32 | 0 | 1 |
| Barbie | historical_regenerated | 1 | 75 | 3 | 2 |
| Barbie | historical_regenerated | 2 | 38 | 3 | 2 |
| Barbie | historical_regenerated | 3 | 70 | 2 | 2 |
| Soul | current_backend | 1 | 155 | 3 | 3 |
| Soul | current_display | 1 | 54 | 6 | 3 |
| Soul | current_backend | 2 | 152 | 2 | 2 |
| Soul | current_display | 2 | 50 | 4 | 2 |
| Soul | current_backend | 3 | 155 | 2 | 2 |
| Soul | current_display | 3 | 38 | 6 | 3 |
| Soul | historical_regenerated | 1 | 56 | 4 | 1 |
| Soul | historical_regenerated | 2 | 64 | 7 | 3 |
| Soul | historical_regenerated | 3 | 74 | 10 | 4 |
| Soul_mixed_history | current_backend | 1 | 159 | 1 | 1 |
| Soul_mixed_history | current_display | 1 | 52 | 0 | 1 |
| Soul_mixed_history | current_backend | 2 | 154 | 4 | 3 |
| Soul_mixed_history | current_display | 2 | 53 | 2 | 2 |
| Soul_mixed_history | current_backend | 3 | 160 | 1 | 1 |
| Soul_mixed_history | current_display | 3 | 49 | 3 | 2 |
| Soul_mixed_history | historical_regenerated | 1 | 111 | 4 | 1 |
| Soul_mixed_history | historical_regenerated | 2 | 99 | 3 | 1 |
| Soul_mixed_history | historical_regenerated | 3 | 88 | 1 | 1 |
| Soul | saved_current_backend_without_negative_boilerplate | 0 | 76 | 1 | 1 |
| Barbie | saved_current_backend_without_negative_boilerplate | 0 | 84 | 0 | 0 |
| Barbie | current_backend_without_negative_boilerplate | 1 | 89 | 1 | 3 |
| Barbie | current_backend_without_negative_boilerplate | 2 | 96 | 1 | 2 |
| Barbie | current_backend_without_negative_boilerplate | 3 | 93 | 1 | 1 |
| Soul | current_backend_without_negative_boilerplate | 1 | 84 | 5 | 6 |
| Soul | current_backend_without_negative_boilerplate | 2 | 81 | 2 | 2 |
| Soul | current_backend_without_negative_boilerplate | 3 | 84 | 3 | 2 |
| Soul_mixed_history | current_backend_without_negative_boilerplate | 1 | 88 | 2 | 1 |
| Soul_mixed_history | current_backend_without_negative_boilerplate | 2 | 83 | 4 | 3 |
| Soul_mixed_history | current_backend_without_negative_boilerplate | 3 | 89 | 1 | 0 |

## Summaries and raw recommendations

### Soul — saved_current_backend — run 0

Summary: The viewer shows a tentative preference for animated family films blending adventure, gentle fantasy, and light comedy, favoring imaginative visuals and playful pacing, while noting that this observation comes from a very small positive sample and is limited. No strong preference is supported for specific plot points or thematic beats in this viewer history, and the limited evidence prevents reliable claims about narrative focus, tone, or character arcs beyond the modest genre observation already stated. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. The Hateful Eight (2015)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Sicario (2015)
4. Blade Runner 2049 (2017)
5. Mission: Impossible - Rogue Nation (2015)
6. Rogue One: A Star Wars Story (2016)
7. Logan (2017)
8. The Revenant (2015)
9. John Wick: Chapter Two (2017)
10. Avengers: Age of Ultron (2015)
11. Captain America: Civil War (2016)
12. Ant-Man (2015)

### Soul — saved_current_display — run 0

Summary: The viewer seems to enjoy warm, imaginative animated stories that blend humor with thoughtful, introspective themes. They may respond well to films that combine whimsical fantasy elements and lively adventure with heart—stories that use playful visuals and comedy to explore meaningful questions about purpose and identity.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Guardians of the Galaxy 2 (2017)
4. Thor: Ragnarok (2017)
5. Avengers: Age of Ultron (2015)
6. Doctor Strange (2016)
7. Rogue One: A Star Wars Story (2016)
8. Logan (2017)
9. Ant-Man (2015)
10. Captain America: Civil War (2016)
11. Coco (2017)
12. Spider-Man: Into the Spider-Verse (2018)

### Barbie — saved_current_backend — run 0

Summary: The viewer shows a preference for lighthearted comedic fare based on available positive feedback, indicating enjoyment of humor driven storytelling, situational comedy elements, and broadly entertaining tones, though this evidence is limited to a small set so the genre preference remains tentative. No strong preference is supported regarding specific plot points, themes, or content preferences because the available record does not include detail on narrative elements, pacing, character focus, or recurring motifs, and this absence prevents drawing reliable conclusions about such content tastes. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Rogue One: A Star Wars Story (2016)
3. Logan (2017)
4. Guardians of the Galaxy 2 (2017)
5. Avengers: Age of Ultron (2015)
6. Zootopia (2016)
7. Thor: Ragnarok (2017)
8. Doctor Strange (2016)
9. The Hateful Eight (2015)
10. Jurassic World (2015)
11. Captain America: Civil War (2016)
12. Ant-Man (2015)

### Barbie — saved_current_display — run 0

Summary: The viewer may enjoy lighthearted, comedic films that blend playful satire with bright, stylized presentation and a broadly upbeat tone. They seem receptive to movies that lean into humor, self-aware commentary, and whimsical or imaginative worldbuilding rather than strictly serious drama.

Recommendations:

1. Doctor Strange (2016)
2. Ant-Man (2015)
3. Guardians of the Galaxy 2 (2017)
4. Star Wars: Episode VII - The Force Awakens (2015)
5. Thor: Ragnarok (2017)
6. Avengers: Age of Ultron (2015)
7. Captain America: Civil War (2016)
8. Zootopia (2016)
9. Untitled Spider-Man Reboot (2017)
10. Spider-Man: Into the Spider-Verse (2018)
11. Wonder Woman (2017)
12. Black Panther (2017)

### Soul_mixed_history — archived_20260826 — run 0

Summary: The viewer consistently enjoys imaginative, large scale genres combining action adventure and science fiction with dramatic weight as well as playful, heartfelt animated adventures that blend comedy and fantasy, indicating a preference for ambitious, genre-spanning storytelling. They favor films that mix high stakes physical or existential journeys with emotional introspection and humor, appreciating narratives that explore identity and meaning through both epic spectacle and intimate, whimsical character moments. They have no strong supported dislike for any particular genre or style based on the provided history. They have no strong supported dislike for specific plot points or content preferences and something others might dislike could still appeal to this viewer because they seem open to bold tonal shifts and unconventional narrative structures.

Recommendations:

1. Spider-Man: Into the Spider-Verse (2018)
2. Joker (2019)
3. Your Name. (2016)
4. Blade Runner 2049 (2017)
5. Avengers: Infinity War - Part II (2019)
6. Thor: Ragnarok (2017)
7. Logan (2017)
8. Coco (2017)
9. Star Wars: Episode VII - The Force Awakens (2015)
10. Zootopia (2016)
11. The Revenant (2015)
12. 1917 (2019)

### Barbie — current_backend — run 1

Summary: The viewer shows a tentative preference for comedic films, with private history indicating enjoyment of at least one comedy, and this observation should be treated as narrowly supported rather than definitive because the available evidence is limited and concentrated in a single example. No strong preference is supported for specific plot points, themes, or content beyond that narrow comedic inclination, so the profile does not assert particular narrative elements liked or disliked and acknowledges that further viewing evidence would be required to describe any finer grained tastes confidently. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Rogue One: A Star Wars Story (2016)
3. Blade Runner 2049 (2017)
4. Thor: Ragnarok (2017)
5. Logan (2017)
6. Guardians of the Galaxy 2 (2017)
7. The Hateful Eight (2015)
8. Knives Out (2019)
9. Zootopia (2016)
10. Spider-Man: Into the Spider-Verse (2018)
11. Spotlight (2015)
12. Avengers: Infinity War - Part II (2019)

### Barbie — current_display — run 1

Summary: The viewer seems receptive to light-hearted, comedic films with playful, satirical tones and an emphasis on broad humor and colorful, optimistic presentation. They may enjoy movies that blend contemporary cultural commentary with upbeat, whimsical storytelling and strong visual style.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Knives Out (2019)
4. Guardians of the Galaxy 2 (2017)
5. Ant-Man (2015)
6. Wonder Woman (2017)
7. Doctor Strange (2016)
8. Spider-Man: Into the Spider-Verse (2018)
9. Black Panther (2017)
10. Thor: Ragnarok (2017)
11. Captain America: Civil War (2016)
12. Coco (2017)

### Barbie — current_backend — run 2

Summary: The viewer shows a preference for films that foreground lighthearted and comedic tones, as indicated by enjoyment of a comedy instance, so they likely appreciate humor driven pacing, playful setups, and situations that prioritize levity, though this evidence is limited and does not define all comedic substyles. No strong preference is supported for specific plot points, themes, or content elements beyond a general openness to comedic framing, so there is insufficient evidence to claim favored narrative beats, emotional arcs, character types, or recurring motifs and further viewing choices would be needed to refine those particulars. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. Guardians of the Galaxy 2 (2017)
4. Rogue One: A Star Wars Story (2016)
5. Thor: Ragnarok (2017)
6. Logan (2017)
7. Spider-Man: Into the Spider-Verse (2018)
8. Avengers: Infinity War - Part II (2019)
9. Doctor Strange (2016)
10. Deadpool 2 (2018)
11. The Hateful Eight (2015)
12. Blade Runner 2049 (2017)

### Barbie — current_display — run 2

Summary: The viewer seems receptive to lighthearted, comedic entertainment that blends playful satire with broad, self-aware humor. They may appreciate films that lean into whimsical premises and upbeat, fast-paced storytelling delivered with a wink rather than solemnity.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Avengers: Age of Ultron (2015)
3. Ant-Man (2015)
4. Guardians of the Galaxy 2 (2017)
5. Doctor Strange (2016)
6. Captain America: Civil War (2016)
7. Rogue One: A Star Wars Story (2016)
8. Thor: Ragnarok (2017)
9. Jurassic World (2015)
10. Mission: Impossible - Rogue Nation (2015)
11. Zootopia (2016)
12. Wonder Woman (2017)

### Barbie — current_backend — run 3

Summary: The viewer shows a preference for lighthearted comedic films characterized by humor and playful tone, with this observation narrowly supported by available viewing history and therefore best treated as a tentative, limited indication of enjoyment within the comedy category rather than a broad genre endorsement. No strong preference is supported for particular plot points, themes, or narrative devices because the private evidence does not include repeated examples or detailed content notes, so the absence of such specifics means only that no reliable, narrowly scoped content preferences can be asserted from the record. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Rogue One: A Star Wars Story (2016)
3. Blade Runner 2049 (2017)
4. The Hateful Eight (2015)
5. Logan (2017)
6. Avengers: Age of Ultron (2015)
7. Doctor Strange (2016)
8. The Revenant (2015)
9. Zootopia (2016)
10. Spotlight (2015)
11. Thor: Ragnarok (2017)
12. Guardians of the Galaxy 2 (2017)

### Barbie — current_display — run 3

Summary: The viewer may enjoy lighthearted, comedic films with playful, satirical, or whimsical tones; they seem receptive to bright, humor-driven storytelling that leans into genre-savvy self-awareness and comedic commentary on social conventions.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Guardians of the Galaxy 2 (2017)
3. Doctor Strange (2016)
4. Thor: Ragnarok (2017)
5. Ant-Man (2015)
6. Rogue One: A Star Wars Story (2016)
7. Avengers: Age of Ultron (2015)
8. Captain America: Civil War (2016)
9. Wonder Woman (2017)
10. Logan (2017)
11. Black Panther (2017)
12. Deadpool 2 (2018)

### Barbie — historical_regenerated — run 1

Summary: The viewer shows a strong liking for comedy and upbeat, stylized films. They favor narratives that emphasize playful satire, vibrant aesthetics, and protagonist-driven journeys with a focus on self discovery and social commentary. No strong dislike of any genre is supported by the available viewing history. No strong dislike of specific plot points or content preferences is supported and others may enjoy a wider range of tones and genres than this single viewing implies.

Recommendations:

1. Spider-Man: Into the Spider-Verse (2018)
2. Thor: Ragnarok (2017)
3. Star Wars: Episode VII - The Force Awakens (2015)
4. Zootopia (2016)
5. Guardians of the Galaxy 2 (2017)
6. Avengers: Infinity War - Part II (2019)
7. Rogue One: A Star Wars Story (2016)
8. Black Panther (2017)
9. Doctor Strange (2016)
10. Coco (2017)
11. Logan (2017)
12. Knives Out (2019)

### Barbie — historical_regenerated — run 2

Summary: The viewer strongly prefers comedies and enjoys lighthearted, humorous films. The viewer likes upbeat, playful narratives with satirical or whimsical elements that explore identity and self discovery. No strong dislike is supported. No strong dislike is supported.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. Spider-Man: Into the Spider-Verse (2018)
4. Thor: Ragnarok (2017)
5. Guardians of the Galaxy 2 (2017)
6. Logan (2017)
7. Rogue One: A Star Wars Story (2016)
8. Avengers: Infinity War - Part II (2019)
9. Coco (2017)
10. Knives Out (2019)
11. Get Out (2017)
12. Joker (2019)

### Barbie — historical_regenerated — run 3

Summary: The viewer shows a strong preference for comedy films. The viewer prefers upbeat, satirical, and visually vibrant stories that play with cultural tropes and self aware humor. The viewer has no strong dislike supported by the viewing history. The viewer has no strong dislike of specific plot points or content preferences supported by the viewing history; other viewers who prefer quieter or more serious dramas may enjoy different films.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. Spotlight (2015)
4. Coco (2017)
5. La La Land (2016)
6. The Revenant (2015)
7. Rogue One: A Star Wars Story (2016)
8. The Hateful Eight (2015)
9. Blade Runner 2049 (2017)
10. Get Out (2017)
11. Dunkirk (2017)
12. Room (2015)

### Soul — current_backend — run 1

Summary: The viewer shows a preference for animated family oriented adventure with gentle comedy and imaginative fantasy elements, suggesting an appreciation for visually creative storytelling that blends whimsical scenarios and accessible emotional themes while acknowledging the evidence is limited and not definitive. They appear to favor stories that weave character focused moments with inventive visual sequences and lighthearted humor that invite gentle reflection and emotional warmth, but no strong preference is supported for specific plot points or broader thematic claims given the narrow evidence. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Zootopia (2016)
2. Blade Runner 2049 (2017)
3. The Hateful Eight (2015)
4. Knives Out (2019)
5. Star Wars: Episode VII - The Force Awakens (2015)
6. The Lobster (2015)
7. Three Billboards Outside Ebbing, Missouri (2017)
8. Get Out (2017)
9. Spider-Man: Into the Spider-Verse (2018)
10. The Nice Guys (2016)
11. The Revenant (2015)
12. Coco (2017)

### Soul — current_display — run 1

Summary: The viewer shows a narrow preference for animated, family-friendly stories that blend humor with whimsical, imaginative themes. They seem to appreciate films that combine heartfelt emotional beats with light comedy and a sense of wonder or fantasy, especially when presented in an accessible, visually driven way suitable for younger viewers and family audiences.

Recommendations:

1. Zootopia (2016)
2. Coco (2017)
3. Moana (2016)
4. Thor: Ragnarok (2017)
5. Spider-Man: Into the Spider-Verse (2018)
6. Doctor Strange (2016)
7. Guardians of the Galaxy 2 (2017)
8. Star Wars: Episode VII - The Force Awakens (2015)
9. Avengers: Infinity War - Part II (2019)
10. Incredibles 2 (2018)
11. Finding Dory (2016)
12. Avengers: Age of Ultron (2015)

### Soul — current_backend — run 2

Summary: The viewer shows a tentative preference for adventure and animated family oriented films with imaginative premises, inferred from a single positively received animated adventure, so this pattern is limited and does not establish a broad genre preference. They seem open to lighthearted, character focused storytelling that incorporates humor alongside inventive settings, but this conclusion is cautious because it rests on only one example and cannot reliably predict interest in varied narrative complexities, tonal shifts, or more adult oriented thematic treatments. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. The Hateful Eight (2015)
4. Blade Runner 2049 (2017)
5. The Revenant (2015)
6. Logan (2017)
7. Rogue One: A Star Wars Story (2016)
8. Knives Out (2019)
9. Guardians of the Galaxy 2 (2017)
10. Get Out (2017)
11. Coco (2017)
12. Doctor Strange (2016)

### Soul — current_display — run 2

Summary: The viewer shows a narrow, positive inclination toward animated stories that blend imaginative, emotionally reflective themes with light comedic moments and a sense of wonder. They may appreciate films that feel adventurous and family-friendly while also exploring introspective or existential ideas through whimsical fantasy elements and warm, character-driven humor.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Coco (2017)
4. Doctor Strange (2016)
5. Moana (2016)
6. Guardians of the Galaxy 2 (2017)
7. Ant-Man (2015)
8. Spider-Man: Into the Spider-Verse (2018)
9. Thor: Ragnarok (2017)
10. Logan (2017)
11. Avengers: Age of Ultron (2015)
12. Rogue One: A Star Wars Story (2016)

### Soul — current_backend — run 3

Summary: The viewer shows a preference for films classified as adventure, animation, children, comedy, and fantasy, indicating an inclination toward imaginative visual storytelling, playful or whimsical tones, and family oriented narratives, while acknowledging that the evidence is limited to genre labels. They have no strong preference supported for particular plot points, themes, or specific narrative beats based on the supplied history, so it is not possible to reliably characterize favored emotional arcs, philosophical questions, or recurring character journeys beyond what the genre labels suggest. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Blade Runner 2049 (2017)
2. The Hateful Eight (2015)
3. Star Wars: Episode VII - The Force Awakens (2015)
4. The Revenant (2015)
5. Zootopia (2016)
6. Logan (2017)
7. Rogue One: A Star Wars Story (2016)
8. Joker (2019)
9. Spider-Man: Into the Spider-Verse (2018)
10. Knives Out (2019)
11. Doctor Strange (2016)
12. Dunkirk (2017)

### Soul — current_display — run 3

Summary: The viewer seems to enjoy whimsical, animated stories that blend imaginative fantasy with warm humor and heartfelt themes. They may prefer films that feel kid-friendly yet thoughtful—animation that combines playful adventure and comedy with emotional, reflective moments.

Recommendations:

1. Zootopia (2016)
2. Coco (2017)
3. Moana (2016)
4. Spider-Man: Into the Spider-Verse (2018)
5. Thor: Ragnarok (2017)
6. Star Wars: Episode VII - The Force Awakens (2015)
7. Guardians of the Galaxy 2 (2017)
8. Doctor Strange (2016)
9. Avengers: Infinity War - Part II (2019)
10. Incredibles 2 (2018)
11. Fantastic Beasts and Where to Find Them (2016)
12. Finding Dory (2016)

### Soul — historical_regenerated — run 1

Summary: The viewer favors animated adventure and family friendly films with whimsical or imaginative tones. They prefer stories that explore personal growth, existential themes, and emotional depth conveyed through playful or fantastical settings. No strong preference against specific genres, themes, or styles is supported. No strong dislike of particular plot points or content preferences is supported.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. The Hateful Eight (2015)
4. Coco (2017)
5. Joker (2019)
6. Blade Runner 2049 (2017)
7. The Revenant (2015)
8. Get Out (2017)
9. Spider-Man: Into the Spider-Verse (2018)
10. Three Billboards Outside Ebbing, Missouri (2017)
11. Your Name. (2016)
12. Logan (2017)

### Soul — historical_regenerated — run 2

Summary: The viewer strongly prefers animated family friendly adventure and comedy stories with fantastical or metaphysical elements. They like uplifting, reflective themes about purpose and personal growth, with imaginative settings and gentle humor. No strong dislike is supported. No strong dislike of specific plot points or content is supported; other viewers who prefer darker, more realistic, or action driven narratives may enjoy different films.

Recommendations:

1. Zootopia (2016)
2. Coco (2017)
3. Moana (2016)
4. Spider-Man: Into the Spider-Verse (2018)
5. Finding Dory (2016)
6. Incredibles 2 (2018)
7. Star Wars: Episode VII - The Force Awakens (2015)
8. Thor: Ragnarok (2017)
9. Guardians of the Galaxy 2 (2017)
10. Doctor Strange (2016)
11. Kung Fu Panda 3 (2016)
12. Avengers: Infinity War - Part II (2019)

### Soul — historical_regenerated — run 3

Summary: The viewer strongly favors animated adventure and family friendly stories with whimsical and imaginative tones. They appreciate themes of personal growth, heartfelt emotional journeys, and uplifting messages about purpose and self discovery. They have no strong dislike supported by the viewing history. They have no strong dislike of specific plot points or content preferences supported by the viewing history; other viewers who prefer darker or more realistic adult dramas may enjoy different choices.

Recommendations:

1. Coco (2017)
2. Zootopia (2016)
3. Moana (2016)
4. Finding Dory (2016)
5. Spider-Man: Into the Spider-Verse (2018)
6. Incredibles 2 (2018)
7. Hidden Figures (2016)
8. Piper (2016)
9. Your Name. (2016)
10. Wonder (2017)
11. Klaus (2019)
12. Kubo and the Two Strings (2016)

### Soul_mixed_history — current_backend — run 1

Summary: The viewer shows a tentative preference for films that blend speculative and adventurous settings with grounded emotional drama and moments of broad comedy, indicating enjoyment of stories that pair imaginative world building and kinetic action with character focused stakes rather than a single narrowly defined genre. They appear to favor narratives where high concept premises are balanced with intimate character arcs and inventive humor, though the viewing record is limited and supports only a cautious, narrow claim rather than a categorical assertion about all preferred plot elements. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. Ant-Man (2015)
4. Doctor Strange (2016)
5. Rogue One: A Star Wars Story (2016)
6. Logan (2017)
7. Guardians of the Galaxy 2 (2017)
8. Captain America: Civil War (2016)
9. Blade Runner 2049 (2017)
10. Avengers: Age of Ultron (2015)
11. The Hateful Eight (2015)
12. Thor: Ragnarok (2017)

### Soul_mixed_history — current_display — run 1

Summary: The viewer appears to gravitate toward high-concept, imaginative films that blend adventurous world-building with emotional or comedic throughlines. They may enjoy stories that combine speculative or science-fiction elements with action and moments of heartfelt character development, and they seem receptive to inventive, genre-mixing approaches that pair visual ambition with personal stakes.

Recommendations:

1. Doctor Strange (2016)
2. Wonder Woman (2017)
3. Ant-Man (2015)
4. Guardians of the Galaxy 2 (2017)
5. Star Wars: Episode VII - The Force Awakens (2015)
6. Rogue One: A Star Wars Story (2016)
7. Captain America: Civil War (2016)
8. Free Guy (2020)
9. Thor: Ragnarok (2017)
10. Avengers: Age of Ultron (2015)
11. Black Panther (2017)
12. Star Trek Beyond (2016)

### Soul_mixed_history — current_backend — run 2

Summary: The viewer's history shows positive responses to films that blend speculative science fiction imagery, kinetic action and adventurous scope with elements of comedy and animation, indicating an appetite for varied formats that mix spectacle with accessible, broadly appealing storytelling across different tones. Their selection history also suggests a preference for concept driven works that pair imaginative worldbuilding with intimate character focus and emotional exploration, favoring creative formal approaches that interweave humor, existential questions, and family oriented elements rather than rigidly conventional realism. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Blade Runner 2049 (2017)
2. The Hateful Eight (2015)
3. Star Wars: Episode VII - The Force Awakens (2015)
4. Zootopia (2016)
5. Get Out (2017)
6. Spider-Man: Into the Spider-Verse (2018)
7. Three Billboards Outside Ebbing, Missouri (2017)
8. The Revenant (2015)
9. Knives Out (2019)
10. The Lobster (2015)
11. Coco (2017)
12. Your Name. (2016)

### Soul_mixed_history — current_display — run 2

Summary: The viewer seems to respond to adventurous, high-concept stories that blend kinetic spectacle with emotional depth and inventive premises. They may enjoy films that combine science-fiction or fantastical worldbuilding with heartfelt character arcs and moments of playful or surreal humor, especially when action and comedic energy are woven into emotionally resonant storytelling.

Recommendations:

1. Doctor Strange (2016)
2. Guardians of the Galaxy 2 (2017)
3. Spider-Man: Into the Spider-Verse (2018)
4. Thor: Ragnarok (2017)
5. Zootopia (2016)
6. Wonder Woman (2017)
7. Ant-Man (2015)
8. Black Panther (2017)
9. Untitled Spider-Man Reboot (2017)
10. Fantastic Beasts and Where to Find Them (2016)
11. Logan (2017)
12. Knives Out (2019)

### Soul_mixed_history — current_backend — run 3

Summary: no strong preference is supported for categorical genre likes in this profile because the available positive viewing history spans multiple broad categories without producing a clear, exclusive pattern that would permit asserting a stable liking for any single genre family across different storytelling approaches. No strong preference is supported for specific plot points, themes, or storytelling devices in this record because the examples include varied tones and narrative aims, leaving only a tentative, noncommittal indication that the viewer has sampled diverse kinds of cinematic pacing and emotional range. No strong preference is supported for disliked genres or styles, leaving this aspect of the viewer's taste unspecified without implying that they welcome every genre or share the same response to all approaches to storytelling. No strong preference is supported for disliked plot points or content, leaving the viewer's individual boundaries unspecified without identifying unwanted story elements or assuming that content enjoyed by other viewers would suit their own personal tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Logan (2017)
3. Avengers: Age of Ultron (2015)
4. Thor: Ragnarok (2017)
5. Rogue One: A Star Wars Story (2016)
6. Doctor Strange (2016)
7. Guardians of the Galaxy 2 (2017)
8. Blade Runner 2049 (2017)
9. Captain America: Civil War (2016)
10. Ant-Man (2015)
11. The Hateful Eight (2015)
12. Zootopia (2016)

### Soul_mixed_history — current_display — run 3

Summary: The viewer seems to favor films that blend high-concept, imaginative premises with emotional or comedic heart. They may enjoy stories that combine speculative or fantastical worldbuilding—often with action or adventurous momentum—with character-driven drama and moments of humor or whimsy, especially when inventive storytelling and emotional stakes are prominent.

Recommendations:

1. Zootopia (2016)
2. Spider-Man: Into the Spider-Verse (2018)
3. Star Wars: Episode VII - The Force Awakens (2015)
4. Coco (2017)
5. Thor: Ragnarok (2017)
6. Guardians of the Galaxy 2 (2017)
7. Knives Out (2019)
8. Doctor Strange (2016)
9. Rogue One: A Star Wars Story (2016)
10. Logan (2017)
11. Avengers: Infinity War - Part II (2019)
12. Blade Runner 2049 (2017)

### Soul_mixed_history — historical_regenerated — run 1

Summary: The viewer strongly favors high-energy speculative and adventure genres, often enjoying visually ambitious action and science fiction alongside adventurous and imaginative family friendly animation. They prefer films that blend emotional depth with existential or fantastical themes, showing an affinity for character journeys that explore meaning, identity, and personal transformation within inventive or surreal settings. They have no strong preference supported for pure romance or horror and no strong preference supported for understated domestic drama. They tend to dislike straightforward procedural stories or conventional genre-only fare, and may not appreciate films that prioritize plot mechanics over emotional resonance while other viewers who favor tightly plotted, genre-pure narratives might enjoy those more.

Recommendations:

1. Your Name. (2016)
2. Coco (2017)
3. Spider-Man: Into the Spider-Verse (2018)
4. Zootopia (2016)
5. Blade Runner 2049 (2017)
6. Joker (2019)
7. Logan (2017)
8. Star Wars: Episode VII - The Force Awakens (2015)
9. Avengers: Infinity War - Part II (2019)
10. Get Out (2017)
11. Thor: Ragnarok (2017)
12. The Revenant (2015)

### Soul_mixed_history — historical_regenerated — run 2

Summary: The viewer strongly prefers high-energy genre blends that combine action and adventure with speculative science fiction and dramatic stakes. They enjoy films that mix inventive fantastical or metaphysical concepts with heartfelt character journeys and playful comedic moments, favoring stories that explore meaning, identity, and emotional growth through imaginative premises. They have no strong preference supported against any specific genres or styles in the provided history. They show no strong preference supported against particular plot points or content, and other viewers with different tastes may appreciate more subdued, purely realistic dramas or films that avoid speculative or surreal elements.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Thor: Ragnarok (2017)
4. Coco (2017)
5. Spider-Man: Into the Spider-Verse (2018)
6. Avengers: Infinity War - Part II (2019)
7. Guardians of the Galaxy 2 (2017)
8. Logan (2017)
9. Rogue One: A Star Wars Story (2016)
10. Doctor Strange (2016)
11. Joker (2019)
12. Blade Runner 2049 (2017)

### Soul_mixed_history — historical_regenerated — run 3

Summary: The viewer strongly favors films that blend speculative and adventurous genres, especially those combining science fiction, action, and broad imaginative adventure elements. They enjoy narratives that explore existential themes, creative worldbuilding, and emotional depth alongside energetic pacing and inventive visual storytelling. They have no strong supported dislike of an entire genre or style based on the provided history. They have no strong supported dislike of specific plot points or content preferences, and other viewers who prefer more conventional realist drama or restrained low energy pacing may differ.

Recommendations:

1. Blade Runner 2049 (2017)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. Spider-Man: Into the Spider-Verse (2018)
4. Logan (2017)
5. Rogue One: A Star Wars Story (2016)
6. Get Out (2017)
7. Joker (2019)
8. Annihilation (2018)
9. The Hateful Eight (2015)
10. The Lobster (2015)
11. Thor: Ragnarok (2017)
12. The Revenant (2015)

### Soul — saved_current_backend_without_negative_boilerplate — run 0

Summary: The viewer shows a tentative preference for animated family films blending adventure, gentle fantasy, and light comedy, favoring imaginative visuals and playful pacing, while noting that this observation comes from a very small positive sample and is limited. No strong preference is supported for specific plot points or thematic beats in this viewer history, and the limited evidence prevents reliable claims about narrative focus, tone, or character arcs beyond the modest genre observation already stated.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. The Hateful Eight (2015)
3. Sicario (2015)
4. Rogue One: A Star Wars Story (2016)
5. Jurassic World (2015)
6. Avengers: Age of Ultron (2015)
7. The Revenant (2015)
8. Blade Runner 2049 (2017)
9. Ant-Man (2015)
10. Zootopia (2016)
11. Captain America: Civil War (2016)
12. Spotlight (2015)

### Barbie — saved_current_backend_without_negative_boilerplate — run 0

Summary: The viewer shows a preference for lighthearted comedic fare based on available positive feedback, indicating enjoyment of humor driven storytelling, situational comedy elements, and broadly entertaining tones, though this evidence is limited to a small set so the genre preference remains tentative. No strong preference is supported regarding specific plot points, themes, or content preferences because the available record does not include detail on narrative elements, pacing, character focus, or recurring motifs, and this absence prevents drawing reliable conclusions about such content tastes.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Rogue One: A Star Wars Story (2016)
3. Thor: Ragnarok (2017)
4. Guardians of the Galaxy 2 (2017)
5. Spotlight (2015)
6. Avengers: Infinity War - Part II (2019)
7. Avengers: Age of Ultron (2015)
8. Logan (2017)
9. Captain America: Civil War (2016)
10. Doctor Strange (2016)
11. Jurassic World (2015)
12. Blade Runner 2049 (2017)

### Barbie — current_backend_without_negative_boilerplate — run 1

Summary: The viewer shows a tentative preference for comedic films, with private history indicating enjoyment of at least one comedy, and this observation should be treated as narrowly supported rather than definitive because the available evidence is limited and concentrated in a single example. No strong preference is supported for specific plot points, themes, or content beyond that narrow comedic inclination, so the profile does not assert particular narrative elements liked or disliked and acknowledges that further viewing evidence would be required to describe any finer grained tastes confidently.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Spotlight (2015)
3. Rogue One: A Star Wars Story (2016)
4. Zootopia (2016)
5. Guardians of the Galaxy 2 (2017)
6. Thor: Ragnarok (2017)
7. Knives Out (2019)
8. Blade Runner 2049 (2017)
9. Avengers: Age of Ultron (2015)
10. Hidden Figures (2016)
11. La La Land (2016)
12. Captain America: Civil War (2016)

### Barbie — current_backend_without_negative_boilerplate — run 2

Summary: The viewer shows a preference for films that foreground lighthearted and comedic tones, as indicated by enjoyment of a comedy instance, so they likely appreciate humor driven pacing, playful setups, and situations that prioritize levity, though this evidence is limited and does not define all comedic substyles. No strong preference is supported for specific plot points, themes, or content elements beyond a general openness to comedic framing, so there is insufficient evidence to claim favored narrative beats, emotional arcs, character types, or recurring motifs and further viewing choices would be needed to refine those particulars.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Zootopia (2016)
3. Guardians of the Galaxy 2 (2017)
4. Thor: Ragnarok (2017)
5. Deadpool 2 (2018)
6. Rogue One: A Star Wars Story (2016)
7. Doctor Strange (2016)
8. Avengers: Infinity War - Part II (2019)
9. Logan (2017)
10. Ant-Man (2015)
11. Avengers: Age of Ultron (2015)
12. Captain America: Civil War (2016)

### Barbie — current_backend_without_negative_boilerplate — run 3

Summary: The viewer shows a preference for lighthearted comedic films characterized by humor and playful tone, with this observation narrowly supported by available viewing history and therefore best treated as a tentative, limited indication of enjoyment within the comedy category rather than a broad genre endorsement. No strong preference is supported for particular plot points, themes, or narrative devices because the private evidence does not include repeated examples or detailed content notes, so the absence of such specifics means only that no reliable, narrowly scoped content preferences can be asserted from the record.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Spotlight (2015)
3. Rogue One: A Star Wars Story (2016)
4. Blade Runner 2049 (2017)
5. Zootopia (2016)
6. Avengers: Age of Ultron (2015)
7. The Revenant (2015)
8. Captain America: Civil War (2016)
9. Doctor Strange (2016)
10. The Hateful Eight (2015)
11. Hidden Figures (2016)
12. Bridge of Spies (2015)

### Soul — current_backend_without_negative_boilerplate — run 1

Summary: The viewer shows a preference for animated family oriented adventure with gentle comedy and imaginative fantasy elements, suggesting an appreciation for visually creative storytelling that blends whimsical scenarios and accessible emotional themes while acknowledging the evidence is limited and not definitive. They appear to favor stories that weave character focused moments with inventive visual sequences and lighthearted humor that invite gentle reflection and emotional warmth, but no strong preference is supported for specific plot points or broader thematic claims given the narrow evidence.

Recommendations:

1. Zootopia (2016)
2. Coco (2017)
3. The Lobster (2015)
4. Moana (2016)
5. Star Wars: Episode VII - The Force Awakens (2015)
6. Room (2015)
7. La La Land (2016)
8. Knives Out (2019)
9. Three Billboards Outside Ebbing, Missouri (2017)
10. The Shape of Water (2017)
11. Isle of Dogs (2018)
12. Kubo and the Two Strings (2016)

### Soul — current_backend_without_negative_boilerplate — run 2

Summary: The viewer shows a tentative preference for adventure and animated family oriented films with imaginative premises, inferred from a single positively received animated adventure, so this pattern is limited and does not establish a broad genre preference. They seem open to lighthearted, character focused storytelling that incorporates humor alongside inventive settings, but this conclusion is cautious because it rests on only one example and cannot reliably predict interest in varied narrative complexities, tonal shifts, or more adult oriented thematic treatments.

Recommendations:

1. Zootopia (2016)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. The Hateful Eight (2015)
4. Spotlight (2015)
5. The Revenant (2015)
6. Coco (2017)
7. Blade Runner 2049 (2017)
8. Three Billboards Outside Ebbing, Missouri (2017)
9. Sicario (2015)
10. Rogue One: A Star Wars Story (2016)
11. Knives Out (2019)
12. Bridge of Spies (2015)

### Soul — current_backend_without_negative_boilerplate — run 3

Summary: The viewer shows a preference for films classified as adventure, animation, children, comedy, and fantasy, indicating an inclination toward imaginative visual storytelling, playful or whimsical tones, and family oriented narratives, while acknowledging that the evidence is limited to genre labels. They have no strong preference supported for particular plot points, themes, or specific narrative beats based on the supplied history, so it is not possible to reliably characterize favored emotional arcs, philosophical questions, or recurring character journeys beyond what the genre labels suggest.

Recommendations:

1. Blade Runner 2049 (2017)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. The Hateful Eight (2015)
4. Zootopia (2016)
5. The Revenant (2015)
6. Joker (2019)
7. Logan (2017)
8. Rogue One: A Star Wars Story (2016)
9. Get Out (2017)
10. Knives Out (2019)
11. Spider-Man: Into the Spider-Verse (2018)
12. Coco (2017)

### Soul_mixed_history — current_backend_without_negative_boilerplate — run 1

Summary: The viewer shows a tentative preference for films that blend speculative and adventurous settings with grounded emotional drama and moments of broad comedy, indicating enjoyment of stories that pair imaginative world building and kinetic action with character focused stakes rather than a single narrowly defined genre. They appear to favor narratives where high concept premises are balanced with intimate character arcs and inventive humor, though the viewing record is limited and supports only a cautious, narrow claim rather than a categorical assertion about all preferred plot elements.

Recommendations:

1. Doctor Strange (2016)
2. Logan (2017)
3. Thor: Ragnarok (2017)
4. Zootopia (2016)
5. Star Wars: Episode VII - The Force Awakens (2015)
6. Guardians of the Galaxy 2 (2017)
7. Rogue One: A Star Wars Story (2016)
8. Ant-Man (2015)
9. Spider-Man: Into the Spider-Verse (2018)
10. Captain America: Civil War (2016)
11. Avengers: Infinity War - Part II (2019)
12. Avengers: Age of Ultron (2015)

### Soul_mixed_history — current_backend_without_negative_boilerplate — run 2

Summary: The viewer's history shows positive responses to films that blend speculative science fiction imagery, kinetic action and adventurous scope with elements of comedy and animation, indicating an appetite for varied formats that mix spectacle with accessible, broadly appealing storytelling across different tones. Their selection history also suggests a preference for concept driven works that pair imaginative worldbuilding with intimate character focus and emotional exploration, favoring creative formal approaches that interweave humor, existential questions, and family oriented elements rather than rigidly conventional realism.

Recommendations:

1. Blade Runner 2049 (2017)
2. Star Wars: Episode VII - The Force Awakens (2015)
3. The Hateful Eight (2015)
4. Spider-Man: Into the Spider-Verse (2018)
5. Get Out (2017)
6. Zootopia (2016)
7. Three Billboards Outside Ebbing, Missouri (2017)
8. The Lobster (2015)
9. The Revenant (2015)
10. Knives Out (2019)
11. Your Name. (2016)
12. Coco (2017)

### Soul_mixed_history — current_backend_without_negative_boilerplate — run 3

Summary: no strong preference is supported for categorical genre likes in this profile because the available positive viewing history spans multiple broad categories without producing a clear, exclusive pattern that would permit asserting a stable liking for any single genre family across different storytelling approaches. No strong preference is supported for specific plot points, themes, or storytelling devices in this record because the examples include varied tones and narrative aims, leaving only a tentative, noncommittal indication that the viewer has sampled diverse kinds of cinematic pacing and emotional range.

Recommendations:

1. Star Wars: Episode VII - The Force Awakens (2015)
2. Logan (2017)
3. Avengers: Age of Ultron (2015)
4. Thor: Ragnarok (2017)
5. Doctor Strange (2016)
6. Blade Runner 2049 (2017)
7. Rogue One: A Star Wars Story (2016)
8. Captain America: Civil War (2016)
9. Guardians of the Galaxy 2 (2017)
10. Avengers: Infinity War - Part II (2019)
11. Spider-Man: Into the Spider-Verse (2018)
12. The Hateful Eight (2015)
