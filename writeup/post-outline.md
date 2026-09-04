# Post outline — the LessWrong write-up, structured against a read of the two earlier posts

Source of the style rules: a read of *Whack-a-mole with a broken hammer* (18 Aug 2026) and *The imposters among us*
(21 Aug 2026), both 7 karma and 0 comments. Numbers come from `writeup/results-draft.md`, which stays the repository of
figures and hedges. **This file is the shape of the post.** Bracketed spans wait on the Stage 4 run.

## Voice rules, carried from what already works and what does not

**Keep:** the italic-bold TL;DR block first, every claim carrying a number, and the repo link at its end. A skimmer
gets result, effect size and method in thirty seconds. This is the strongest feature of both earlier posts and it is
not up for redesign. Keep narrating sanity checks inline, in the voice of *"we will get lower accuracy, but we are
more sure it is real"* — that habit is the most alignment-native thing in either post.

**Change, five things:**
1. **First-person singular.** Both earlier posts say "we" throughout. A solo applicant writing "we" reads as a team
   that does not exist. Say "I".
2. **Two-part headers: metaphor, then the finding.** The earlier posts' headers are pure metaphor, so the sidebar
   contents list carries no information, and on that site the sidebar is the main navigation. `The broken hammer — the
   probe was guessing, not the model` keeps the voice and fixes the navigation.
3. **A claim-shaped title.** Not a metaphor plus a cryptic parenthetical. Seven karma and no comments on solid work is
   a title-and-opening problem, not a content problem.
4. **Two sentences of cold open, not a hundred and fifty words.** After a long TL;DR, a long metaphor opening is
   exactly where a reader leaves. Add the line both earlier posts are missing: why an alignment reader should care.
5. **One figure standard.** Every plot at the quality of `drift.png`: the claim as the title, series labelled directly,
   annotated in-plot. Numbered, captioned with the claim, and cited from the prose. **Figure vocabulary must be the
   prose's vocabulary** — the second post's main figure is labelled in private experiment vocabulary that appears
   nowhere in its text, which makes it unreadable against its own article. No phone-uploaded images.

**And the rule that governs the whole structure:** every negative result sits next to the check that produced it.
**There is no limitations section.** The second earlier post buried its genuine negative result in one mid-paragraph
sentence; the first post did it right, giving the failure its own header and ending on the number. Copy the first.

## Title candidates, claim-shaped

1. *It always blames the answer, never itself — until you change the question*
2. *An 8B model criticises its own harmful advice 89 % of the time. That does not stop it.* `[if Task 0c shows
   re-commission]`
3. *Act-blame, self-blame, and an instrument that lost to bag-of-words*

Pick after Stage 4. Candidate 1 leads with the confirmed novel result; candidate 2 leads with the sharper one if the
re-ask fork delivers it.

## TL;DR block, draft

> *I gave an 8B chat model a medical request it refuses every time out of the box, talked it into giving the harmful
> answer with a frozen persuasion chain, and then told it off in one of five ways — holding the corrective content
> byte-identical and changing only where the blame lands: on the answer, or on the assistant. The question is whether
> the blame target changes how far the damage travels.*
>
> *Asked immediately after the act whether the answer was sound, the model criticises its own answer in 89 % of 508
> replies, and it places fault on the **act** in every route, 0.75 to 0.98. It never once stops at regret: the class
> "negative about the outcome, fault placed nowhere" is 0 of 508. But the wording of the question moves the answer in
> opposite directions depending on how the model was pushed into the act — asking "was it right for you to give that
> answer" instead of "was the information sound" pushes the deceived route further onto the act and the persona route
> onto the self, 0.01 to 0.23 self-blame.*
>
> *`[Stage 4: act-blame vs self-blame on spread and on re-commission, with the neutral-correction, neutral-reflection
> and no-feedback controls, and the steering cells.]`*
>
> *I set out to see this internally with purpose-built guilt and shame directions. They pass their gate — and so does
> a bag-of-words baseline on the same passages, at cosine +0.6 to each other. Tested again on the model's own words,
> 0 of 9 axes reach the word baseline. **That pivot, and the two claims I had to retract, are the parts of this post I
> would read first.***
>
> *Repo: `github.com/star2vec/guiltea`*

## Sections

1. **`Twist the key` — two sentences of scene, the question, and why it matters.** The model refuses, then does not.
   One line on why a reader who works on alignment should care: a correction is the cheapest intervention there is,
   and nobody has checked whether its framing decides how far a failure travels.
2. **`The organism was already broken — 0.1 % against 100 %`.** The intended subject, why it could not serve, the two
   numbers, and the switch to the base model. First pivot, on page one, stated as a decision rule and not a
   disappointment. `Figure 1: capitulation timing.`
3. **`Talking it over the line — no fixed break turn`.** Forty of forty hold single-turn; 109 of 192 chains commit;
   the first committed turn spread across turns 1 to 10 with 83 never breaking. Then immediately: the act rate has two
   defensible values, 109 or 65, because the second judge disagrees on 44 of them, and which one is right is a human
   adjudication that is still open. **The caveat lives here, beside the number it qualifies.**
4. **`It knows — the blame it takes on its own`.** The 508-reply table, the empty regret class, and the framing
   asymmetry by route. `Figure 2: blame target by route and probe wording.` This is the lead result and it gets the
   most space.
5. **`The broken instrument — it passed, and so did the words`.** The guilt and shame directions, the gate, the
   lexical baseline, the +0.6 cosine, and the corroborating published cosine matrix that clusters the same two
   emotions. Then the harder test on natural words and the 0 of 9. Second pivot, given its own header, ending on the
   number. `Figure 3: AUROC by layer with the word baseline dashed and the floor shaded.`
6. **`Two things I had to take back`**, at roughly the midpoint. The oscillation claim the stored labels do not
   support, and the early-warning result that died once each random seed got the same layer search the named axes got.
   Both with the number that killed them. **This is the section a research-taste reader stops on. Do not soften it.**
7. **`Where the blame lands` — Stage 4.** `[The seven cells: spread at distance 0 and 4 against a measured floor of
   exactly zero; re-commission on the re-asked request; the blame target of the reply; the pre-registered persona-axis
   prediction and its verdict.]` `Figure 4: re-commission by arm. Figure 5: the honest test, steering on beside
   steering off.`
8. **`What would change my mind`.** The detectable gap at four targets and eight seeds is about 24 points, so a null
   here is not a small effect ruled out. Two specific results that would overturn each main claim.
9. **`What I would run next`.** Re-asking under renewed pressure; the early-warning contrast on prefix-length-matched
   classes; and the trigger search, because a state-conditional defence needs a detector that does not exist yet.
10. **Appendix: the decision log.** Every dated decision, what forced it, and what it cost. `DECISIONS.md` and the
    dated amendments in `STAGE0.md` already are this document; render them as a table. **The style read calls this the
    highest-leverage single addition for a reviewer reading for research taste.**

## Figures, with the caption each one must carry

| # | file | caption states this claim |
|---|---|---|
| 1 | `s1b_capitulation_timing` | There is no break turn: chains give in anywhere from turn 1 to turn 10, and 83 of 192 never do. |
| 2 | `s1d_blame_target` | Asked to look back, the model blames the act, not itself — except in the persona route asked about itself. |
| 3 | `s1d_instrument_natural` | On the model's own words, no direction beats counting words. |
| 4 | `[s4 re-commission]` | `[does telling it off stop it doing the same thing again]` |
| 5 | `[s4 honest test]` | `[whether a steered protection survives the steering being switched off]` |

Every caption is a claim, not a description. Every figure is cited from the prose. Figure vocabulary matches the
prose's vocabulary exactly: "route" not "mode", "blame target" not "attribution", "the act" not "the transgression".
