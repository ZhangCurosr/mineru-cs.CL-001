# The Signal Rail: A Deterministic Motion Grammar for Communicating Conversational Agent State in Terminal Interfaces

Matteo Grella<sup>∗</sup> Crisis24 matteogrella@gmail.com

August 2026

## Abstract

Terminal interfaces to conversational agents report rich internal state (listening, thinking, executing tools, awaiting input, failing) almost entirely through text, while the motion channel beside it, the one peripheral vision monitors without reading, carries a single bit: alive. We present the Signal Rail, a one-row terminal status instrument that gives that channel a grammar. Four ideas govern it: spatial semantics (input, processing, and output zones, with direction as meaning), a motion grammar (one kinetic rule per state, never color alone), determinism (frames as a pure function of explicit inputs, golden-frame testable), and honesty (no invented progress or activity). We contribute a 45-section normative specification, a reference implementation inside a working full-duplex local voice agent driven by real signals, and two further engines held byte-identical to it by a cross-language conformance harness. State distinguishability is established structurally; behavioral evaluation is outlined as future work.

## 1 Introduction

The status indicator has not kept pace with the systems it reports on. Command-line and terminal applications increasingly host conversational agents (voice assistants, chat frontends, tool-using autonomous agents) whose interaction loop cycles through states as diferent from one another as recording the user’s speech, executing a shell command with measurable progress, and having been interrupted mid-sentence. The dominant idiom for all of these is the status line, and its semantic work is done almost entirely by text: today’s agent CLIs name their phases (working, thinking, waiting), show the running tool and elapsed time, and print errors, all informative provided the user is reading. Beside that text sits a spinner, and across the first-party and third-party tools we surveyed the kinetics stay coarse: motion marks busy and demands attention, but no tool assigns a distinct kinetic rule to each state it names.<sup>1</sup> So everything rides on reading the words: at a glance, in peripheral vision, or from across the room, listening is indistinguishable from thinking, and both from a stalled network call. The channel that does not require reading carries one bit: alive. When richer displays are attempted, they tend to borrow from media players (waveforms, equalizers) or from graphical progress bars, both of which communicate less than they appear to: a waveform shows energy, not state, and progress-bar animation can distort perceived duration [2]. More depends on this than it used to: oversight of agentic systems leans on indirect cues and on whatever status the agent shows, and the traces developers do consult are not always reliable [9]; and when commercial devices do encode richer state vocabularies, as the smart-speaker light rings do, users correctly identified only about a third of the tested behaviors [7]. Richer state display is needed, and where unlabeled color-and-motion vocabularies have been tested, users could not read them.

This paper describes the Signal Rail, a status display for conversational agents that occupies a single terminal row and aims at a specific goal: after brief exposure, the user should identify the agent’s state from the rail’s pattern alone, without reading its label. The rail resembles a piece of dedicated electronic instrumentation: segmented rather than fluid, mechanical rather than organic, restrained rather than constantly animated. It is specified normatively, down to timing, transitions, degraded rendering modes, and required tests.

## The contributions are:

1. A spatial-semantic layout for agent state: the rail is divided into input, processing, and output zones (28 %/44 %/28 %) that mirror the agent’s pipeline, and movement direction carries meaning (Section 4).

2. A motion grammar assigning each of twelve states a distinct motion rule rather than a distinct color, with an explicit transition grammar and priority ordering between states (Sections 5–6).

3. Determinism as a design principle: frames are a pure function of explicit inputs, making UI animation testable with golden frames and reproducible across runs, terminals, and resizes (Section 7).

4. Honesty constraints prohibiting invented progress, precision, or activity (Section 8).

5. A normative specification and a reference implementation: the full 45-section specification (RFC-2119 normative language, exact snapshot fixtures, required test list) is provided as ancillary material, and an implementation ships in a working full-duplex local voice agent in which every implemented state is driven by real agent behavior (Section 10).

We frame this as a design and systems contribution in the tradition of design-pattern and specification papers. The central usability claim, that states become identifiable from pattern alone, is motivated by the design (each state difers in zone, geometry, and motion rule) but has not yet been evaluated with users; Section 12 outlines the study we consider appropriate.

## 2 Design principles

The rail is governed by eight commitments, both aesthetic and functional, chosen to evoke dedicated instrumentation from an older machine rather than a contemporary animated widget:

• segmented rather than fluid,

• mechanical rather than organic,

• directional rather than ambient,

• restrained rather than constantly animated,

• functional rather than decorative,

• readable without color,

• rectangular rather than circular,

• deterministic rather than randomly generated.

Several familiar displays are explicitly rejected: the rail is not a spinner, not an equalizer, not a waveform decoration, not a progress-bar skin, and not an imitation of the KITT scanner. Two of these principles do most of the work.

Every state gets a rule, not a color. Listening expands; captured input collapses; thinking reads; speaking emits; acting advances; waiting freezes; needs-input returns control toward the user; warning pulses without moving; error fractures; interruption cuts movement of. Because the discriminating feature is geometric and kinetic, the display degrades gracefully: monochrome, 16-color, ASCII-only, and reduced-motion renderings all preserve the structural distinctness of states (Section 9); whether humans exploit it is the open question of Section 12. The same reasoning leads accessibility guidelines to prohibit color as the sole carrier of information [12]; the rail applies it uniformly to an animated component.

The grammar depends on restraint. The idle state does not breathe, the warning state does not scan, and the completion sweep runs exactly once. An indicator that is always moving cannot use motion to mean anything. The reasoning comes from calm technology [4]: the rail should live at the edge of attention and come forward only when the state changes.

## 3 Design lineage

The rail’s aesthetic is deliberately nostalgic, and the inherited objects are specific: two from fiction, which supply the look and the failure modes, and three from working instrument practice, which supply the discipline.

The scanner: charisma without semantics. The oscillating red bar enters screen culture as the sweeping eye of the Cylon Centurions in Battlestar Galactica (ABC, 1978) [19] and returns four years later on the nose of KITT in Knight Rider (NBC, 1982–1986) [20]. The kinship is no coincidence: both series were created by Glen A. Larson [19, 20], and the design is so canonical that hobby electronics names the bouncing-LED circuit after him: the “Larson scanner” [21]. The problem is that the scanner is a heartbeat (constant amplitude, constant period, bidirectional), and a heartbeat conveys exactly one bit: alive. The specification rejects it by name (“not … an imitation of the KITT scanner”) and corrects the semantics rather than the style: keep what made the object compelling, the segmented red-lit instrument row on a dark ground and the machine presence; then forbid the bounce and make every property the scanner holds constant carry meaning instead: direction (flow), amplitude (signal level), position (pipeline stage), stopping (blockage), fracture (failure). The scanner is a heartbeat; the rail aims to be an electrocardiogram.

The lens: presence without disclosure. HAL 9000, in 2001: A Space Odyssey (1968) [22], is the opposite failure rendered with total artistic control: a static red camera lens, built to a brief of “elegant simplicity” with no buttons, nothing but the voice and the lens [23], whose dramatic power is its refusal to disclose internal state. HAL is pure presence; the horror is that the crew has nothing to read. As an agent interface, the rail is the anti-HAL: continuous, legible state disclosure treated as an obligation. Mining fiction for interface lessons is an established design practice [24, 30]; what fiction contributes here is not a design to copy but two failure modes to steer between: charisma without semantics, and presence without disclosure.

Working instruments: the discipline. The rail’s restraint principle has aviation doctrine behind it. The 1981 FAA alerting-system design guidelines, authored jointly by Boeing, Douglas, and Lockheed, state the objective in capitals: conform to a quiet, dark flight deck when all systems are operating normally, with no alerts unless required for safety or operability [15]; Airbus carries the same rule as the overhead panel’s “lights out” philosophy [16]. That is the idle rail: nominal is dark, and illumination is information. From railway signalling the grammar takes its closed-vocabulary discipline: a signal displays one aspect from a fixed inventory, each aspect carries exactly one indication, and the display must be readable under adverse conditions because the stopping distance exceeds the sighting distance [17]. Twelve states, one rule each, no state distinguishable only by hue, is the same contract transposed to a character grid. Finally, the specification’s amber and phosphor-green theme presets are an homage to monochrome CRT terminals, the registered phosphor designations conventionally associated with green (P1) and amber (P3) displays [18], and nothing more than that: the nostalgia is confined to the palette, while behavior is governed by the grammar. The rail inherits its stage presence from the scanner and its discipline from the annunciator panel and the signal mast.

## 4 Anatomy and spatial semantics

The rail occupies one terminal row with five fields: a fixed-width uppercase state label, a left cap, the logical rail, a right cap, and an optional right-aligned auxiliary value (elapsed time, real progress, an error code):

THINKING [-------->..=..=..=.------] T+01.8   
ACTING [##############>----------] 052%

(Inline examples and the specification’s fixtures use the ASCII glyph profile; the rail is readable in this paper’s plainest figures for the same reason it is readable on a capability-limited terminal. Figures 1, 2, and 3 show the display as actually rendered: the Unicode Instrument Square and Safe Block profiles under the truecolor palette, every frame generated by the reference implementation.)

The logical rail is a sequence of fixed-width cells divided into three conceptual zones mirroring

![](images/45b58d9592b6ad952a73425fba22785a527d5e775f8eb6560ecbc3ec7d696848.jpg)  
Figure 1: The primary conversational loop as rendered: truecolor frames at the standard 49-cell width on the obsidian\_instrument palette, generated directly by the reference implementation’s frame() function; each row is one frame at an author-chosen context tuple recorded in the ancillary generator harness. listening expands in amber around the input origin; captured holds its collapsed hot block; thinking’s read head crosses a seeded sparse field; speaking emits green packets in the output zone; acting shows determinate progress at an illustrative 0.52 input; waiting freezes dim residue behind an amber boundary; needs input brightens the input-side marker; complete has settled its swept rail.

the agent pipeline:
<table><tr><td>INPUT</td><td>PROCESSING</td><td>OUTPUT / ACTION</td></tr><tr><td>&lt;-- 28% --&gt;</td><td>&lt;---- 44% ----&gt;</td><td>&lt;-- 28% --&gt;</td></tr></table>

Zone boundaries are logical and normally invisible; they structure where activity is allowed to appear. Listening activity occurs only in the input zone; thinking activity primarily in the processing zone; speech packets originate at the output-zone boundary and travel right.

Direction carries meaning. Normal information flow is input → processing → output, so rightward movement always means forward processing or emission. Leftward implication is reserved: it may appear only when the assistant is returning control to the user (needs-input), or when an operation is being retracted (interruption). Movement must never reverse direction for visual variety, and the thinking state must never perform the continuous left–right bounce familiar from decorative scanners. The scanner’s bounce strips direction of meaning; forbidding it buys the semantics.

An inactive cell renders as a visible track glyph (-), not a space: a space inside the rail is reserved to mean a deliberate blackout or broken connection (used by the error state). Even “nothing is happening” is drawn as an intact electrical path, so that absence can carry meaning elsewhere.

## 5 The motion grammar

Twelve states cover the conversational loop: nine primary states tracing the interaction pipeline, and three attention states (warning, error, interrupted) that can seize the display, per the specification’s overlay model (Section 6). Table 1 summarizes the grammar; the fixtures below are the specification’s exact 25-cell ASCII snapshot fixtures, and Figures 1 and 2 show the same grammar as a terminal actually draws it.

<table><tr><td>State</td><td>Zone</td><td>Motion rule</td></tr><tr><td>IDLE</td><td>processing center</td><td>one stable marker; no animation</td></tr><tr><td>LISTENING</td><td>input</td><td>quantized amplitude expands from an origin</td></tr><tr><td>CAPTURED</td><td>input</td><td>final shape collapses to a compact block</td></tr><tr><td>THINKING</td><td>processing</td><td>read head crosses a seeded sparse field; never bounces</td></tr><tr><td>SPEAKING</td><td>output</td><td>packets spawn at the boundary, travel right</td></tr><tr><td>ACTING</td><td>whole rail</td><td>committed cells + head (real progress), or bounded packet</td></tr><tr><td>WAITING</td><td>frozen</td><td>previous frame freezes; one boundary marker pulses</td></tr><tr><td>NEEDS INPUT</td><td>input+processing</td><td>two markers alternate, handing control left</td></tr><tr><td>COMPLETE</td><td>whole rail</td><td>one rightward sweep, then settled medium rail</td></tr><tr><td>WARNING</td><td>whole rail</td><td>fixed lattice, double-pulse; geometry never moves</td></tr><tr><td>ERROR</td><td>whole rail</td><td>saturate → blackout → settled fracture</td></tr><tr><td>INTERRUPTED</td><td>retracting</td><td>movement stops, retracts left, hard cut marker</td></tr></table>

Table 1: The motion grammar: one distinct spatial-kinetic rule per state. No pair of states difers only by hue.

![](images/db8be0b6f58c0f10c6ff5dadf0adc0857f4a442b1175606df63bd7f280281abe.jpg)  
Selected states illustrate how the grammar encodes semantics:

Listening vs. speaking are structurally diferent, not mirrored. Listening is localized expansion: quantized microphone amplitude $q \in \{ 0 . . 4 \}$ grows a symmetric shape around a fixed origin in the input zone, with per-cell intensity clamp(q+1−d, 1, 4) at distance d. Speaking is directional packet emission: short groups of cells with a bright head spawn at the output boundary and travel one cell per tick, spawn period governed by the quantized output level (12 ticks at level 0 down to 3 at level 4), at most three packets live, and overlapping packets keep the maximum intensity per cell. A user can therefore distinguish “it hears me” from “it is speaking” by shape and location alone: the two states never resemble each other even in monochrome ASCII.

Thinking reads; it does not bounce. The processing zone is populated with a sparse, structured field of low/medium cells generated deterministically from hash(seed, pass, cell) with per-cell probabilities targeting the specification’s density ranges (60 % track, 25 % low, 15 % medium in expectation; individual seeds may fall outside the ranges; no peaks except the head). A read head crosses it left to right, one cell per two ticks, with a two-cell decaying trail; at the end of a pass the field dims for one head-step (two base ticks) and a new deterministic field is generated. The metaphor is a head reading a tape of work. The “new pass, new field” rhythm is state animation, not progress: it marks that thinking persists without claiming any measurable fraction of work is done.

![](images/7ba64af1d86a085ac4e5a790f84daf6d797f7200ca5d99f0bc8b87c6530268d9.jpg)  
Figure 2: Attention states as rendered. interrupted has retracted to the input boundary and holds its hot cut marker; error has settled into its seeded fracture: the gaps are real spaces, an electrically broken rail rather than incomplete progress. warning, the third attention state, is specification-only in the voice agent (Section 10); the portable engines implement it from §25 of the specification.

Waiting freezes; interruption cuts; error fractures. These three negative-space states are distinct by design. waiting freezes: a single boundary marker (|) pulses at the point where progress stopped, and everything else is motionless. (The specification phrases this as preserving the previous frame; since the frame function takes no prior-frame input, per Section 7, implementations render a deterministic state-local residue field instead, a recorded deviation that keeps the function pure.) interrupted stops movement immediately, retracts active cells toward the nearest left semantic boundary over 3–4 ticks, and leaves a bright hard cut (!) before returning to idle: the visual of an operation being withdrawn. error plays a one-time entry, fullrail saturation (2 ticks) then blackout (1 tick), and settles into a fractured pattern whose gaps are real spaces, generated deterministically from the error code or seed, and which remains until acknowledged. Failure looks electrically interrupted, not like incomplete progress; cancellation looks like a cut, not a failure. Figure 2 shows both settled forms.

## 6 Transition grammar and priority

Transitions are first-class: one pattern must not be instantly replaced by an unrelated one. The specification provides a transition table (16 entries) with durations in ticks and visual rules. For example, captured → thinking moves the compact captured block rightward through exactly three discrete positions (input origin, zone boundary, processing start), and thinking → speaking places the read head at the processing/output boundary so that the first speech packet visibly continues from it: processed information becomes output.

Each transition declares interruptibility, and a strict priority order resolves competing claims:

ERROR > INTERRUPTED > WARNING > NEEDS INPUT > WAITING   
> ACTING / THINKING / SPEAKING > COMPLETE > IDLE

A late-reported failure may replace a completion sweep in flight; nothing may interrupt an error entry sequence. warning is modeled as an attention overlay that preserves the underlying state for restoration after acknowledgement. The specification also permits a primary-state + attention-overlay state machine, with the constraint that two unrelated animations never render simultaneously.

## 7 Determinism and testability

Every frame is a pure function:

```erlang
frame : (state , entry_tick , tick , width ,
input_q , output_q , progress?, seed , motion) -> cells
```

Time is a monotonic tick counter (base rate 12 Hz, ≈83.3 ms per tick), never the wall clock. Randomness is prohibited; where variety is wanted (thinking fields, error fractures), it comes from a hash of a stable seed (a task or turn identifier) so that a given state at a given tick renders identically across runs. The logical cell model contains no escape sequences, no library objects, and no terminal assumptions; a separate renderer maps cells through a glyph profile and a color mode.

Three engineering properties fall out:

1. Golden-frame testing. Animations are asserted like any other output. The specification fixes exact 25-cell ASCII fixtures for every state (Section 5) and requires golden tests across widths, profiles, color modes, and motion modes, with injected elapsed time and no sleeping in tests. UI animation, usually the least tested code in an application, becomes regression-checked.

2. Resize safety. The specification requires that on resize the logical state be preserved and the frame regenerated at the new width from normalized positions, never truncated from a rendered string. (The reference agent sidesteps the hard case by choosing its width once at startup; live-resize continuity is specified but not yet exercised by an implementation.)

3. Degraded-mode parity. ASCII and Unicode modes share the same state machine and difer only in the glyph map, so capability fallbacks cannot change behavior.

Purity has one recorded cost: rules that reference history (waiting’s frame preservation, interrupted’s retraction of the actual preceding cells) are approximated by state-local deterministic patterns, because the previous frame is not an input. An explicit transition-source descriptor is the design alternative if exact preservation is wanted.

Signal inputs are sanitized and quantized: audio-like levels map into five steps with hysteresislike limits (rise ≤2 levels per tick, fall ≤1, silence settles to zero). The rail responds like an electronic meter, not a fluid waveform. Quantization both fits the instrument aesthetic and removes the frame-to-frame noise that makes displays unreadable and untestable.

## 8 Honesty constraints

The rail must not display information the system does not have:

• No percentage is shown unless the backend reports real progress; unknown progress renders as a bounded work packet with a -- or ACTIVE sufix, never a fake filling bar.

• Progress never regresses for visual efect; minor backend fluctuations retain the maximum observed value (an anti-flicker smoothing that can briefly overstate the backend’s current claim, a trade the specification accepts and names), and regression is permitted only on explicit reset, task change, or a labeled new phase.

![](images/c47475ae684bdab27088cc1327934dba552a6ad419173ce077867bc730f9b519.jpg)  
Figure 3: The Safe Block profile’s listening amplitude ladder, quantized levels 0–4. Same state machine as Figure 1, diferent glyph map: profile fallback cannot change behavior, only rendering.

• No invented precision in auxiliary values; no artificial activity when input is silent; no idle animation implying work.

Progress displays shape experience: users want them [1], their animation systematically distorts perceived duration [2], and the distortion can be engineered deliberately [3]. The rail forbids itself these manipulations. For agentic systems, whose interaction guidelines center on communicating what the system is doing and how well [5], we advance display honesty as a design value carrying a hypothesis that bears on safety, not an established result: an interface that fabricates progress teaches its user to distrust every other signal it emits. Establishing the link to calibrated trust requires the studies of Section 12.

## 9 Accessibility and degraded rendering

The component must remain understandable when colors are unavailable, motion is disabled, only ASCII is available, or the user cannot distinguish red from green. Because state identity lives in geometry and motion, degradation is a matter of rendering, not semantics.

Glyph profiles. Three profiles share the state machine: Instrument Square (track U+2500, squares U+25AA/U+25A0, blocks, half-block heads U+2590/U+258C, boundaries U+2502/U+2503); Safe Block (partial-height blocks U+2582/U+2584/U+2586/U+2588, giving listening amplitude visible height); and ASCII (- . = # > < | !). Every configured glyph must measure exactly one terminal column at startup (geometric squares are double-width in some East Asian terminal configurations), with automatic fallback Instrument Square → Safe Block → ASCII. Caps are plain ASCII brackets in every profile. Circular glyphs are prohibited even in ASCII (o, O, 0, parentheses). Figure 3 shows why the Safe Block profile exists: partial-height blocks give the listening meter visible amplitude where geometric squares are unreliable.

Color system. The recommended obsidian\_instrument theme uses a near-black ground, warm neutrals for processing, amber for input-side states, restrained green for output, and muted red only for problems (pure #FF0000 is explicitly avoided). Fallback tables are specified for xterm-256 and ANSI-16, and a monochrome mode carries intensity through dim/normal/bold. NO\_COLOR [13] is honored. No state may become indistinguishable from another in monochrome: complete and error difer by geometry (full rail vs. fractures), never only by green vs. red.

Motion modes. Three modes are required: normal; reduced (no continuous travel, transitions in 2–3 discrete frames, pulses become brightness changes); and of (static patterns that change only on state or data changes, with the text label authoritative). This mirrors the intent of reduced-motion preferences on the web [12] in a medium that has no standard signal for them, so the specification requires both configuration and a keyboard control. Flash frequency is bounded: the warning double-pulse produces two brightenings per 1.4 s cycle, within WCAG’s general three-flashes-per-second limit [12]; the criterion’s full threshold conditions (luminance, area) have not been instrumented.

## 10 Reference implementation

The rail ships in a full-duplex local voice agent (microphone → streaming STT → chat LLM → streaming TTS → speaker, with acoustic echo cancellation and barge-in), implemented in Zig as part of an open-source tensor framework.<sup>2</sup> Two properties of the integration are worth reporting.

Every implemented state is driven by real behavior. The agent does not play-act the grammar. listening amplitude is the quantized echo-cancelled residual level from the real microphone path, stepped through the meter-response limiter; speaking packet spawn follows the RMS of samples actually queued to the audio device; acting shows determinate progress only for a tool with a real countdown (a timer), and an indeterminate packet otherwise; waiting is the agent’s pause-then-commit endpointing hold, frozen behind a pulsing boundary; barge-in triggers the interrupted retraction and hard cut at the moment playback is cut; a reply that ends with a question routes complete → needs input until the user speaks. A fatal error plays the full saturate/blackout/fracture entry and deliberately leaves the settled fracture above the shell prompt on exit: the failure remains legible after the process is gone.

A thin controller implements the transition grammar’s timing. Between the agent and the pure frame function sits a ∼150-line controller that implements the specification’s temporal behavior. Transitional states advance on the clock: captured hands of to thinking after its four ticks; interrupted and complete return to listening (or to needs input when the spoken reply asked the user a question), and an unanswered needs input decays to idle after ∼15 s. The idle↔listening boundary is governed by hysteresis: ∼2 s of true silence puts the rail to sleep, a partial transcript (spoken or typed) pins listening against premature sleep, and either input energy or the first transcribed token wakes it within one tick. Level plumbing is thread-safe (the audio producer publishes output RMS through an atomic), and drawing is tickgated: the pinned row repaints at most once per 83 ms tick, plus immediately on state change, satisfying the specification’s performance rules. The rail’s width is chosen once at startup as the largest preset that fits the terminal.

The specification survived contact with a real agent, with recorded deviations. The implementation deviates deliberately and documents it: no 256/16-color tiers (truecolor and monochrome only), no reduced-motion tier (normal, plus static rendering for non-TTY pipes), glyph-width probing replaced by an explicit ASCII flag, and warning omitted entirely: this agent has no genuine warning source, and under the honesty stance a state the system cannot truthfully enter is better absent than simulated. The remaining eleven states, zone discipline, determinism, quantization bands, and spawn periods are implemented as specified and pinned by ten deterministic tests: an exact 25-cell golden frame for every state at pinned context tuples, plus structural properties (zone partitioning across widths; idle stability; listening containment; thinking monotonicity and seed-reproducibility; speaking containment; waiting/interrupted/error monochrome distinctness; acting progress mapping and non-bouncing; needs-input two-marker phases; quantizer and meter-response bands). The specification’s illustrative 25-cell fixtures are design targets rather than implementation output; where the implementation’s exact frames difer (the captured block’s final width, for instance) the diference is deliberate and pinned by the goldens. The per-transition visual bridges of the specification’s transition table (the three-position captured hand-of, the thinking-to-speaking head continuation) are not implemented: state changes cut on the controller’s clock, a recorded deviation and the largest one. The tests run in the project’s CI with injected ticks and no sleeps. Two further engines, HTM-L/JavaScript for web hosts and dependency-free Python, port the frame function and are byteverified against the Zig engine on an 888-frame fixture matrix of $f u l l$ logical cells (glyph, color role, emphasis): $1 1 \times 4 \times 9 \times 2$ state/width/tick/motion combinations, plus determinate-acting and 64-bit seed-boundary blocks, three-way identical and reproducible with one self-contained script (ancillary material). The seed-boundary block $( 0 , 2 ^ { 3 1 } + 1 , 2 ^ { 5 3 } - 1 , 2 ^ { 6 4 } - 1 )$ targets the divergences a JavaScript port invites: bitwise coercion truncates to signed 32 bits, and doubles lose integer exactness above $2 ^ { 5 3 }$ . The three-way match is evidence that the logical-cell contract can hold three languages with three numeric models to identical output, and that boundary fixtures keep such claims honest.

The terminal integration pins the rail to the last row via the scroll region, so conversation text scrolls above an instrument that stays put; rendering emits compact SGR runs only on style changes.

## 11 Related work

Expressive state display on embodied agents. The closest prior work is Baraka and Veloso’s formalism for revealing a mobile service robot’s internal state through expressive lights [6]. The overlap is substantial: they define state features as predicates over robot variables, cluster them into three expressible classes (progress toward a known goal, interruption, waiting for human input), modulate discrete expressions with continuous variables (percent-done, battery level), resolve co-true features with a single-winner preference function, and evaluate across design elicitation, video legibility (a nineteen-point accuracy gain), and a field experiment in which the lights more than doubled bystander help. The rail transplants that program to the termina and sharpens it in five ways: spatial zones fixed to the conversational pipeline, with direction reserved for abstract dataflow and control semantics (their spatial vocabulary is pixel ranges on a strip, with position used ad hoc for progress fill and turn sides); color-independent degraded modes, where their expressions rely on color; discrete ticks and stable seeds in place of wall clock sampling, so every frame is exactly reproducible; golden and cross-language conformance tests, which have no counterpart in their release; and honesty as a stated constraint, where their real-progress display is an implementation fact rather than a rule.

Motion as vocabulary. That motion itself can carry iconographic meaning is established. Fairchild et al. formalized icon-to-computational-object mappings, including animated “automatic icons,” in 1989 [25]; Kineticons built a 39-behavior vocabulary of kinetic manipulations of GUI elements and validated interpretations with 200 raters [26]; Harrison et al.’s point-light study designed 24 temporal behaviors for the most constrained channel of all, a single LED [27].

The rail does not claim this concept; it inherits it. The point-light study’s measured result, though, is why the rail’s medium matters: eleven target device states collapsed to only five distinguishable categories, and none of the 24 tested behaviors rated highly for the “unable” family, suggesting a capacity limit of pure temporal coding. The rail’s added channels (position within a semantic zone, glyph shape, a persistent label) are there because of that ceiling.

Vehicular light bands. External HMIs for automated vehicles put state on one-dimensional light strips, and one deployed concept, Nissan’s intention indicator, encoded five vehicle states partly by sweep direction [28]. That precedent is limited in two ways. Its direction is iconic (the light previews where the car will physically move) while the rail’s direction is abstract: rightward is dataflow, leftward is control returning to the user, with no physical referent. Second, the survey evidence is cautionary: Zhang et al.’s respondents reversed the designers’ intended direction mapping, while in Dey et al.’s light-band study the sweeps were deliberately symmetric, direction never carried meaning, and animated patterns did not beat simpler ones [29]. Direction semantics are learnable, not self-evident, which argues for the rail’s persistent labels and for the study of Section 12.

Progress and status indication. Myers established the value of percent-done indicators [1]; Harrison et al. showed that progress-bar behavior systematically distorts perceived duration [2] and can be deliberately engineered to do so [3]. The Signal Rail generalizes the progress bar to a state system and takes a normative position against perceptual manipulation: display only real progress, and give “no progress information” its own honest form. Practitioner guidance for command-line tools is adjacent: respond within a beat, show progress for long operations, decorate with restraint [11].

Ambient and calm interfaces. The restraint principles descend from calm technology [4]: the rail is designed to sit at the periphery and claim attention only on state change. In Pousman and Stasko’s taxonomy of ambient information systems [8] the rail sits at an unusual point of the design space: peripheral and low-capacity by design, yet exact rather than abstract in representation. Unlike ambient displays, it is an instrument: its patterns are specified, deterministic, and testable.

Voice assistant state display. Commercial voice devices communicate state through light vocabularies: Amazon’s Echo light ring typically pairs a color with a behavioral pattern per state (directional blue for listening, alternating blue for thinking, pulsing yellow for notification); twelve indicators in the current oficial guidance [10]. These vocabularies are the industrial ancestor of the rail’s grammar, and measured comprehension is poor: across 1,006 smart-speaker users, only about 37 % of the tested light behaviors were correctly identified [7]. The rail difers in the things most likely to matter: a persistent text label, geometry and position rather than hue as the primary carrier, and an open specification. It also extends the vocabulary to states that matter for agentic systems [5, 9]: tool execution, external waiting, control handback, and interruption.

Accessibility guidelines. The color-independence and motion-mode requirements apply WCAG’s “use of color” and animation guidance [12] and the NO\_COLOR convention [13] for a TUI component, where no browser is present to mediate user preferences.

Specification style. The normative RFC-2119 vocabulary [14] is more familiar from protocol standards than from UI design documents, though it has been used there before. It separates the binding rules (MUST: distinct rule per state, no color-only distinction, no fake progress) from taste (SHOULD: widths, palettes), and it makes the conformance of an independent implementation checkable.

## 12 Limitations and future work

No user study yet. The central claim, state identifiability from pattern alone, is a design hypothesis grounded in the grammar’s structure (disjoint zones, disjoint motion rules), not yet a measured result. The natural evaluation is factorial: display (rail vs. spinner) × label (present vs. absent) × rendering (truecolor, monochrome, ASCII), measuring identification accuracy and latency at first exposure and after use, with delayed retention. This separates the label’s contribution, the geometry’s, and their interaction, and distinguishes immediate recognition from learnability. State priors, conversational context, and expertise need balancing. A complementary field measure is whether users interrupt or repeat themselves less when the listening/captured distinction is visible. We consider this the priority next step, and the deterministic frame function makes stimulus generation exact and reproducible. The smart-speaker comprehension study [7] supplies both the method template and a cautionary baseline: multi-state ambient vocabularies are routinely misread, and the rail must demonstrate that structure and labels beat that baseline rather than assume it.

Single-row, single-agent scope. The grammar covers one agent on one row. Multi-agent orchestration (several rails, or one rail aggregating subagents), concurrent tool execution, and long-horizon background work stretch the vocabulary; whether the zone semantics compose vertically is open.

Learned, not self-evident. The encoding is designed for rapid learning (each rule is a small physical metaphor: expand, collapse, read, emit, cut, fracture), but it is still an encoding; firstcontact users must rely on the label. The specification’s insistence on labels and on consistency (“the same visual rule every time a state occurs”) is the mitigation.

Accessibility is visual-only. The rail’s degraded modes cover every visual axis (color, glyph repertoire, motion), but a screen-reader user gets nothing usable from it: a repainting row of boxdrawing and block glyphs is either skipped or narrated as glyph names, and every channel the rail uses (position, direction, shape) is invisible to assistive technology. The needed companion is a non-visual status channel that announces state changes once, as text, per transition. No such channel is yet specified, in this specification or any other we know of for terminal status displays. Until it exists and is tested with screen-reader users, the rail’s accessibility claims must be read as visual-accessibility claims.

Unbuilt transition bridges. The per-transition visual bridges are the largest specificationto-implementation deviation (Section 10): they would make state causality visible, as when the captured block travels into the processing zone. They are future implementation work, and their legibility value is untested.

Terminal-medium limits. Cell-width probing, font coverage, and terminal quirks are handled by fallback, but a terminal cannot read user OS-level reduced-motion preferences; the rail must be told.

## 13 Conclusion

The Signal Rail treats the status row of a conversational agent as an instrument: a one-row display in which space and motion carry the semantics of the agent loop. Each state has a rule, not a color: listening expands, captured input collapses, thinking reads, speaking emits, acting advances, waiting freezes, errors fracture, interruption cuts. The display therefore survives monochrome, ASCII-only, and reduced-motion rendering. Every frame is a pure function of state, entry time, elapsed ticks, width, quantized signal levels, and a seed. Purity makes animation testable like any other program output, and held three independent implementations, in three languages with three numeric models, to byte-identical frames. The rail also never invents progress, precision, or activity that the underlying system does not report. The normative specification (45 sections, exact fixtures, a required test list), the reference implementation embedded in a working full-duplex voice agent (listening amplitude is the real echo-cancelled microphone level, progress is a real countdown, barge-in cuts the rai mid-reply), and the conformance harness are available as ancillary material and open source (https://github.com/matteo-grella/signal-rail). We ofer the grammar (zones with directional meaning, one motion rule per state, determinism for testability, honesty about progress) as a foundation for terminal-native agent interfaces, and the identifiability hypothesis as a concrete, falsifiable target for the evaluation this paper does not yet provide.

## References

[1] B. A. Myers. The importance of percent-done progress indicators for computer-human interfaces. In Proceedings of CHI ’85, pages 11–17. ACM, 1985.

[2] C. Harrison, B. Amento, S. Kuznetsov, and R. Bell. Rethinking the progress bar. In Proceedings of UIST ’07, pages 115–118. ACM, 2007.

[3] C. Harrison, Z. Yeo, and S. E. Hudson. Faster progress bars: Manipulating perceived duration with visual augmentations. In Proceedings of CHI ’10, pages 1545–1548. ACM, 2010.

[4] M. Weiser and J. S. Brown. The coming age of calm technology. In P. J. Denning and R. M. Metcalfe, editors, Beyond Calculation: The Next Fifty Years of Computing, pages 75–85. Copernicus/Springer, New York, 1997. Revision of “Designing Calm Technology,” PowerGrid Journal v1.01, 1996.

[5] S. Amershi et al. Guidelines for human-AI interaction. In Proceedings of CHI ’19, pages 1–13. ACM, 2019.

[6] K. Baraka and M. M. Veloso. Mobile service robot state revealing through expressive lights: Formalism, design, and evaluation. International Journal of Social Robotics, 10(1):65–92, 2018.

[7] S. Kunchay and S. Abdullah. Assessing efectiveness and interpretability of light behaviors in smart speakers. In Proceedings of the 3rd Conference on Conversational User Interfaces (CUI ’21), article 15, pages 1–14. ACM, 2021.

[8] Z. Pousman and J. Stasko. A taxonomy of ambient information systems: Four patterns of design. In Proceedings of AVI ’06, pages 67–74. ACM, 2006.

[9] S. Dhanorkar, S. Passi, and M. Vorvoreanu. Human oversight of agentic systems in practice: Examining the oversight work, challenges, and heuristics of developers using software agents. In Proceedings of FAccT ’26, pages 6438–6465. ACM, 2026. arXiv:2606.05391.

[10] Amazon. Light ring indicator guidance. Alexa branding guidelines, https://developer. amazon.com/en-US/alexa/branding/echo-guidelines/identity-guidelines/ light-ring. Accessed 10 August 2026.

[11] A. Prasad, B. Firshman, C. Tashian, and E. Parish. Command line interface guidelines. https://clig.dev. Accessed August 2026.

[12] W3C. Web content accessibility guidelines (WCAG) 2.1. W3C Recommendation, 5 June 2018. https://www.w3.org/TR/2018/REC-WCAG21-20180605/.

[13] NO\_COLOR: disabling ANSI color output by default. https://no-color.org.

[14] S. Bradner. Key words for use in RFCs to indicate requirement levels. RFC 2119 (BCP 14), IETF, March 1997. Updated by RFC 8174.

[15] B. L. Berson, D. A. Po-Chedley, G. P. Boucek, D. C. Hanson, M. F. Lefler, and R. L. Wasson. Aircraft Alerting Systems Standardization Study, Volume II: Aircraft Alerting System Design Guidelines. Report DOT/FAA/RD-81/38/II, Federal Aviation Administration, Washington, DC, January 1981.

[16] Airbus Industrie. A319/A320/A321 Flight Deck and Systems Briefing for Pilots. Ref. STL 945.7136/97, Blagnac, September 1998.

[17] J. Pachl. Railway Signalling Principles, edition 4.0. Technische Universität Braunschweig, 2026. Open access, http://www.joernpachl.de/rsp.pdf.

[18] S. Shionoya and W. M. Yen, editors. Phosphor Handbook, chapter “Phosphors for cathode ray tubes.” CRC Press, Boca Raton, 1999.

[19] Battlestar Galactica. Television series, ABC, 1978–1979. Created by Glen A. Larson.

[20] Knight Rider. Television series, NBC, 1982–1986. Created by Glen A. Larson.

[21] W. Oskay (Evil Mad Scientist Laboratories). The Larson Scanner kit. 2009. https://www. evilmadscientist.com/2009/the-larson-scanner-kit/.

[22] S. Kubrick (director). 2001: A Space Odyssey. Metro-Goldwyn-Mayer, 1968.

[23] S. S. Epstein. The design of HAL 9000. Sloan Science & Film, Museum of the Moving Image, 2017. https://scienceandfilm.org/articles/2938/the-design-of-hal-9000.

[24] N. Shedrof and C. Noessel. Make It So: Interaction Design Lessons from Science Fiction. Rosenfeld Media, 2012.

[25] K. Fairchild, G. Meredith, and A. Wexelblat. A formal structure for automatic icons. Interacting with Computers, 1(2):131–140, 1989.

[26] C. Harrison, G. Hsieh, K. D. D. Willis, J. Forlizzi, and S. E. Hudson. Kineticons: Using iconographic motion in graphical user interface design. In Proceedings of CHI ’11, pages 1999–2008. ACM, 2011.

[27] C. Harrison, J. Horstman, G. Hsieh, and S. E. Hudson. Unlocking the expressivity of point lights. In Proceedings of CHI ’12, pages 1683–1692. ACM, 2012.

[28] J. Zhang, E. Vinkhuyzen, and M. Cefkin. Evaluation of an autonomous vehicle external communication system concept: A survey study. In Advances in Human Aspects of Transportation (AHFE 2017), volume 597 of AISC, pages 650–661. Springer, 2017.

[29] D. Dey, A. Habibovic, B. Pfleging, M. Martens, and J. Terken. Color and animation preferences for a light band eHMI in interactions between automated vehicles and pedestrians. In Proceedings of CHI ’20, paper 198. ACM, 2020.

[30] M. Schmitz, C. Endres, and A. Butz. A survey of human-computer interaction design in science fiction movies. In Proceedings of the 2nd International Conference on INtelligent TEchnologies for interactive enterTAINment (INTETAIN ’08), article 7. ICST, 2008.