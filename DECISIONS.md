# Development decisions

Short records of the choices that shaped the system, in the order they were made.

1. **One lesson file is the single source of truth.** Video, captions, chapters, SEO pack and
   web page are all generated from `lessons/<id>/lesson.yaml`. Nothing is hand-edited downstream.
2. **Numbers are never typed into a lesson.** A lesson may only quote `{placeholders}` that a
   tested module publishes through `lesson_facts()`. The loader rejects anything else. This
   removed a whole class of errors (a wrong digit in a narration) before the first render.
3. **Every equation cites its test.** `checked_by: test_name` must name an existing pytest
   function, so a reviewer can go from a formula on screen to the code that verifies it.
4. **Narration first, animation second.** Each chapter is synthesised as one continuous
   speech so the voice keeps its flow; beat boundaries are recovered from word timings and
   every animation beat is stretched to its narration. A beat that would run longer than its
   speech fails the build.
5. **One still per beat is inspected before a full render.** Rendering on a modest workstation
   is slow, so the pipeline renders a QA still at every beat boundary first and only then the
   full video. Stills are decoded sequentially to exact timestamps; seek-based grabs were
   found to land on keyframes and look blank.
6. **No LangChain, no LangGraph, no RAG.** The agent roles (profiler, historian, math author,
   math verifier, lesson writer, animation director, visual critic, experiment builder) share
   files, not a chat. Retrieval is reading the lesson file, the tested module and the skill
   rules. A vector database would add a dependency without adding a check.
7. **Verifier and critic never see the reasoning of the role they check.** The verifier gets
   the author's code, the critic gets rendered frames. Independence is what catches mistakes.
8. **Direction change, October 2026: AI first, history second.** Early lessons opened with
   the history of a concept. Viewer feedback showed the audience wants to know how the maths
   is used in AI/ML today, so history chapters were removed and every lesson now opens with a
   real decision (a loan, a diagnosis, a demand forecast) and the data behind it.
9. **The novelty rule.** Every lesson must (a) model a real public dataset on screen with the
   lesson's own mathematics, (b) show a which-model-when chart, (c) have a when-it-fails
   chapter with the failure demonstrated numerically, and (d) use enterprise, medical or
   peer-reviewed examples with citations. "What is a matrix?" is not a lesson title.
10. **Voice clean-up in the assembly step, not in the TTS.** Sibilance from the neural voice
    is removed with a denoise, de-ess, high-shelf and low-pass chain applied per clip, so the
    narration does not have to be re-synthesised.
11. **What is public and what is private.** Lessons, mathematics modules, tests, datasets,
    architecture documents and the website are public. The orchestration rules, narration
    timing, scene code, renderer and site generator stay private.
