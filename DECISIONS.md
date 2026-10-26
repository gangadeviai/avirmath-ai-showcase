# Development decisions

Short records of the product choices that shaped the system, in the order they were made.

1. **One lesson file is the single source of truth.** Video, captions, chapters, SEO pack and
   web page are all generated from `lessons/<id>/lesson.yaml`. Nothing is hand-edited downstream.
2. **Numbers are never typed into a lesson.** A lesson may only quote `{placeholders}` that a
   tested module publishes through `lesson_facts()`. The loader rejects anything else. This
   removed a whole class of errors (a wrong digit in a narration) before the first render.
3. **Every equation cites its test.** `checked_by: test_name` must name an existing pytest
   function, so a reviewer can go from a formula on screen to the code that verifies it.
4. **Direction change, October 2026: AI first, history second.** Early lessons opened with
   the history of a concept. Viewer feedback showed the audience wants to know how the maths
   is used in AI/ML today, so history chapters were removed and every lesson now opens with a
   real decision (a loan, a diagnosis, a demand forecast) and the data behind it.
5. **The novelty rule.** Every lesson must (a) model a real public dataset on screen with the
   lesson's own mathematics, (b) show a which-model-when chart, (c) have a when-it-fails
   chapter with the failure demonstrated numerically, and (d) use enterprise, medical or
   peer-reviewed examples with citations. "What is a matrix?" is not a lesson title.
6. **What is public and what is private.** Lessons, mathematics modules, tests, datasets,
   architecture overview and the website are public. The production pipeline (orchestration
   rules, narration and animation timing, scene code, renderer and site generator) is private.
